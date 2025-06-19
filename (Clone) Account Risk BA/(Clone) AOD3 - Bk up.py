# Databricks notebook source
import pyspark
import re
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession 
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import row_number 
from pyspark.sql.functions import date_format
from pyspark.sql.functions import lag, col,lit
from pyspark.sql.functions import datediff,date_add
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import to_date
from pyspark.sql.functions import abs
from pyspark.sql.types import StringType
from pyspark.sql.functions import monotonically_increasing_id

# COMMAND ----------

dir_payment_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/PAYMENT_BASE'
dir_bill_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/BILL_BASE'

df_payment_base = spark.read.format('delta').load(dir_payment_base)
df_bill_base = spark.read.format('delta').load(dir_bill_base)

# COMMAND ----------

# DBTITLE 1,fix on df_bill_base
df_bill_base = (df_bill_base
        .withColumn('latest', f.row_number().over(Window.partitionBy('account_no','bill_no').orderBy(f.desc('mod_t'))))
        .filter(f.col('latest')==1)     
        )
df_bill_base = df_bill_base.drop(col('latest'))

# COMMAND ----------

# DBTITLE 1,fix payment base
ls_col = [ 'ACCOUNT_NO', 'poid_id0', 'poid_type', 'CREATED_T', 'MOD_T','ACCOUNT_OBJ_ID0', 'EFFECTIVE_T', 'ITEM_TOTAL', 'OBJ_ID0'] 

df_payment_base = (
df_payment_base
.select(ls_col)
.withColumn('latest', f.row_number().over(Window.partitionBy('poid_id0').orderBy(f.desc('mod_t'))))
        .filter(f.col('latest')==1)
# .filter(col('poid_id0')=='1982087588818')
)
# acct no 349688317

# COMMAND ----------

vt_snapshot_date = '2024-03-03'

# COMMAND ----------

# look back at last 12 bills 
df_bill_12mo = (df_bill_base
         #.filter(col('account_no') =='337838335')
        .select('account_no','created_t','bill_no', 'due_t', 'end_t', 'closed_t', 'previous_total', 'total_due','due' )
        .withColumn('rnk_desc', f.row_number().over(Window.partitionBy('account_no').orderBy(f.desc('end_t'))))
        .filter(col('rnk_desc')<=12)
        .orderBy('end_t')
        )


# COMMAND ----------

 display(df_bill_12mo.filter(col('account_no') =='337838335'))

# COMMAND ----------

display(df_payment_base.filter(col('account_no') =='337838335'))

# COMMAND ----------

df_min_end_t = (
    df_bill_12mo
    .groupBy('account_no')
    .agg(f.min('end_t').alias('min_end_t'))
)


# COMMAND ----------

df_bill_credit_12mo = (
    df_bill_12mo
    .withColumn('total_charge', f.col('total_due')- f.col('previous_total'))
    .filter(col('total_charge') <0)
    .select('account_no', 'total_charge')
    .groupBy('account_no')
    .agg(f.sum('total_charge').alias('total_credit'))
    # .filter(col('account_no')=='349688317')
)


# COMMAND ----------

# filter out payment in the last 12 months after earliest bill end and before snapshot date 
#df_payment_12mo = 


df_payment_12mo = (
    df_bill_12mo.alias('a')
    .join(df_payment_base.alias('b'), col('a.account_no') == col('b.account_no') , 'left')
    .join(df_min_end_t.alias('c'), col('a.account_no') == col('c.account_no'), 'inner')
    .filter(col('b.created_t') >=  col('c.min_end_t') ) 
    .filter(col('b.created_t') <= lit(vt_snapshot_date))
    .select('a.account_no', 'b.created_t', 'b.item_total')
    # .filter(col('b.account_no') =='349688317')
    .distinct()
    .groupBy('account_no')
    .agg(f.sum('item_total').alias('payment_sum')
         , f.min('created_t').alias('min_pay_t')
         , f.max('created_t').alias('max_pay_t')
         )
    )

# get credit invoices - if customer change plans, would have an invoice with positive value --29/02/24 00:00:00	Invoice	635436799	acc-349688317

# df_bill_credit_12mo 

# COMMAND ----------

df_receive_12mo =(
    df_payment_12mo.alias('p')
    .join(df_bill_credit_12mo.alias('c'), col('p.account_no') == col('c.account_no'), 'left')
    .select('p.*', 'c.total_credit')
    .withColumn('total_receive', f.coalesce('payment_sum', lit(0)) + f.coalesce('total_credit', lit(0)))
   # .filter(col('p.account_no') == '349688317')
)

# COMMAND ----------

display(df_payment_12mo.filter(col('account_no') =='349688317'))

# COMMAND ----------

display(df_payment_12mo.filter(col('account_no') =='337838335'))

# COMMAND ----------

#windowSpec = Window.partitionBy('account_no').orderBy("bill_txn_date").rowsBetween(Window.unboundedPreceding, 0)
df_bills_payments = (
    df_bill_12mo.alias('a')
    .join(df_receive_12mo.alias('b'), col('a.account_no') == col('b.account_no'), 'left')
    .select('a.*', 'b.total_receive', 'b.min_pay_t', 'b.max_pay_t')
    .withColumn('rnk_asc', f.row_number().over(Window.partitionBy('account_no').orderBy('end_t')))         
    #.withColumn('TOTAL_CHARGE', f.col('TOTAL_DUE') - f.col('PREVIOUS_TOTAL'))
    .withColumn('TOTAL_CHARGE_Combine', f.when(col('rnk_asc') ==1, col('total_due'))
                .otherwise(f.col('TOTAL_DUE') - f.col('PREVIOUS_TOTAL'))   )
    .withColumn('cumulative_bills',  
                f.sum("TOTAL_CHARGE_Combine")
                    .over(Window.partitionBy('a.account_no')
                      .orderBy('end_t')
                      .rowsBetween(Window.unboundedPreceding, 0)))
    # .filter(col('a.account_no') =='337838335')
)

# COMMAND ----------

display(df_bills_payments.filter(col('account_no')=='337838335'))

# COMMAND ----------

df_bill_remian= (
     df_bills_payments
     .filter(col('TOTAL_CHARGE_Combine')>=0)
     .withColumn('excess', f.col('total_receive') + f.col('cumulative_bills'))
     .withColumn('bill_cleared_flag', f.when(f.col('excess')<=0, 1).otherwise(0))
     .withColumn('excess_to_next', 
                 lag('excess',1,0).over(Window.partitionBy('account_no').orderBy('end_t'))
     )
     .withColumn('true_access', 
                 # if the overall payment does not cover the earliest overdue bill at all 
                 f.when((col('excess_to_next')==0) & (col('TOTAL_CHARGE_Combine') > abs(f.col('total_receive')))
                                                      & (f.col('bill_cleared_flag')==0) , col('total_receive') )  # payment sum cannot pay first bill 
                 .when(col('excess_to_next') >0,0)
                 .otherwise(col('excess_to_next')))
     .withColumn('due_remain', 
                 f.when(col('bill_cleared_flag') ==1, 0)
                 .otherwise(f.col('TOTAL_CHARGE_Combine')+f.col('true_access')))
)


# COMMAND ----------

display(df_bill_remian.filter(col('account_no')=='337838335'))

# COMMAND ----------

df_bill_transform =(
    df_bill_remian.alias('a')
    .withColumn('od_days', datediff(lit(vt_snapshot_date),f.col('due_t')))
    .withColumn('age_of_debt_bucket', 
                f.when(col('od_days') <=0, 'Aod_Current')
                .when(col('od_days') <30,'Aod_01To30' )
                .when(col('od_days') <60 , 'Aod_31To60')
                .when(col('od_days') <90 , 'Aod_61To90')
                .when(col('od_days') <120, 'Aod_91To120')
                .when(col('od_days') < 150, 'Aod_121To150')
                .when(col('od_days') <180, 'Aod_151To180')
                .otherwise('Aod_181Plus')
                )
    .orderBy('od_days')
    )

# COMMAND ----------

df_aod = (df_bill_transform.groupBy('account_no',lit(vt_snapshot_date).alias('snapshot_date'))
        .pivot('age_of_debt_bucket')
        .agg(f.sum('due_remain'))
        .fillna(0)
        )
# display(df_aod) 

# COMMAND ----------

dbutils.fs.rm('dbfs:/mnt/ml-lab/dev_users/dev_sc/AOD',True)

# COMMAND ----------

dir_aod_output = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/AOD'
df_aod.write.format('delta').mode('overwrite').save(dir_aod_output)
df_aod = spark.read.format('delta').load(dir_aod_output)

# COMMAND ----------

dir_atb_report = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Coll_kpi_03032024221357.csv'  # 2024-03-03
df_atb_rpt = spark.read.csv(dir_atb_report,header = True)


# COMMAND ----------

df_joined = (df_aod
.join(df_atb_rpt, col('account_no') == col('Account Ref No'), 'inner')
)

# COMMAND ----------

dictionary_compaire = {
  'Aod_Current' : 'Aod Current',
 'Aod_01To30': 'Aod 01To30',
 'Aod_121To150': 'Aod 121To150',
 'Aod_151To180': 'Aod 151To180', 
 'Aod_181Plus': 'Aod 181Plus', 
 'Aod_31To60':  'Aod 31To60',
 'Aod_61To90': 'Aod 61To90' ,
 'Aod_91To120': 'Aod 91To120' }

# COMMAND ----------

# MAGIC %run "./Function"

# COMMAND ----------

df_result = check_similarity(dataframe=df_joined, dict_pairs=dictionary_compaire, value_adj=1, threshold=0.95, excl_zero = True)

# COMMAND ----------

display(
    df_result
    .groupBy('fs_col','benchmark_col')
    .agg(
        f.count('*').alias('count')
        , f.sum('similar_flag').alias('align')
    )
    .withColumn(
        'rate'
        , f.col('align')/f.col('count')
    )
    .withColumn('not_align', f.col('count') - f.col('align'))
)


# COMMAND ----------

df_diff = (df_joined.
        withColumn('diff', f.col('Aod_01To30').cast('double') - f.col('Aod 01To30').cast('double') )
        .select('account_no', 'snapshot_date', 'Aod_Current', 'Aod_01To30', 'Aod_31To60', 'Aod_61To90', 
                'Aod Current', 'Aod 01To30',  'Aod 31To60',    'Aod 61To90'   )
        .filter(f.col('diff') >=10) 
        .filter(f.col('Segment') =='CONSUMER')
        .distinct()
        )

display(df_diff.limit(100))
# 342617884 missing adjustment record in item_t cdc 
# 343173107 missing adjustment 

# COMMAND ----------

display(df_bill_transform.filter(col('account_no')=='343173107'))

# COMMAND ----------

display(df_payment_base.filter(col('account_no')=='343173107'))

# COMMAND ----------


