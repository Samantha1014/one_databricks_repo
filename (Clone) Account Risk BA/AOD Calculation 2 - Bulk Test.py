# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Set Up

# COMMAND ----------

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



# COMMAND ----------

# dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
dir_atb_report = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Coll_kpi_03032024221357.csv'  # 2024-03-03
dir_finan_report =  'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01032024053404.csv' # 2024-03-01
dir_finan_report_2023 = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01052023102202.csv' # 2023-05-01

#dir_payment_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/PAYMENT_BASE'
dir_bill_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/BILL_BASE'

#df_payment_base = spark.read.format('delta').load(dir_payment_base)
df_bill_base = spark.read.format('delta').load(dir_bill_base)
#df_oa_consumer = spark.read.format('delta').load(dir_oa_consumer)
df_atb_rpt = spark.read.csv(dir_atb_report,header = True)
df_finan_report = spark.read.csv(dir_finan_report, header = True)
df_finan_report_2023 = spark.read.csv(dir_finan_report_2023, header = True)

# COMMAND ----------

# DBTITLE 1,Load Transaction Table
df_txn = spark.read.format('delta').load('dbfs:/mnt/ml-lab/dev_users/dev_sc/TXN/')

# COMMAND ----------

# DBTITLE 1,Set snapshot date
vt_snapshot_date = '2024-03-03'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transform on Transaction Table

# COMMAND ----------

# DBTITLE 1,make sure all started with bill
df_txn = (df_txn
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_no').orderBy('txn_date')))
        .withColumn('rm_flg', f.when( (col('type').isin('/item/payment', '/item/adjustment')) &(col('rnk')==1), 1)
                    .otherwise(0))
        .filter(col('rm_flg') ==0)
        .drop('rnk', 'rm_flg')
)
        

# COMMAND ----------

# DBTITLE 1,check on accnt
display(df_txn.filter(col('account_no')=='334700082'))

# COMMAND ----------

#display(df_txn.limit(10))
#
#display(df_txn.filter(f.col('account_no') =='490011852'))

# COMMAND ----------

# DBTITLE 1,join with bill base
# total due in bill_t is the overall balance for its invoice
df_txn = (df_txn.alias('a')
        .join(df_bill_base.alias('b'),
               (f.col('a.txn_key') ==f.col('b.bill_no')) & (f.col('a.account_no') == f.col('b.account_no')), 'left' )
        #.filter(col('a.account_no') =='334700082')
        .select('a.*', 'b.total_due')
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_no').orderBy('txn_date')))
        .withColumn('intial_amount_fix', f.when(col('rnk') ==1, col('total_due')).otherwise(col('amount'))
                    )
        .drop('rnk','txn_month')
        )

# COMMAND ----------

# display(df_txn.filter(f.col('account_no') =='490011852'))

# COMMAND ----------

df_txn_transform = (df_txn
       #  .filter(f.col('account_no')==lit(vt_account_no))
        .distinct()
        .withColumn('overall balance', f.sum("intial_amount_fix")
                    .over(Window.partitionBy('account_no')
                          .orderBy(f.desc('txn_date'))
                          .rowsBetween( Window.currentRow, Window.unboundedFollowing)
                          )
                    )
        
        )


# COMMAND ----------

# DBTITLE 1,check one account
# display(df_txn_transform.filter(f.col('account_no')=='501397171').orderBy(f.desc('txn_date')))

# COMMAND ----------

# DBTITLE 1,Get Overdue Records
df_txn_transform_test = (
  df_txn_transform
  .distinct()
  # make a flag to identify orrcurance of credit balance or 0 balacen 
  .withColumn('credit_flag', f.when(f.col('overall balance')<=0, 1).otherwise(0)) 
  # cumsum of credit flag to identify rows after credit balance 
  .withColumn('latest_od_flag', f.sum('credit_flag').over(Window.partitionBy('account_no').orderBy(f.desc('txn_date')).rangeBetween(Window.unboundedPreceding,0)))
  # fitler out trasaction before snapshot 
  .filter(f.col('txn_date')<= lit(vt_snapshot_date))
  # filter out the problem area - where the overdue balance start to accumulate 
  #.filter(f.col('latest_od_flag') ==0)
  .filter(f.col('latest_od_flag').isin(0,1))
 )

# COMMAND ----------

 display(df_txn_transform_test.filter(col('account_no') =='1013106')) 
 # this account maybe have miss payment record in cdc 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Payment Allocation

# COMMAND ----------

df_bills = df_txn_transform_test.filter(df_txn_transform_test['TYPE'] == 'Bill')
df_bills = df_bills.withColumnRenamed("TXN_DATE", "bill_txn_date").orderBy('bill_txn_date')
df_payments = df_txn_transform_test.filter(df_txn_transform_test['TYPE'].isin(['/item/payment','/item/adjustment']))
df_payments = df_payments.withColumnRenamed("TXN_DATE", "payment_txn_date").orderBy('payment_txn_date')

# COMMAND ----------

# DBTITLE 1,Transform Bill F
# Define a window spec to sum payments up to the current row
windowSpec = Window.partitionBy('account_no').orderBy("bill_txn_date").rowsBetween(Window.unboundedPreceding, 0)

# Calculate the cumulative sum of bill invoice 
df_bills = df_bills.withColumn("cumulative_bills", f.sum("AMOUNT").over(windowSpec))

# COMMAND ----------

display(df_bills.filter(col('account_no') =='1013106'))

# COMMAND ----------

# get transfer plan credit 
df_tranfer_credit_by_account = (
    df_bills
    .filter(f.col('amount') < 0)
    .groupBy('account_no')  
    .agg(f.coalesce(f.sum('amount'), f.lit(0)).alias('transfer_credit'))
)
#display(df_tranfer_credit_by_account.limit(3))

# COMMAND ----------

# DBTITLE 1,Get Payment Amount
df_pay_by_account = (
    df_payments
    .groupBy('account_no')  
    .agg(f.coalesce(f.sum('amount'), f.lit(0)).alias('total_pay'))
)
# display(df_pay_by_account.limit(3))



# COMMAND ----------

# DBTITLE 1,Get Total Credit
df_total_credit = (
    df_pay_by_account.alias('a')
    .join(df_tranfer_credit_by_account.alias('b')
          , col('a.account_no') ==col('b.account_no'), 'left')
    .select('a.*', 'b.transfer_credit')
    .withColumn('total_credit', 
                f.coalesce('transfer_credit',lit(0)) + 
                 f.coalesce('total_pay',lit(0)))
    
)

#display(df_total_credit.limit(3))

# COMMAND ----------

# DBTITLE 1,Get Bill and Payment in One
df_bills_payments = (
    df_bills.alias('b')
    .join(df_total_credit.alias('c'), col('b.account_no') == col('c.account_no'), 'left')
    .select('b.*', 'c.total_credit')
)

# COMMAND ----------

# DBTITLE 1,Get Bill Overdue Remain Amount
df_bill_remian= (
     df_bills_payments
     .filter(col('amount')>=0)
     .withColumn('excess', f.col('total_credit') + f.col('cumulative_bills'))
     .withColumn('bill_cleared_flag', f.when(f.col('excess')<=0, 1).otherwise(0))
     .withColumn('excess_to_next', 
                 lag('excess',1,0).over(Window.partitionBy('account_no').orderBy('bill_txn_date'))
     )
     .withColumn('true_access', 
                 # if the overall payment does not cover the earliest overdue bill at all 
                 f.when((col('excess_to_next')==0) & (col('cumulative_bills') > abs(f.col('total_credit')) ), col('total_credit') ) 
                 .when(col('excess_to_next') >0,0)
                 .otherwise(col('excess_to_next')))
     .withColumn('due_remain', 
                 f.when(col('bill_cleared_flag') ==1, 0)
                 .otherwise(f.col('amount')+f.col('true_access')))
)
     #.withColumn('bill_rnk_asc', f.row_number().over(Window.partitionBy('account_no').orderBy('bill_txn_date')))
    # .drop('excess_to_next','excess','credit_flag', 'latest_od_flag','txn_month')
 

# COMMAND ----------

# DBTITLE 1,Transform Age Bucket
df_bill_transform =(
    df_bill_remian.alias('a')
    .withColumn('due_date', date_add(col('bill_txn_date'),16))
    .withColumn('od_days', datediff(lit(vt_snapshot_date),f.col('due_date')))
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

# DBTITLE 1,Group By and Pivot
df_aod = (df_bill_transform.groupBy('account_no',lit(vt_snapshot_date).alias('snapshot_date'))
        .pivot('age_of_debt_bucket')
        .agg(f.sum('due_remain'))
        )
# display(df_aod) 

# COMMAND ----------

dir_aod_output = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/AOD'
df_aod.write.format('delta').save(dir_aod_output)
df_aod = spark.read.format('delta').load(dir_aod_output)

# COMMAND ----------



# COMMAND ----------

df_aod = df_aod.fillna(0)

# COMMAND ----------

display(df_atb_rpt.limit(3))

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
)


# COMMAND ----------

df_diff = (df_joined.
        withColumn('diff', f.col('Aod_01To30').cast('double') - f.col('Aod 01To30').cast('double') )
        .filter(f.col('diff') >=10) 
        .filter(f.col('Segment') =='CONSUMER')
        .distinct()
        )

display(df_diff.limit(100))

# COMMAND ----------

display(df_bill_transform.filter(col('account_no') =='337838335'))

# COMMAND ----------

display(df_txn.filter(col('account_no') =='337838335'))

# COMMAND ----------

# check item_t 
dir_item_t = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_ITEM_T'
df_item_t = spark.read.format('delta').load(dir_item_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))



# COMMAND ----------

display(df_item_t
        .filter(col('account_obj_id0') =='101173065002')
        .filter(col('poid_type') == '/item/payment')
        )


# COMMAND ----------

dir_payment_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/PAYMENT_BASE'
df_payment_base = spark.read.format('delta').load(dir_payment_base)

# COMMAND ----------

display(df_payment_base.filter(col('account_no') =='337838335'))

# COMMAND ----------

display(df_txn.filter(col('account_no')=='337838335'))
