# Databricks notebook source
# MAGIC %md
# MAGIC ###S0001 Set up 

# COMMAND ----------

# DBTITLE 1,Library
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
from pyspark.sql.functions import datediff
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import to_date
from pyspark.sql.functions import abs
from pyspark.sql.types import StringType
from pyspark.sql.functions import monotonically_increasing_id

# COMMAND ----------

# DBTITLE 1,Directory
dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
#dir_atb_report = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Coll_kpi_03032024221357.csv'
#dir_finan_report =  'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01032024053404.csv'
#dir_finan_report_2023 = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/FinanceReport_01052023102202.csv'

dir_payment_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/PAYMENT_BASE'
dir_bill_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/BILL_BASE'

df_payment_base = spark.read.format('delta').load(dir_payment_base)
df_bill_base = spark.read.format('delta').load(dir_bill_base)
df_oa_consumer = spark.read.format('delta').load(dir_oa_consumer)
#df_atb_rpt = spark.read.csv(dir_atb_report,header = True)
#df_finan_report = spark.read.csv(dir_finan_report, header = True)
#df_finan_report_2023 = spark.read.csv(dir_finan_report_2023, header = True)

# COMMAND ----------

# DBTITLE 1,run function
# MAGIC %run "./Function"

# COMMAND ----------

# DBTITLE 1,Fix Bil T duplicate
df_bill_base = (df_bill_base
        .withColumn('latest', f.row_number().over(Window.partitionBy('account_no','bill_no').orderBy(f.desc('mod_t'))))
        .filter(f.col('latest')==1)     
        )

# COMMAND ----------

# DBTITLE 1,Load Consumer
df_oa_consumer = (
    df_oa_consumer
    .select('fs_cust_id','fs_acct_id')
    .filter(f.col('reporting_date') >='2021-01-01') 
    .distinct()
)

# 746521

# COMMAND ----------

# DBTITLE 1,Get OA Bill
df_oa_bill = (
    df_bill_base
    .join(df_oa_consumer, f.col("account_no") == f.col('fs_acct_id'), 'inner' )
    .drop('latest')
)

# COMMAND ----------

# DBTITLE 1,Cnt of OA Bill
display(df_oa_bill
        .select('account_no')
        .distinct()
        .count()
        ) 
# 20,269,186 rows 
# 745622 accunts 

# COMMAND ----------

# DBTITLE 1,Get OA Payment
df_oa_payment = (
    df_payment_base
    .join(df_oa_consumer, f.col("account_no") ==f.col("fs_acct_id"),'inner')
    .withColumn('POID_ID0', f.col('POID_ID0').cast(StringType()))
)

# COMMAND ----------

# DBTITLE 1,Cnt of OA Payment
# 686479 for consumer account  # 20,677,955 rows  
display(df_oa_payment.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### S0002 Get Transaction Table

# COMMAND ----------

# DBTITLE 1,Get Relavent Field for Pay
ls_pay_col = ['ACCOUNT_NO','ITEM_TOTAL','CREATED_T','POID_ID0','POID_TYPE']
df_oa_payment = (
    df_oa_payment
    .select(ls_pay_col)
)

# COMMAND ----------

# DBTITLE 1,Get Relavent Field for Bill
ls_bill_col = ['ACCOUNT_NO','TOTAL_CHARGE', 'END_T', 'BILL_NO', 'TYPE']
df_oa_bill = (
    df_oa_bill
    .withColumn('TOTAL_CHARGE', f.col('TOTAL_DUE') - f.col('PREVIOUS_TOTAL'))
    .withColumn('TYPE', lit('Bill'))
    .select(ls_bill_col)
)


# COMMAND ----------

display(df_oa_bill.filter(f.col('account_no') =='1000008'))

# COMMAND ----------

# DBTITLE 1,Column Rename
ls_col_rename = ['ACCOUNT_NO', 'AMOUNT', 'TXN_DATE', 'TXN_KEY','TYPE']

for old, new in zip(ls_bill_col, ls_col_rename):
    df_oa_bill = df_oa_bill.withColumnRenamed(old, new)

for old, new in zip(ls_pay_col, ls_col_rename):
    df_oa_payment = df_oa_payment.withColumnRenamed(old, new)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Processing

# COMMAND ----------

# DBTITLE 1,Union Result
df_txn = df_oa_bill.union(df_oa_payment).orderBy('account_no', 'txn_date')

# COMMAND ----------

# DBTITLE 1,cnt of union
display(df_txn.count()) # 

# COMMAND ----------

# DBTITLE 1,find directory
dbutils.fs.ls('/mnt/ml-lab/dev_users/dev_sc/')

# COMMAND ----------

# DBTITLE 1,add partition key
df_txn = (df_txn.withColumn('txn_month', date_format('txn_date', 'yyyy-MM')))

# COMMAND ----------

# DBTITLE 1,get one example
display(df_txn.limit(1))

# COMMAND ----------

# DBTITLE 1,export delta
dir_export = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/TXN/'
export_data(
    df = df_txn
    , export_path = dir_export
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = True
    , ls_dynamic_partition=["txn_month"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Development

# COMMAND ----------

# DBTITLE 1,reload txn
df_txn = spark.read.format('delta').load('dbfs:/mnt/ml-lab/dev_users/dev_sc/TXN/')

# COMMAND ----------

#vt_account_no = '496357249'
#vt_account_no = '491356828'  #latest record happened to have 1st occurance of 0 # fix with latest_od_flag in (0,1)
#vt_account_no = '490593184'
#vt_account_no = '480296899'
#vt_account_no = '500923472'
vt_account_no = '501397171'  # Pass 
vt_snapshot_date = '2024-03-03'

# COMMAND ----------

# DBTITLE 1,Check 1 Account
display(df_txn.filter(f.col('account_no') ==lit(vt_account_no)).distinct())

# COMMAND ----------

# DBTITLE 1,pick one account for test
df_txn_test = (df_txn
        .filter(f.col('account_no')==lit(vt_account_no))
        .distinct()
        .withColumn('overall balance', f.sum("amount")
                    .over(Window.partitionBy('account_no')
                          .orderBy(f.desc('txn_date'))
                          .rowsBetween( Window.currentRow, Window.unboundedFollowing)
                          )
                    )
        
        )

        # 490593184

# COMMAND ----------

# DBTITLE 1,check one account txn
display(df_txn_test.filter(f.col('account_no')==lit(vt_account_no)).orderBy(f.desc('txn_date')))

# COMMAND ----------

# DBTITLE 1,filter out latest records with overdue balance

df_txn_transform_test = (
  df_txn_test
  .distinct()
  # make a flag to identify orrcurance of credit balance or 0 balacen 
  .withColumn('credit_flag', f.when(f.col('overall balance')<=0, 1).otherwise(0)) 
  # cumsum of credit flag to identify rows after credit balance 
  .withColumn('latest_od_flag', f.sum('credit_flag').over(Window.orderBy(f.desc('txn_date')).rangeBetween(Window.unboundedPreceding,0)))
  # fitler out trasaction before snapshot 
  .filter(f.col('txn_date')<= lit(vt_snapshot_date))
  # filter out the problem area - where the overdue balance start to accumulate 
  #.filter(f.col('latest_od_flag') ==0)
  .filter(f.col('latest_od_flag').isin(0,1))
 )
# .filter(f.col('txn_date')<= lit(vt_snapshot_date))

# COMMAND ----------

display(df_txn_transform_test)

# COMMAND ----------

df_close = (df_bill_base
.filter(f.col('account_no') == lit(vt_account_no))
.select('end_t', 'closed_t', 'bill_no', 'due_t', 'account_no')
)

# COMMAND ----------

display(df_close)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Payment Allocation

# COMMAND ----------

bills_df = df_txn_transform_test.filter(df_txn_transform_test['TYPE'] == 'Bill')
bills_df = bills_df.withColumnRenamed("TXN_DATE", "bill_txn_date").orderBy('bill_txn_date')
payments_df = df_txn_transform_test.filter(df_txn_transform_test['TYPE'].isin(['/item/payment','/item/adjustment']))
payments_df = payments_df.withColumnRenamed("TXN_DATE", "payment_txn_date").orderBy('payment_txn_date')

# COMMAND ----------

display(payments_df)

# COMMAND ----------

# Define a window spec to sum payments up to the current row
windowSpec = Window.orderBy("bill_txn_date").rowsBetween(Window.unboundedPreceding, 0)

# Calculate the cumulative sum of bill invoice 
bills_df = bills_df.withColumn("cumulative_bills", f.sum("AMOUNT").over(windowSpec))


# COMMAND ----------

# DBTITLE 1,Get Transfer Credit
# get transfer plan credit 
vt_tranfer_credit = (
    bills_df
    .filter(f.col('amount')<0)
    .select('amount')
    .agg(f.sum('amount').alias('sum_amount')) 
    .collect()[0]['sum_amount']
    )

vt_tranfer_credit = 0 if vt_tranfer_credit is None else vt_tranfer_credit

# COMMAND ----------

# DBTITLE 1,Get Payment Amount
if payments_df.rdd.isEmpty():
    vt_total_pay = 0 
else: 
    vt_total_pay = payments_df.select(f.sum('amount')).collect()[0][0] 


 

# COMMAND ----------

# DBTITLE 1,Get Total Credit
vt_total_credit = vt_total_pay + vt_tranfer_credit

# COMMAND ----------

# DBTITLE 1,Print Total Credit
print(vt_total_credit)

# COMMAND ----------

# DBTITLE 1,Get Bill Overdue Remain Amount
df_bill_remian= (
     bills_df
     .filter(col('amount')>=0)
     .withColumn('excess', f.lit(vt_total_credit) + f.col('cumulative_bills'))
     .withColumn('bill_cleared_flag', f.when(f.col('excess')<=0, 1).otherwise(0))
     .withColumn('excess_to_next', 
                 lag('excess',1,0).over(Window.partitionBy('account_no').orderBy('bill_txn_date'))
     )
     .withColumn('true_access', 
                 # if the overall payment does not cover the earliest overdue bill at all 
                 f.when((col('excess_to_next')==0) & (col('cumulative_bills') > abs(lit(vt_total_credit))), lit(vt_total_credit) ) 
                 .when(col('excess_to_next') >0,0)
                 .otherwise(col('excess_to_next')))
     .withColumn('due_remain', 
                 f.when(col('bill_cleared_flag') ==1, 0)
                 .otherwise(f.col('amount')+f.col('true_access')))
     
     #.withColumn('bill_rnk_asc', f.row_number().over(Window.partitionBy('account_no').orderBy('bill_txn_date')))
    # .drop('excess_to_next','excess','credit_flag', 'latest_od_flag','txn_month')
 )

# COMMAND ----------

# DBTITLE 1,Check Bill Remain Table
display(df_bill_remian)

# COMMAND ----------

# DBTITLE 1,Transform Age Bucket
df_bill_transform =(
    df_bill_remian.alias('a')
    .join(df_close.alias('b'), (f.col('a.account_no') == f.col('b.account_no')) & (f.col('bill_no') == f.col('txn_key')),'inner') 
    .select('a.*', 'b.due_t', 'b.closed_t')
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

display(df_bill_transform)

# COMMAND ----------

# DBTITLE 1,Group By and Pivot to AOD Amount
df_aod = (df_bill_transform.groupBy('account_no',lit(vt_snapshot_date).alias('snapshot_date'))
        .pivot('age_of_debt_bucket')
        .agg(f.sum('due_remain'))
        )

display(df_aod) 
