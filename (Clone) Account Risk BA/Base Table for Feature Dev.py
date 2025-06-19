# Databricks notebook source
# MAGIC %md
# MAGIC ### S0001 Set Up

# COMMAND ----------

# DBTITLE 1,Library
import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession 
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import row_number 
from pyspark.sql.functions import date_format
from pyspark.sql.functions import lag, col
from pyspark.sql.functions import datediff
from pyspark.sql.functions import regexp_replace 

# COMMAND ----------

dbutils.fs.ls('/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Source Directory

# COMMAND ----------

# DBTITLE 1,bill and payment source table
# source table directory 
dir_bill_t = '/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_BILL_T' 
dir_item_t = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_ITEM_T' # item_t after partition 
#dir_item_t = '/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T' # raw pinpap item t 
dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
dir_acct = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T'
dir_event_bill_payment_t = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T'
dir_event_bal_impact_t = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T'
dir_config_pay_type = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_PAY_TYPES_T'
dir_config_channel_map = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_CHANNEL_MAP_T'


# df load with lates record 
# use _is_latest  = 1  and _is_deleted = 0 to dedupe record --- need further refine because there is duplicate for invoice level  
df_oa_consumer = spark.read.format('delta').load(dir_oa_consumer)
df_bill_t = spark.read.format('delta').load(dir_bill_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_item_t = spark.read.format('delta').load(dir_item_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_account_t = spark.read.format('delta').load(dir_acct).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_event_bal_impact_t = spark.read.format('delta').load(dir_event_bal_impact_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_event_bill_payment_t = spark.read.format('delta').load(dir_event_bill_payment_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_config_pay_type = spark.read.format('delta').load(dir_config_pay_type).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_config_channel_map = spark.read.format('delta').load(dir_config_channel_map).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### OA Comsumer 

# COMMAND ----------

# DBTITLE 1,FS OA Consumer
df_oa_consumer = (df_oa_consumer
        .select('fs_cust_id', 'fs_acct_id')
        .filter(f.col('reporting_date')>='2021-01-01')
        .distinct())
 
#  746K ACCOUNT SINCE 2021 

# COMMAND ----------

# DBTITLE 1,Run Function
# MAGIC %run "./Function"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Account T

# COMMAND ----------

# DBTITLE 1,Account T
df_account_t = df_account_t.select('poid_id0', 'account_no')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bill T 

# COMMAND ----------

# DBTITLE 1,Billing Base Table Selection
ls_col = ['POID_ID0',
'CREATED_T',
'MOD_T',
'ACCOUNT_OBJ_ID0',
'INVOICE_OBJ_ID0',
'BILL_NO',
'CURRENCY',
'DUE_T',
'END_T',
'PREVIOUS_TOTAL',
'START_T',
'SUBORDS_TOTAL',
'TOTAL_DUE',
'ADJUSTED',
'DUE',
'RECVD',
'WRITEOFF',
'TRANSFERRED',
'BILLINFO_OBJ_ID0',
'CLOSED_T',
'CURRENT_TOTAL',
'AR_BILLINFO_OBJ_ID0'
]

df_bill_base = (
    df_bill_t.alias('b')
    .join(df_account_t.alias('a'),f.col('b.ACCOUNT_OBJ_ID0') == f.col('a.poid_id0'))
    .select(
        *[f.col('b.'+ col_name) for col_name in ls_col] +
        [f.col('a.account_no')]
    )
)


#display(df_bill_base.limit(3))

# COMMAND ----------

# DBTITLE 1,Convert Epoch to NZT
ls_time_convert = [c for c in df_bill_base.columns if c.endswith('_T')]
df_bill_base = to_nzt(df_bill_base, ls_time_convert)

# COMMAND ----------

# DBTITLE 1,Bill Base Transformation
df_bill_base = (
    df_bill_base
    # .filter(f.col('END_T') >= '2021-01-01')
    .filter(~f.col('account_no').startswith('S'))   # remove susbscription account 
    .withColumn('account_no', regexp_replace("account_no", "^0+", "")) #remove leading 0 
    .withColumn('due_month', date_format('due_t', "yyyy-MM"))
  )  
 

# COMMAND ----------

# check mapping # still 594 cust not mapped - actually has no bill - maybe new joiner in OA cons
display(
    df_oa_consumer
    .join(df_bill_base, f.col('account_no') == f.col('fs_acct_id'), 'anti')
    .select('fs_acct_id')
    .distinct()
    .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export Bill Base Table

# COMMAND ----------

# DBTITLE 1,Export Bill Base Table and Add Partition
# display(df_bill_base.count())  # 34,388,450
dir_bill_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/BILL_BASE'

export_data(
    df = df_bill_base
    , export_path = dir_bill_base
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = True
    , ls_dynamic_partition= ["due_month"]
)


# COMMAND ----------

df_bill_base  = spark.read.format('delta').load(dir_bill_base)
df_bill_base = df_bill_base.drop("due_month")

# COMMAND ----------

# DBTITLE 1,Check outcome
display(df_bill_base.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Payment Base

# COMMAND ----------

# DBTITLE 1,Payment Base Fields
ls_col_aa = ['POID_ID0', 'POID_TYPE', 'CREATED_T', 'MOD_T', 'ACCOUNT_OBJ_ID0', 'EFFECTIVE_T', 'ITEM_NO',  'ITEM_TOTAL', 'BILLINFO_OBJ_ID0']
df_payment = (
    df_item_t
    .select(ls_col_aa)
    .filter(f.col('poid_type').isin('/item/payment', '/item/adjustment'))
)



# COMMAND ----------

# DBTITLE 1,Make Sure Select Adjustment
display(df_payment.select('poid_type').distinct())

# COMMAND ----------

# DBTITLE 1,Convert Item_T epoch time to NZT
ls_time_convert = [c for c in df_payment.columns if c.endswith('_T')]
#ls_time_convert
df_payment = to_nzt(df_payment, ls_time_convert)
# df_bill_base = to_nzt(df_bill_base, ls_time_convert)

# COMMAND ----------

# DBTITLE 1,Field Select for event bal impact t
ls_col_a = ['OBJ_ID0', 'ACCOUNT_OBJ_ID0', 'AMOUNT', 'ITEM_OBJ_ID0', 'ITEM_OBJ_TYPE']
df_bal_impact  = (
    df_event_bal_impact_t
    .filter(f.col('ITEM_OBJ_TYPE').isin('/item/payment','/item/adjustment'))
    .select(ls_col_a)
)


# COMMAND ----------

# DBTITLE 1,Field Select for bill payment t
ls_col_b = ['OBJ_ID0', 'AMOUNT', 'TRANS_ID', 'PAY_TYPE', 'CHANNEL_ID', 'ACCOUNT_NO']
df_bill_payment = (
df_event_bill_payment_t
.select(ls_col_b)
)

# COMMAND ----------

# DBTITLE 1,Config
ls_col_c = ['REC_ID','PAYINFO_TYPE', 'PAYMENT_EVENT_TYPE', 'REFUND_EVENT_TYPE']
df_config_pay_type = df_config_pay_type.select(ls_col_c)

# COMMAND ----------

# DBTITLE 1,Config
ls_col_d = ['CHANNEL_ID','SOURCE','SUBTYPE','TYPE']
df_config_channel_map = df_config_channel_map.select(ls_col_d)

# COMMAND ----------

# DBTITLE 1,Final Column Selection
combined_unique_list = list(dict.fromkeys(ls_col_aa + ls_col_a + ls_col_b + ls_col_c + ls_col_d))
print(combined_unique_list)
#ls_col = ['b.OBJ_ID0', 'a.ACCOUNT_OBJ_ID0', 'b.AMOUNT', 'ITEM_OBJ_ID0', 'ITEM_OBJ_TYPE', 'TRANS_ID', 'PAY_TYPE', 'c.CHANNEL_ID', 'aa.ACCOUNT_NO', 'REC_ID', 'PAYINFO_TYPE', 'PAYMENT_EVENT_TYPE', 'REFUND_EVENT_TYPE', 'SOURCE', 'SUBTYPE', 'TYPE']

ls_col = ['a.POID_ID0', 'POID_TYPE', 'CREATED_T', 'MOD_T', 'a.ACCOUNT_OBJ_ID0', 'EFFECTIVE_T', 'ITEM_NO', 'ITEM_TOTAL', 'BILLINFO_OBJ_ID0', 'b.OBJ_ID0', 'b.AMOUNT', 'ITEM_OBJ_ID0', 'ITEM_OBJ_TYPE', 'TRANS_ID', 'PAY_TYPE', 'c.CHANNEL_ID', 'aa.ACCOUNT_NO', 'REC_ID', 'PAYINFO_TYPE', 'PAYMENT_EVENT_TYPE', 'REFUND_EVENT_TYPE', 'SOURCE', 'SUBTYPE', 'TYPE']

# COMMAND ----------

# DBTITLE 1,Payment Base Table
df_payment_base = (df_payment.alias('a')
    .join(df_account_t.alias('aa'), f.col('a.account_obj_id0') == f.col('aa.poid_id0'), 'inner')
    .join(df_bal_impact.alias('b'),f.col('a.poid_id0') ==f.col('b.item_obj_id0'),'left')
    .join(df_bill_payment.alias('c'), f.col('b.obj_id0') ==f.col('c.obj_id0'),'left')
    .join(df_config_pay_type.alias('pt'), f.col('c.pay_type')==f.col('pt.rec_id'),'left')
    .join(df_config_channel_map.alias('cm'), f.col('c.channel_id')==f.col('cm.channel_id'), 'left')
    .select(ls_col)
    .withColumn('Created_month', date_format('created_t', 'yyyy-MM'))
)


# COMMAND ----------

# DBTITLE 1,Export  Payment and Partition
dir_payment_base = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/PAYMENT_BASE'

export_data(
    df = df_payment_base
    , export_path = dir_payment_base
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = True
    , ls_dynamic_partition= ["Created_month"]
)



# COMMAND ----------

# DBTITLE 1,Reload
df_payment_base = spark.read.format('delta').load(dir_payment_base).drop('created_month')

# COMMAND ----------

display(df_payment_base.limit(100))
