# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp, date_format
from pyspark.sql.functions import regexp_replace 

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Directory

# COMMAND ----------

# source table directory 
# dir_item_t_test = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_ITEM_T' # item_t after partition 
dir_item_t = '/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T' # raw pinpap item t 
#dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
dir_acct = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T'
dir_event_bill_payment_t = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T'
dir_event_bal_impact_t = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T'
dir_config_pay_type = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_PAY_TYPES_T'
dir_config_channel_map = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_CHANNEL_MAP_T'
dir_prim_oa = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'


# COMMAND ----------

 #df_prim_oa = spark.read.format('delta').load(dir_prim_oa)

# COMMAND ----------

# df_prim_oa_01 = (
#     df_prim_oa
#     .select('fs_acct_id')
#     .filter(col('reporting_date')>='2021-01-01')
#     .distinct()

# )

# COMMAND ----------

display(df_item_t.count()) # 4,122,249,458

# COMMAND ----------

# DBTITLE 1,Load Table
# df_oa_consumer = spark.read.format('delta').load(dir_oa_consumer)
# df_bill_t = spark.read.format('delta').load(dir_bill_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_item_t = spark.read.format('delta').load(dir_item_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_account_t = spark.read.format('delta').load(dir_acct).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_event_bal_impact_t = spark.read.format('delta').load(dir_event_bal_impact_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_event_bill_payment_t = spark.read.format('delta').load(dir_event_bill_payment_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_config_pay_type = spark.read.format('delta').load(dir_config_pay_type).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_config_channel_map = spark.read.format('delta').load(dir_config_channel_map).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00003 Select Payment Required Fields

# COMMAND ----------

df_account_t = (
    df_account_t
    .filter(col('_is_latest') ==1) # pick latest record
    .filter(col('_is_deleted') ==0) 
    .filter(~col('account_no').startswith('S')) # remove subscription account 
    .withColumn('account_no', regexp_replace("account_no", "^0+", "")) # remove leading 0 in account_t 
    .select('account_no', 'poid_id0')
)


# COMMAND ----------

ls_col_payment = ['POID_ID0', 'POID_TYPE', 'CREATED_T', 'MOD_T', 'ACCOUNT_OBJ_ID0', 'EFFECTIVE_T', 'ITEM_NO',  'ITEM_TOTAL', 'BILLINFO_OBJ_ID0']
df_payment = (
    df_item_t
    .select(ls_col_payment)
    .filter(f.col('poid_type').isin('/item/payment', '/item/adjustment'))
)


# COMMAND ----------

display(df_payment.count()) #118,455,286

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,join oa consumer
df_oa_payment =  (df_payment.alias('a')
        .join(df_account_t.alias('b'), col('a.account_obj_id0') == col('b.poid_id0'), 'inner' )
        .join(df_prim_oa_01, col('account_no') == col('fs_acct_id'), 'right')
        .drop('account_no')
) 

# 50,092,410 rows 
# 743,383 (inner) vs. 759457 (right)

# COMMAND ----------

# DBTITLE 1,Convert to NZT
ls_time_convert = [c for c in df_payment.columns if c.endswith('_T')]
#ls_time_convert
df_payment = to_nzt(df_payment, ls_time_convert)

# COMMAND ----------

ls_col_impact = ['OBJ_ID0', 'ACCOUNT_OBJ_ID0', 'AMOUNT', 'ITEM_OBJ_ID0', 'ITEM_OBJ_TYPE']

df_bal_impact  = (
    df_event_bal_impact_t
    .filter(f.col('ITEM_OBJ_TYPE').isin('/item/payment'))
    .select(ls_col_impact)
)


ls_col_billpay = ['OBJ_ID0', 'AMOUNT', 'TRANS_ID', 'PAY_TYPE', 'CHANNEL_ID', 'ACCOUNT_NO']
df_bill_payment = (
df_event_bill_payment_t
.select(ls_col_billpay)
)



# COMMAND ----------

ls_col_pay_type = ['REC_ID','PAYINFO_TYPE', 'PAYMENT_EVENT_TYPE']
df_config_pay_type = df_config_pay_type.select(ls_col_pay_type)


ls_col_channel_map = ['CHANNEL_ID','SOURCE','SUBTYPE','TYPE']
df_config_channel_map = df_config_channel_map.select(ls_col_channel_map)

# COMMAND ----------

# DBTITLE 1,Join Table Together
ls_col = ['aa.ACCOUNT_NO', 'a.ACCOUNT_OBJ_ID0' ,'a.POID_ID0', 'a.POID_TYPE', 'CREATED_T', 'MOD_T', 'EFFECTIVE_T','ITEM_NO' ,
          'a.ITEM_TOTAL', 'PAY_TYPE', 'c.CHANNEL_ID',  'PAYMENT_EVENT_TYPE', 'SOURCE','SUBTYPE', 'TYPE']


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

display(df_payment_base.count()) # 37,263,427 for item_t history before 
# 117,463,584


# COMMAND ----------

# MAGIC %md
# MAGIC ### S00004 Rename Payment Columns

# COMMAND ----------

std_params = {
    'col_map': {
        'account_no': 'fs_acct_id'
        , 'account_obj_id0': 'bill_account_obj_id0'
        , 'poid_id0': 'item_poid_id0'
        , 'poid_type': 'item_poid_type'
        , 'created_t' : 'rec_created_dttm'
        , 'mod_t': 'rec_mod_dttm'
        , 'effective_t': 'rec_effective_dttm'
        , 'item_no': 'item_no'
        , 'item_total': 'item_amount'
        , 'pay_type': 'pay_type'
        , 'channel_id': 'payment_channel_id'
        , 'payment_event_type': 'payment_event_type'
        , 'source': 'payment_source'
        , 'subtype': 'payment_subtype'
        , 'type': 'payment_type'
        , 'created_month': 'rec_created_month'
    }
}

ls_param_col_map = std_params['col_map']

# COMMAND ----------

df_payment_base = (lower_col_names(df_payment_base))

# COMMAND ----------

df_output_curr = (
    df_payment_base
    .select([f.col(c).alias(ls_param_col_map.get(c, c)) for c in df_payment_base.columns])
)

# COMMAND ----------



# COMMAND ----------

display(df_output_curr.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## S00005 Export Payment to Staging Layer

# COMMAND ----------

(
    df_output_curr
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("rec_created_month")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/PAYMENT_BASE")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### S00006 Write off Base

# COMMAND ----------

dir_item_t = '/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T' # raw pinpap item t 
df_item_t = spark.read.format('delta').load(dir_item_t)

# COMMAND ----------

ls_col_wo = ['POID_ID0', 'POID_TYPE', 'CREATED_T', 'MOD_T', 'ACCOUNT_OBJ_ID0', 'EFFECTIVE_T', 'ITEM_NO',  'ITEM_TOTAL']
df_write_off = (
    df_item_t
    .filter(f.col('_is_latest')==1)
    .filter(f.col('_is_deleted')==0)
    .select(ls_col_wo)
    .filter(f.col('poid_type').isin('/item/writeoff'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00007 Write off Base Transformation

# COMMAND ----------

# DBTITLE 1,convert to nzt
ls_time_convert = [c for c in df_write_off.columns if c.endswith('_T')]
#ls_time_convert
df_write_off = to_nzt(df_write_off, ls_time_convert)

# COMMAND ----------

# DBTITLE 1,Join with Acct and get first record of Wrtie off
df_write_off_base = (df_write_off
        .withColumnRenamed('poid_id0', 'item_poid_id0')
        .join(df_account_t, f.col('account_obj_id0') == f.col('poid_id0'), 'inner')
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy('created_t')))
        .filter(col('rnk') ==1)
        .drop('rnk','poid_id0')
)



# COMMAND ----------

# MAGIC %md
# MAGIC ### S00008 Rename Write off Column

# COMMAND ----------

# DBTITLE 1,rename column write off
std_params = {
    'col_map': {
        'item_poid_id0': 'item_poid_id0'
        , 'poid_type': 'item_poid_type'
        , 'created_t' : 'rec_created_dttm'
        , 'mod_t': 'rec_mod_dttm'
        , 'account_obj_id0': 'bill_account_obj_id0'
        , 'effective_t': 'rec_effective_dttm'
        , 'item_no': 'write_off_item_no'
        , 'item_total': 'write_off_item_amount'
        , 'account_no': 'fs_acct_id'
    }
}

ls_param_col_map = std_params['col_map']

# COMMAND ----------

df_write_off_base = (lower_col_names(df_write_off_base))

# COMMAND ----------

df_output_curr = (
    df_write_off_base
    .select([f.col(c).alias(ls_param_col_map.get(c, c)) for c in df_write_off_base.columns])
)

# COMMAND ----------

display(df_output_curr.count()) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00009 Export to Staging Layer

# COMMAND ----------

(
    df_output_curr
    .write
    .format("delta")
    .mode("overwrite")
   # .partitionBy("rec_created_month")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/WRITEOFF_BASE")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### QA

# COMMAND ----------

df_test = spark.read.format('delta').load("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/WRITEOFF_BASE")

display(df_test.limit(100))

# COMMAND ----------

display(df_test
       .withColumn('write_off_month', date_format('rec_created_dttm', 'yyyy-MM'))
       # .filter(col('write_off_month')=='2023-11')
       .groupBy('write_off_month')
       .agg(f.count('fs_acct_id'))
        )

