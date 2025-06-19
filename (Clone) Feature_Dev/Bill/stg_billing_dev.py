# Databricks notebook source
# MAGIC %md
# MAGIC ###S00001 Library

# COMMAND ----------

import pyspark 
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql.functions import regexp_replace 
from pyspark.sql import Window
from pyspark.sql.functions import col 

# COMMAND ----------

# DBTITLE 1,Directory
dir_bill_t = '/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T'
dir_acct_t = '/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T'

# COMMAND ----------

# MAGIC %run "./Function"

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00002 Account T 

# COMMAND ----------

# DBTITLE 1,Treatment On Account T
df_account_t = spark.read.format('delta').load(dir_acct_t)

df_account_t = (
    df_account_t
    .filter(col('_is_latest') ==1) # pick latest record
    .filter(col('_is_deleted') ==0) 
    .filter(~col('account_no').startswith('S')) # remove subscription account 
    .withColumn('account_no', regexp_replace("account_no", "^0+", "")) # remove leading 0 in account_t 
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### S00003 Bill T 

# COMMAND ----------

# DBTITLE 1,Filter Bill End Date 2021
df_bill_t = spark.read.format('delta').load(dir_bill_t)

df_bill_t_01 = (
    df_bill_t
    .filter(col('_is_latest') ==1)
    .filter(col('_is_deleted') ==0)
    .filter(col('end_t') >= 1609412400) # 2021-01-01
)

# COMMAND ----------

# DBTITLE 1,Remove Duplicate in Bill_T
df_bill_t_02 = (df_bill_t_01
.withColumn('latest', 
                    f.row_number()
                    .over(Window.partitionBy('account_obj_id0','bill_no')
                          .orderBy(f.desc('mod_t')))) # get the latest modified time 
        .filter(f.col('latest')==1)     
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00004 Transformation

# COMMAND ----------

# DBTITLE 1,Transformation v1
# in here we do 
# 1. Join with Account_T and get rid of pending bills 
# 2. Get Required Fields 
# 3. Convert all the epoch to NZT 

ls_col_bill = ['POID_ID0',
'ACCOUNT_OBJ_ID0',
#'INVOICE_OBJ_ID0',
#'AR_BILLINFO_OBJ_ID0',
#'BILLINFO_OBJ_ID0',
'BILL_NO',
'CURRENCY',
'CREATED_T',
'MOD_T',
'DUE_T',
'START_T',
'END_T',
'CLOSED_T',
'PREVIOUS_TOTAL',
'TOTAL_DUE',
'ADJUSTED',
'DUE',
'RECVD',
'TRANSFERRED', 
'SUBORDS_TOTAL',
'CURRENT_TOTAL',
'WRITEOFF'
]

df_bill_base = (
    df_bill_t_02.alias('b')
    .join(df_account_t.alias('a'), col('b.account_obj_id0') == col('a.poid_id0'), 'inner')
    .filter(col('b.bill_no').isNotNull()) # get rid of pending bill
    .filter(col('invoice_obj_id0')!= 0) # get rid of pending bill 
    .select( [col('a.account_no')] + [col(f'b.{col_name}') for col_name in ls_col_bill])
)


# Convert From Epoch to NZT 
ls_time_convert = [c for c in df_bill_base.columns if c.endswith('_T')]

df_bill_base = to_nzt(df_bill_base, ls_time_convert)

df_bill_base = (df_bill_base
                .withColumn('end_month', date_format('end_t', 'yyyy-MM'))
                )

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00005 Rename Column

# COMMAND ----------

std_params = {
    'col_map': {
        'account_no': 'fs_acct_id'
        , 'poid_id0': 'bill_poid_id0'
        , 'account_obj_id0': 'bill_account_obj_id0'
        , 'bill_no': 'bill_no'
        , 'currency': 'bill_currency'
        , 'created_t': 'rec_created_dttm'
        , 'mod_t': 'rec_mod_dttm'
        , 'due_t': 'bill_due_dttm'
        , 'start_t' : 'bill_start_dttm'
        , 'end_t': 'bill_end_dttm'
        , 'closed_t' : 'bill_closed_dttm'
        , 'previous_total': 'bill_previous_total'
        , 'total_due' : 'bill_total_due'
        , 'adjusted':  'bill_adjusted'
        , 'due' : 'bill_due_amt'
        , 'recvd': 'bill_recvd_amt'
        , 'transferred' : 'bill_transferred_amt'
        , 'subords_total' : 'bill_subords_total'
        ,  'current_total' : 'bill_current_total'
        ,  'writeoff' : 'bill_writeoff_amt'
        ,  'end_month' : 'bill_end_month'
    }
}

ls_param_col_map = std_params['col_map']

# COMMAND ----------

df_bill_base = (lower_col_names(df_bill_base))

# COMMAND ----------

df_output_curr = (
    df_bill_base
    .select([f.col(c).alias(ls_param_col_map.get(c, c)) for c in df_bill_base.columns])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### S00006 Export to Staging Layer

# COMMAND ----------

# dbutils.fs.rm("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/BILL_BASE", True)

# COMMAND ----------

(
    df_output_curr
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("bill_end_month")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/BILL_BASE")
)

# COMMAND ----------

# spark.sql("""
#           OPTIMIZE delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T`
#           ZORDER BY (ACCOUNT_OBJ_ID0)
#           """)
