# Databricks notebook source
# MAGIC %md
# MAGIC ### s01 set up

# COMMAND ----------

# DBTITLE 1,library
import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
import pandas as pd
from delta.tables import *

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/dev_sc")

# COMMAND ----------

# DBTITLE 1,feature store 02
dir_data_raw =  os.path.join(dir_data_parent_shared, "d100_raw")
dir_data_meta = os.path.join(dir_data_parent_users, "d000_meta")
dir_data_stg = os.path.join(dir_data_parent_users, "d200_staging")
dir_data_int =  os.path.join(dir_data_parent_users, "d200_intermediate")
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d301_mobile_oa_consumer")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d401_mobile_oa_consumer")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d501_mobile_oa_consumer")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ### s02 data import

# COMMAND ----------

df_prm_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base")
df_global_calendar_meta = spark.read.format("delta").load("dbfs:/mnt/feature-store-prod-lab/d000_meta/d001_global_cycle_calendar")
df_stg_payment = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist")

# COMMAND ----------

# MAGIC %md
# MAGIC ### s03 data process

# COMMAND ----------

# DBTITLE 1,node parameters 01
vt_param_ssc_reporting_date = "2024-11-24"
vt_param_ssc_reporting_cycle_type = "rolling cycle"
vt_param_ssc_start_date = "2024-11-17"
vt_param_ssc_end_date = "2024-11-24"
# vt_param_coll_lookback_cycles = 12
# vt_param_lookback_cycle_unit_type = "days"
# vt_param_lookback_units_per_cycle = 28


# COMMAND ----------

# DBTITLE 1,node parameters 02
ls_param_payment_joining_keys = [
    "fs_acct_id"
]

ls_param_unit_base_fields = [
    "reporting_date"
    , "reporting_cycle_type"
    , "fs_cust_id"
    , "fs_acct_id"
    ]

# export fields
ls_param_export_fields = [
    *ls_param_unit_base_fields
    , "payment_date"
    , "payment_amt"
    , "payment_id"
    , "data_update_date"
    , "data_update_dttm"
]


# COMMAND ----------

# DBTITLE 1,unit base
    # unit base
df_base_unit_base_curr = (
    df_prm_unit_base
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    .select(ls_param_unit_base_fields)
    .distinct()
)

# COMMAND ----------

# DBTITLE 1,check sample output
display(df_base_unit_base_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,latest cycle payment
df_base_coll_payment = (
    df_stg_payment
    # convert to nzt 
    .withColumn(
        "payment_effective_dttm_nzt"
        , from_utc_timestamp("payment_effective_dttm", "Pacific/Auckland")
    ) # convert to nzt 
    .withColumn("payment_effective_date", f.to_date("payment_effective_dttm_nzt"))
    .filter(f.col("payment_effective_date") <= vt_param_ssc_end_date)
    .filter(f.col("payment_effective_date") >= vt_param_ssc_start_date)
    # exclude duplicate payments if any 
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(
                *ls_param_payment_joining_keys
                , "item_poid_id0"
            )
            .orderBy(f.desc("payment_mod_dttm"))
        )
    )
    .filter(f.col("index") == 1) 
    .filter(f.col("item_poid_type") == "/item/payment") # only payment 
    .select(
        "fs_acct_id"
        , f.col("payment_effective_date").alias("payment_date")
        , f.col("item_total").alias("payment_amt")
        , f.col("item_poid_id0").alias("payment_id")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### s04 export data

# COMMAND ----------

# DBTITLE 1,export data
df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_coll_payment, ls_param_payment_joining_keys, "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(*ls_param_export_fields)
)

# COMMAND ----------

display(df_output_curr.limit(10))

# COMMAND ----------

export_data(
    df=df_output_curr
    , export_path = os.path.join(dir_data_prm, 'prm_coll_action_payment_curr')
    , export_format = 'delta'
    , export_mode = 'overwrite'
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
    , ls_dynamic_partition = ['reporting_date', 'reporting_cycle_type']
)
