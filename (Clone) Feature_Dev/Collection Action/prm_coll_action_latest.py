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

# DBTITLE 1,utilities
# MAGIC %run "/Workspace/Users/ethan.li@one.nz/s04_resources/s01_utility_functions/utils_spark_df"

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

# MAGIC %md
# MAGIC #### directory

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/dev_sc")

# COMMAND ----------

dir_data_raw =  os.path.join(dir_data_parent_shared, 'd100_raw')
dir_data_meta = os.path.join(dir_data_parent_users, 'd000_meta')
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
df_global_calendar_meta = spark.read.format('delta').load('dbfs:/mnt/feature-store-prod-lab/d000_meta/d001_global_cycle_calendar')
df_stag_coll_action = spark.read.format("delta").load(os.path.join(dir_data_stg, 'stg_collection_action'))

# COMMAND ----------

print("collection")
display(df_stag_coll_action.limit(10))

print("unit base")
display(df_prm_unit_base.limit(10))

# print('prm payment')
# display(df_prm_latest_payment.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### s03 data processing

# COMMAND ----------

# DBTITLE 1,node parameters 01
vt_param_ssc_reporting_date = '2024-11-24'
vt_param_ssc_reporting_cycle_type = 'rolling cycle'
# vt_param_ssc_start_date = "2024-11-17"
# vt_param_ssc_end_date = "2024-11-24"
# vt_param_coll_lookback_cycles = 12
# vt_param_lookback_cycle_unit_type = "days"
# vt_param_lookback_units_per_cycle = 28


# COMMAND ----------

# DBTITLE 1,node parameters 02
ls_param_unit_base_fields = [
        "reporting_date", "reporting_cycle_type"
        , "fs_cust_id", "fs_acct_id"
]
  
ls_param_coll_joining_keys = [
        "fs_acct_id"
]

# export fields
ls_param_export_fields = [
        *ls_param_unit_base_fields
        , 'latest_coll_id'
        , 'latest_coll_complete_date'
        , 'latest_coll_action_name'
        , 'latest_coll_action_type'
        , 'data_update_date'
        , 'data_update_dttm'
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

# DBTITLE 1,sample output check
display(df_base_unit_base_curr.limit(10))

# COMMAND ----------

display(df_stag_coll_action.limit(10))

# COMMAND ----------

# DBTITLE 1,latest collection action
df_base_coll_curr = (
    df_stag_coll_action
    .filter(
        (f.col('coll_complete_date') <= vt_param_ssc_reporting_date)
    )
    .drop(
        'reporting_date'
        , 'reporting_cycle_type'
        , 'data_update_date'
        , 'data_update_dttm')
    # unit base exclusion 
    .join(
        df_base_unit_base_curr
        .select(ls_param_coll_joining_keys)
        .distinct()
        , ls_param_coll_joining_keys
        , 'inner'
    )
    .withColumn(
        'index'
        , f.row_number().over(
            Window
            .partitionBy(ls_param_coll_joining_keys)
            .orderBy(f.desc('coll_complete_date'))
        )
    )
    .filter(f.col('index') == 1)
    .withColumnRenamed('coll_poid_id', 'latest_coll_id')
    .withColumnRenamed('coll_complete_date', 'latest_coll_complete_date')
    .withColumnRenamed('coll_action_name', 'latest_coll_action_name')
    .withColumnRenamed('coll_action_type', 'latest_coll_action_type')
    #.limit(10)
)

# COMMAND ----------

# DBTITLE 1,output
df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_coll_curr, ls_param_coll_joining_keys, "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(*ls_param_export_fields)
)

# COMMAND ----------

# DBTITLE 1,check output example
display(df_output_curr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### s04 export output

# COMMAND ----------

export_data(
    df=df_output_curr
    , export_path = os.path.join(dir_data_prm, 'prm_coll_action_latest')
    , export_format = 'delta'
    , export_mode = 'overwrite'
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
    , ls_dynamic_partition = ['reporting_date', 'reporting_cycle_type']
)
