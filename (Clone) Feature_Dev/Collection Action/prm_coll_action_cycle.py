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

# collection action stag 
df_stag_coll_action = spark.read.format("delta").load(os.path.join(dir_data_stg, 'stg_collection_action'))

# COMMAND ----------

print("collection")
display(df_stag_coll_action.limit(10))

print("unit base")
display(df_prm_unit_base.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### s03 data processing

# COMMAND ----------

# DBTITLE 1,node parameters 01
vt_param_ssc_reporting_date = '2024-11-24'
vt_param_ssc_reporting_cycle_type = 'rolling cycle'
vt_param_ssc_start_date = "2024-11-17"
vt_param_ssc_end_date = "2024-11-24"
vt_param_coll_lookback_cycles = 12
vt_param_lookback_cycle_unit_type = "days"
vt_param_lookback_units_per_cycle = 28


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
        ,'coll_id'
        , 'coll_complete_date'
        , 'coll_action_name'
        , 'coll_action_type'
        , 'coll_action_category'
        , 'coll_action_category_num'
        , 'cycle_date'
        , 'cycle_index'
        , 'data_update_date'
        , 'data_update_dttm'
]



# COMMAND ----------

df_param_partition_curr = get_lookback_cycle_meta(
    df_calendar=df_global_calendar_meta
    , vt_param_date=vt_param_ssc_reporting_date
    , vt_param_lookback_cycles=vt_param_coll_lookback_cycles
    , vt_param_lookback_cycle_unit_type=vt_param_lookback_cycle_unit_type
    , vt_param_lookback_units_per_cycle=vt_param_lookback_units_per_cycle
)

display(df_param_partition_curr)

df_param_partition_summary_curr = (
    df_param_partition_curr
    .agg(
        f.min("partition_date").alias("date_min")
        , f.max("partition_date").alias("date_max")
    )
)

vt_param_partition_start_date = df_param_partition_summary_curr.collect()[0].date_min
vt_param_partition_end_date = df_param_partition_summary_curr.collect()[0].date_max

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

# DBTITLE 1,collection base
df_base_coll_curr = (
    df_stag_coll_action
    .filter(
        (f.col('coll_complete_date') <= vt_param_partition_end_date)
        & (f.col('coll_complete_date') >= vt_param_partition_start_date)
    )
    .withColumn(
        'coll_action_category'
        , f.when(
            f.col('coll_action_name').isin(
                'Letter Collection Disclosure Statement'
                , 'Swipe Payment Reminder'
                , 'SMS : NZ Delinquent'
                , 'VMG : NZ Delinquent'
            )
            , f.lit('Early')
        )
        .when(
            f.col('coll_action_name').isin(
                'Bar Initiated'
                , 'Redirection'
                , 'SMS : NZ DHNR/BRKN Arr Pre-Bar'
                , 'Swipe Pre Bar Reminder'
                , 'Letter Data Reminder'
                , 'SMS : NZ Delinquent Serious')
            , f.lit('Mid')
        )
        .when(
            f.col('coll_action_name').isin(  
                'InBar'
                , 'Letter Consumer Inbar'
                , 'Disconnect Initiated'
                , 'Proposed Disconnect'
                , 'Disconnect'
                , 'Letter Collection Final Demand Decision')
            , f.lit('Late')
        )
        .when(
            f.col('coll_action_name').isin(
                'Writeoff Proposed Auto Action'
                , 'Send to DCA Decision'
                , 'DCA Assignment'
                , 'Complete Worklist')
            , f.lit('Pre-writeoff')
        )
        .when(
            f.col('coll_action_name').isNull()
            , f.lit('unknown')
        )
        .otherwise(
            f.lit('other')
        )                       
    )
    .withColumn(
        'coll_action_category_num'
        , f.when(
            f.col('coll_action_category') == f.lit('Early')
            , f.lit(1)
        )
        .when(f.col('coll_action_category') == f.lit('Mid')
            , f.lit(2)
        )
        .when(f.col('coll_action_category') == f.lit('Late')
              , f.lit(3)
        )
        .when(f.col('coll_action_category') == f.lit('Pre-writeoff')
              , f.lit(4)
        )
        .when(f.col('coll_action_category') == f.lit('unknown')
              , f.lit(-1)
        )
        .when(f.col('coll_action_category') == f.lit('other')
              , f.lit(0)
        )
    )
    .drop('reporting_date', 'reporting_cycle_type', 'data_update_date', 'data_update_dttm')
    # unit base exclusion 
    .join(
        df_base_unit_base_curr
        .select(ls_param_coll_joining_keys)
        .distinct()
        , ls_param_coll_joining_keys
        , 'inner'
    )
     # cycle attachment 
    .withColumn('partition_date', f.col('coll_complete_date'))
    .join(
        df_param_partition_curr
        .select('partition_date', 'cycle_date', 'cycle_index')
        , ['partition_date']
        , 'left'
    )
    .withColumnRenamed('coll_poid_id', 'coll_id')
)

# COMMAND ----------

# DBTITLE 1,check example
display(df_base_coll_curr
        #.limit(100)
        .filter(f.col('fs_acct_id')== '371954')
)

# COMMAND ----------

# DBTITLE 1,output
df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_coll_curr, ls_param_coll_joining_keys, "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(ls_param_export_fields)
)

# COMMAND ----------

display(df_output_curr
        .limit(10)
)

# COMMAND ----------

export_data(
    df=df_output_curr
    , export_path = os.path.join(dir_data_prm, 'prm_coll_action_cycle_rolling_12')
    , export_format = 'delta'
    , export_mode = 'overwrite'
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
    , ls_dynamic_partition = ['reporting_date', 'reporting_cycle_type']
)
