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

# DBTITLE 1,data lake
dir_data_dl_brm = "/mnt/prod_brm/raw/cdc"
dir_data_dl_edw = "/mnt/prod_edw/raw/cdc"

# COMMAND ----------

# DBTITLE 1,brm directory
dir_brm_coll_action = os.path.join(dir_data_dl_brm, 'RAW_PINPAP_COLLECTIONS_ACTION_T')
dir_brm_coll_config_action = os.path.join(dir_data_dl_brm, 'RAW_PINPAP_CONFIG_COLLECTIONS_ACTION_T')
dir_brm_acct = os.path.join(dir_data_dl_brm, 'RAW_PINPAP_ACCOUNT_T')

# COMMAND ----------

# MAGIC %md
# MAGIC ### s02 data process

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/dev_sc")

# COMMAND ----------

param_batch_load_field = 'coll_complete_date'
param_batch_load_start_date = '2024-01-01'
param_batch_load_end_date = '2024-12-31'

# COMMAND ----------

# DBTITLE 1,load data raw
df_coll_action  = spark.sql(
    f"""
        with extract_target as (
            select 
                regexp_replace(aa.account_no, '^0+', '') as fs_acct_id
                , a.poid_id0 as coll_poid_id
                , a.scenario_obj_id0 as coll_scenario_obj_id
                , a.poid_type as coll_poid_type
                , a.created_t as coll_created_t
                , from_utc_timestamp(to_timestamp(a.created_t), 'Pacific/Auckland') as coll_create_dttm
                , to_date(from_utc_timestamp(to_timestamp(a.created_t), 'Pacific/Auckland')) as coll_create_date
                , a.mod_t as coll_mod_t
                , from_utc_timestamp(to_timestamp(a.mod_t), 'Pacific/Auckland') as coll_mod_dttm
                , to_date(from_utc_timestamp(to_timestamp(a.mod_t), 'Pacific/Auckland')) as coll_mod_date
                , a.due_t as coll_due_t
                , from_utc_timestamp(to_timestamp(a.due_t), 'Pacific/Auckland') as coll_due_dttm
                , to_date(from_utc_timestamp(to_timestamp(a.due_t), 'Pacific/Auckland')) as coll_due_date
                , a.completed_t as coll_completed_t
                , from_utc_timestamp(to_timestamp(a.completed_t), 'Pacific/Auckland') as coll_complete_dttm
                , to_date(from_utc_timestamp(to_timestamp(a.completed_t), 'Pacific/Auckland')) as coll_complete_date
                , cca.action_descr as coll_action_descr
                , cca.action_name as coll_action_name
                , cca.action_type as coll_action_type
            from delta.`{dir_brm_coll_action}` a 
            left join delta.`{dir_brm_coll_config_action}` cca on a.config_action_obj_id0 = cca.obj_id0
                and cca._is_latest = 1 
                and cca._is_deleted = 0 
            left join delta.`{dir_brm_acct}` aa on aa.poid_id0 = a.account_obj_id0
                and aa._is_latest = 1
                and aa._is_deleted = 0
                and aa.account_no not like 'S%'
            where 
                a._is_latest = 1 
                and a._is_deleted = 0
                and a.completed_t !=0   -- filter out pending collection actions 
        ) 
        select * from extract_target 
        where 
            1=1 
            and {param_batch_load_field} between '{param_batch_load_start_date}' and '{param_batch_load_end_date}'
    """ 
)


# COMMAND ----------

# export_data(
#     df=df_coll_action
#     , export_path = os.path.join(dir_data_parent_users, '/d200_intermediate/stg_collection_action')
#     , export_format = 'delta'
#     , export_mode = 'overwrite'
#     , ls_dynamic_partition = ''

# )
