# Databricks notebook source
# MAGIC %md
# MAGIC ## s01 environment set up 

# COMMAND ----------

import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
import pandas as pd
from delta.tables import *

# COMMAND ----------

# MAGIC %md
# MAGIC ### utility

# COMMAND ----------

# DBTITLE 1,spark df
# MAGIC %run "/Workspace/Users/ethan.li@one.nz/s04_resources/s01_utility_functions/utils_spark_df"

# COMMAND ----------

# MAGIC %md
# MAGIC ### directory

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

# DBTITLE 1,feature store 01
vt_param_segment = "1_mobile_oa_consumer"
dir_data_fs = "/mnt/feature-store-prod-lab"
dir_data_fs_shared = os.path.join(dir_data_fs, "")
dir_data_fs_users = os.path.join(dir_data_fs, "")

# COMMAND ----------

# DBTITLE 1,feature store 02
dir_data_fs_raw =  os.path.join(dir_data_fs_shared, 'd100_raw')
dir_data_fs_meta = os.path.join(dir_data_fs_users, 'd000_meta')
dir_data_fs_stag = os.path.join(dir_data_fs_shared, "d200_staging")
dir_data_fs_int =  os.path.join(dir_data_fs_users, "d200_intermediate")
dir_data_fs_prm =  os.path.join(dir_data_fs_users, f"d300_primary/d30{vt_param_segment}")
dir_data_fs_fea =  os.path.join(dir_data_fs_users, f"d400_feature/d40{vt_param_segment}")
dir_data_fs_mvmt = os.path.join(dir_data_fs_users, f"d500_movement/d50{vt_param_segment}")
dir_data_fs_serv = os.path.join(dir_data_fs_users, "d600_serving")
dir_data_fs_tmp =  os.path.join(dir_data_fs_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## s02 data import

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/')

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

# COMMAND ----------

# DBTITLE 1,load data
## unit base 
df_fea_unitbase = spark.read.format('delta').load(os.path.join(dir_data_fs_fea, 'fea_unit_base'))

## payment 
df_stag_payment = spark.read.format('delta').load(os.path.join(dir_data_fs_stag, 'd299_src/stg_brm_payment_hist'))

# COMMAND ----------

# DBTITLE 1,collection action
df_coll_action  = spark.sql(
        f"""
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
        left join delta.`{dir_brm_acct}` aa on aa.poid_id0 = a.account_obj_id0
        where 
            a._is_latest = 1 
            and cca._is_latest = 1 
            and aa._is_latest = 1
            and completed_t !=0  
            --- shceduled collection action in futture but not complete 
        """ 
        )


# COMMAND ----------

display(df_coll_action.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## s03 data process

# COMMAND ----------

# DBTITLE 1,parameters 00
dict_params = {
  "lookback_billing_cycles": 6
  , "lookback_months": 6
}

# COMMAND ----------

# DBTITLE 1,parameters 01

ls_param_unit_base_fields = [
    "reporting_date", "reporting_cycle_type"
    , "fs_cust_id", "fs_acct_id", "fs_srvc_id"
]

ls_param_bill_joining_keys = [
    "fs_acct_id"
]



# COMMAND ----------

# DBTITLE 1,dedupe for unit base
# dedup 
df_base_unit_base = (df_fea_unitbase
        .select(*ls_param_bill_joining_keys, 'cust_start_date' )
        .distinct()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## s04 data stag layer 

# COMMAND ----------

# DBTITLE 1,stag layer 01 - add collection action category
## category has been modified according to Shaun's feedback 
# 
#  bar initiated = redirection 
# SMS : NZ DHNR/BRKN Arr Pre-Bar = Swipe Pre Bar Reminder


df_coll_action_01 = (
    df_coll_action
    .join(df_base_unit_base, ['fs_acct_id'], 'inner')  # get unit base 
    .withColumn('complete_date', f.to_date('complete_dttm_nzt') )
    .withColumn('coll_action_category', f.when(f.col('action_name')
                                               .isin(
                                                   'Letter Collection Disclosure Statement'
                                                 , 'Swipe Payment Reminder'
                                                 , 'SMS : NZ Delinquent'
                                                 , 'VMG : NZ Delinquent'
                                                                  )
                                               , 'Early')
                                         .when(f.col('action_name')
                                               .isin(
                                               'Bar Initiated'
                                             , 'Redirection'
                                             , 'SMS : NZ DHNR/BRKN Arr Pre-Bar'
                                             , 'Swipe Pre Bar Reminder'
                                             , 'Letter Data Reminder'
                                             , 'SMS : NZ Delinquent Serious' 
                                                    )
                                               , 'Mid'
                                               )
                                         .when(
                                             f.col('action_name')
                                             .isin(  'InBar'
                                                   , 'Letter Consumer Inbar'
                                                   ,  'Disconnect Initiated'
                                                   ,  'Proposed Disconnect'
                                                   ,  'Disconnect'
                                                   ,  'Letter Collection Final Demand Decision'
                                                   )
                                             , 'Late'
                                             )
                                         .when(
                                             f.col('action_name')
                                             .isin('Writeoff Proposed Auto Action'
                                                   , 'Send to DCA Decision'
                                                   , 'DCA Assignment'
                                                   , 'Complete Worklist'
                                                   )
                                             , 'Pre-writeoff'
                                            )
                                         .when(
                                             f.col('action_name').isNull()
                                             , 'unknown'
                                              )
                                         .otherwise('other')
                                               
                )
    .withColumn('coll_action_category_num'
                , f.when(f.col('coll_action_category') == 'Early', 1)
                 .when(f.col('coll_action_category') == 'Mid', 2)
                 .when(f.col('coll_action_category') == 'Late', 3)
                 .when(f.col('coll_action_category') == 'Pre-writeoff', 4)
                 .when(f.col('coll_action_category') == 'unknown', -1)
                 .when(f.col('coll_action_category') == 'other', 0 )
                )
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
