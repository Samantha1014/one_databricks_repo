# Databricks notebook source
# MAGIC %md
# MAGIC ### s01 environment set up 

# COMMAND ----------

# DBTITLE 1,library
### libraries
import pyspark
import os

import re
import numpy as np

from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import monotonically_increasing_id, row_number

from itertools import islice, cycle

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

# DBTITLE 1,feature store 02
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

df_prm_coll_action_latest = spark.read.format('delta').load(os.path.join(dir_data_prm, 'prm_coll_action_latest'))
df_prm_coll_payment_curr = spark.read.format('delta').load(os.path.join(dir_data_prm, 'prm_coll_action_payment_curr'))
#df_stg_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist')
df_fea_unit_base = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

# COMMAND ----------

# MAGIC %md
# MAGIC ### s03 data processing 

# COMMAND ----------

# DBTITLE 1,parameters 00
vt_param_ssc_reporting_date = "2024-11-24"
vt_param_ssc_reporting_cycle_type = "rolling cycle"
vt_param_ssc_start_date = "2024-11-17"
vt_param_ssc_end_date = "2024-11-24"
#vt_param_payment_flag_field = "payment_cycle_rolling_flag"
#vt_param_payment_cycle_type = "cycle"
#vt_param_payment_lookback_cycles = 3

# COMMAND ----------

# DBTITLE 1,parameters 01
ls_param_unit_base_fields = [
        "reporting_date", "reporting_cycle_type"
        , "fs_cust_id", "fs_acct_id", "fs_srvc_id"
]

ls_param_coll_joining_keys = [
        "fs_acct_id"
]

ls_param_coll_keys = [
        'latest_coll_id'
]

ls_param_reporting_date_keys = [
        'reporting_date', 'reporting_cycle_type'
]
ls_param_export_fillna_fields_num = [
        'total_coll_payment_5d'
        , 'cnt_payments_post_coll_5d'
]

ls_param_export_fillna_fields_flag = [
        'pay_5d_post_coll_flag'     
]

# export fields
ls_param_export_fields = [
        *ls_param_unit_base_fields
        , 'latest_coll_id'
        , 'latest_coll_complete_date'
        , 'latest_coll_action_type'
        , 'latest_coll_action_name'
        , 'min_pay_date_5d'
        , 'total_coll_payment_5d'
        , 'cnt_payments_post_coll_5d'
        , 'avg_days_to_pay_post_coll_5d'
        , 'pay_5d_post_coll_flag'
        , 'data_update_date'
        , 'data_update_dttm'
]



# COMMAND ----------

# DBTITLE 1,unit base
df_base_unit_base_curr = (
    df_fea_unit_base
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    .select(ls_param_unit_base_fields)
    .distinct()
)

# COMMAND ----------

display(df_prm_coll_payment_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,collection pay within 5 days
df_coll_action_pay_agg = (
    df_prm_coll_action_latest
    .filter( 
            (f.col('reporting_date') == vt_param_ssc_reporting_date)
            & (f.col('reporting_cycle_type') == vt_param_ssc_reporting_cycle_type)
    )
    .join(
        df_prm_coll_payment_curr
        , ls_param_coll_joining_keys + ls_param_reporting_date_keys
        , 'left'
    )
    .withColumn(
        'days_to_pay_latest_coll'
        , f.datediff(f.col('payment_date'), f.col('latest_coll_complete_date'))
    )
    .filter(f.col('days_to_pay_latest_coll').between(0, 5))
     # filter on only those ones that pay 
    .groupBy(
        'fs_acct_id'
        , 'reporting_date'
        , 'reporting_cycle_type'
        , 'latest_coll_id'
    )
    .agg(
        f.min('payment_date').alias('min_pay_date_5d')
        , f.sum('payment_amt').alias('total_coll_payment_5d')
        , f.countDistinct('payment_id').alias('cnt_payments_post_coll_5d')
        , f.avg('days_to_pay_latest_coll').alias('avg_days_to_pay_post_coll_5d')
    )
    .withColumn('pay_5d_post_coll_flag', f.lit('Y'))
    #.filter(f.col('fs_acct_id') == '1049129')
)

# COMMAND ----------

# DBTITLE 1,join to collection base
df_coll_base_curr = (
      df_prm_coll_action_latest
      .drop('fs_cust_id', 'data_update_date', 'data_update_dttm')
      .join(
            df_coll_action_pay_agg, 
            ls_param_coll_joining_keys + ls_param_reporting_date_keys + ls_param_coll_keys
            , 'left')
      .fillna(value=0, subset= ls_param_export_fillna_fields_num)
      .fillna(value = 'N', subset= ls_param_export_fillna_fields_flag)
) 

# COMMAND ----------

# DBTITLE 1,check example
display(df_coll_action_pay_agg.limit(10))

display(df_coll_action_pay_agg.count())

# COMMAND ----------

display(df_coll_base_curr.limit(10))

display(df_coll_base_curr.count())

# COMMAND ----------

# DBTITLE 1,output curr
df_output_curr = (
      df_base_unit_base_curr
      .join(
            df_coll_base_curr
            , ls_param_coll_joining_keys + ls_param_reporting_date_keys
            , 'left'
      )
      .withColumn("data_update_date", f.current_date())
      .withColumn("data_update_dttm", f.current_timestamp())
      .select(ls_param_export_fields)
)

# COMMAND ----------

# DBTITLE 1,check output
display(df_output_curr
        .limit(10)
)
