# Databricks notebook source
# MAGIC %md ## s1 environment setup

# COMMAND ----------

# DBTITLE 1,libraries
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

# MAGIC %md ### utility functions

# COMMAND ----------

# DBTITLE 1,spkdf
# MAGIC %run "../../utility_functions/spkdf_utils"

# COMMAND ----------

# DBTITLE 1,qa
# MAGIC %run "../../utility_functions/qa_utils"

# COMMAND ----------

# MAGIC %run "../../utility_functions/fsr"

# COMMAND ----------

# DBTITLE 1,misc
# MAGIC %run "../../utility_functions/misc"

# COMMAND ----------

# MAGIC %md ### directories

# COMMAND ----------

# DBTITLE 1,data lake
dir_data_dl_brm = "/mnt/prod_brm/raw/cdc"
dir_data_dl_edw = "/mnt/prod_edw/raw/cdc"

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/dev_el/2024q4_moa_account_risk")

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
# MAGIC ## s1 data import

# COMMAND ----------

# DBTITLE 1,data import
df_prm_aod = spark.read.format("delta").load(os.path.join(dir_data_prm, "prm_aod"))
df_fea_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base")
df_fsr_field_meta = spark.read.format("delta").load(os.path.join(dir_data_meta, "d004_fsr_meta/fsr_field_meta"))

# COMMAND ----------

# DBTITLE 1,sample data check
print("aod")
display(df_prm_aod.limit(10))

print('unit base')
display(df_fea_unit_base.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## s2 data processing

# COMMAND ----------

# DBTITLE 1,node parameters 01
vt_param_ssc_reporting_date = "2024-02-29"
vt_param_ssc_reporting_cycle_type = "calendar cycle"
vt_param_ssc_start_date = "2024-02-01"
vt_param_ssc_end_date = "2024-02-29"

# COMMAND ----------

# DBTITLE 1,node parameters 02
vt_param_unit_base_table = "fea_unit_base_mobile_oa_consumer"
vt_param_aod_table = "prm_aod_mobile_oa_consumer"
vt_param_export_table = "fea_aod_mobile_oa_consumer"

ls_param_unit_base_fields = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_unit_base_table
)["primary_keys"]

ls_param_aod_fields = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_aod_table
    , keep_all=True
)["all"]

ls_param_aod_primary_keys = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_aod_table
    # , keep_all=True
)["primary_keys"]

ls_param_aod_joining_keys = get_joining_keys(
    df_fsr_field_meta
    , vt_param_aod_table
    , vt_param_unit_base_table
)

# export fields
ls_param_export_fields = get_registered_fields(
    df_fsr_field_meta
    , vt_param_export_table
)

ls_param_export_fillna_fields_num = get_fillna_fields(
    df_fsr_field_meta
    , vt_param_export_table
    , vt_type="num"
)

ls_param_export_fillna_fields_flag = get_fillna_fields(
    df_fsr_field_meta
    , vt_param_export_table
    , vt_type="flag"
)


# COMMAND ----------

# DBTITLE 1,unit base
df_base_unit_base_curr = (
    df_fea_unit_base
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    .select(ls_param_unit_base_fields)
)

# COMMAND ----------

df_base_aod_curr = (
    df_prm_aod
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    .select(ls_param_aod_fields)
    .withColumn(
        "flag"
        , f.lit('Y')
    )
)

# COMMAND ----------

df_base_aod_pivot_num_curr = add_missing_cols_v2(
    df_base_aod_curr
    .groupBy(ls_param_aod_primary_keys)
    .pivot("aod_ind")
    .agg(f.sum("value"))
    , ls_param_export_fillna_fields_num
)

# COMMAND ----------

df_base_aod_pivot_flag_curr = add_missing_cols_v2(
    df_base_aod_curr
    .withColumn(
        "aod_ind"
        , f.concat(f.col("aod_ind"), f.lit("_flag"))
    )
    .groupBy(ls_param_aod_primary_keys)
    .pivot("aod_ind")
    .agg(f.first("flag"))
    , ls_param_export_fillna_fields_flag
    , vt_datatype="string"
)

# COMMAND ----------

df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_aod_pivot_num_curr, ls_param_aod_joining_keys, "left")
    .join(df_base_aod_pivot_flag_curr, ls_param_aod_joining_keys, "left")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .fillna(value=0, subset=ls_param_export_fillna_fields_num)
    .fillna(value='N', subset=ls_param_export_fillna_fields_flag)
    .select(ls_param_export_fields)
)


# COMMAND ----------

display(df_output_curr.limit(10))
