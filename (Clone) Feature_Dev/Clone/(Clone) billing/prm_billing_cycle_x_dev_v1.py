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

# DBTITLE 1,misc
# MAGIC %run "../../utility_functions/misc"

# COMMAND ----------

# MAGIC %run "../../utility_functions/cycle_utils"

# COMMAND ----------

# MAGIC %run "../../utility_functions/fsr"

# COMMAND ----------

# MAGIC %md ### directories

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/xx/2024q4_moa_account_risk")

# COMMAND ----------

# DBTITLE 1,feature store 02
dir_data_raw =  os.path.join(dir_data_parent_shared, 'd100_raw')
dir_data_meta = os.path.join(dir_data_parent_users, 'd000_meta')
dir_data_stg = os.path.join(dir_data_parent_users, "d200_staging")
dir_data_int =  os.path.join(dir_data_parent_users, "d200_intermediate")
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d302_mobile_pp")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d402_mobile_pp")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d502_mobile_pp")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## s2 data import

# COMMAND ----------

df_stg_bill = spark.read.format("delta").load(os.path.join(dir_data_stg, "d299_src/stg_brm_bill_t"))
df_prm_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base")
df_global_calendar_meta = spark.read.format('delta').load('dbfs:/mnt/feature-store-prod-lab/d000_meta/d001_global_cycle_calendar')
df_fsr_field_meta = spark.read.format("delta").load(os.path.join(dir_data_meta, "d004_fsr_meta/fsr_field_meta"))

# COMMAND ----------

print("bill")
display(df_stg_bill.limit(10))

print("unit base")
display(df_prm_unit_base.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## s2 data processing

# COMMAND ----------

# DBTITLE 1,node parameters 01
vt_param_ssc_reporting_date = "2024-02-29"
vt_param_ssc_reporting_cycle_type = "calendar cycle"
vt_param_ssc_start_date = "2024-02-01"
vt_param_ssc_end_date = "2024-02-29"
vt_param_lookback_billing_cycles = 6

# COMMAND ----------

# DBTITLE 1,node parameters 02
ls_param_unit_base_fields = [
    "reporting_date", "reporting_cycle_type"
    , "fs_cust_id", "fs_acct_id"
]

ls_param_bill_joining_keys = [
    "fs_acct_id"
]

# COMMAND ----------

vt_param_export_table = "prm_bill_cycle_billing_6_mobile_oa_consumer"

# export fields
ls_param_export_fields = get_registered_fields(
    df_fsr_field_meta
    , vt_param_export_table
)

# COMMAND ----------

# DBTITLE 1,unit base
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

display(df_stg_bill.limit(10))

# COMMAND ----------

# DBTITLE 1,bill base
df_base_bill_curr = (
    df_stg_bill
    .filter(
        (f.col("bill_end_t") != 0 )
        # (f.col("bill_start_date") <= vt_param_ssc_reporting_date)
        # (f.col("bill_due_date") <= vt_param_ssc_reporting_date)
    )
    .drop("reporting_date", "reporting_cycle_type")
    .join(
        df_base_unit_base_curr
        .select(ls_param_bill_joining_keys)
        .distinct()
        , ls_param_bill_joining_keys
        , "inner"
    )
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(
                *ls_param_bill_joining_keys
                , "bill_poid_id0"
            )
            .orderBy(f.desc("bill_mod_dttm"))
        )
    )
    .filter(f.col("index") == 1)
    .withColumn(
        "billing_cycle_finish_flag"
        , f.when(
            f.col("bill_due_date") <= vt_param_ssc_reporting_date
            , f.lit('Y')
        )
        .otherwise(f.lit('N'))
    )
    .withColumn(
        "billing_cycle_index_desc"
        , f.row_number().over(
            Window
            .partitionBy(
                *ls_param_bill_joining_keys, "billing_cycle_finish_flag"
            )
            .orderBy(f.desc("bill_due_date"))
        )
    )
    .withColumn(
        "billing_cycle_index_asc"
        , f.row_number().over(
            Window
            .partitionBy(
                *ls_param_bill_joining_keys, "billing_cycle_finish_flag"
            )
            .orderBy("bill_due_date")
        )
    )
    .withColumn(
        "billing_cycle_index"
        , f.when(
            f.col("billing_cycle_finish_flag") == 'N'
            , -(f.col("billing_cycle_index_asc") - 1)
        )
        .otherwise(f.col("billing_cycle_index_desc"))
    )
    .filter(
        (f.col("billing_cycle_index") <= vt_param_lookback_billing_cycles)
        & (f.col("billing_cycle_index") >= -1)
    )
    .withColumnRenamed("bill_poid_id0", "bill_id")
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_bill_curr
    .filter(f.col("fs_acct_id") == '1000008')
    .select('fs_acct_id', "billing_cycle_finish_flag", "bill_start_date", "bill_due_date", "bill_end_date", "bill_close_date", "billing_cycle_index")
    .orderBy("billing_cycle_index")
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,output
df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_bill_curr, ls_param_bill_joining_keys, "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(ls_param_export_fields)
    #.select(
    #    "reporting_date"
    #    , "reporting_cycle_type"
    #    , "fs_cust_id"
    #    , "fs_acct_id"
    #    , "bill_id"
    #    , "bill_no"
    #    , "billing_cycle_finish_flag"
    #    , "billing_cycle_index"
    #    , "bill_create_date"
    #    , "bill_mod_date"
    #    , "bill_start_date"
    #    , "bill_due_date"
    #    , "bill_end_date"
    #    , "bill_close_date"
    #    , "previous_total"
    #    , "total_due"
    #    , "adjusted"
    #    , "due"
    #    , "recvd"
    #    , "transferred"
    #    , "subords_total"
    #    , "current_total"
    #    , 'writeoff'
    #)
)

# COMMAND ----------

display(df_output_curr.limit(10))
