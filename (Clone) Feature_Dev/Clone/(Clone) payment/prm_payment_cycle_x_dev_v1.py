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

# MAGIC %md ### directories

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
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d302_mobile_pp")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d402_mobile_pp")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d502_mobile_pp")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## s2 data import

# COMMAND ----------

dbutils.fs.ls("/mnt/feature-store-prod-lab/d000_meta")

# COMMAND ----------

df_stg_payment = spark.read.format("delta").load(os.path.join(dir_data_stg, "d299_src/stg_brm_payment_hist"))
df_prm_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base")
df_global_calendar_meta = spark.read.format('delta').load('dbfs:/mnt/feature-store-prod-lab/d000_meta/d001_global_cycle_calendar')

# COMMAND ----------

print("payment")
display(df_stg_payment.limit(10))

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
vt_param_lookback_cycles = 6
vt_param_lookback_cycle_unit_type = "days"
vt_param_lookback_units_per_cycle = 28

# COMMAND ----------

# DBTITLE 1,node parameters 02
ls_param_unit_base_fields = [
    "reporting_date", "reporting_cycle_type"
    , "fs_cust_id", "fs_acct_id"
]

ls_param_payment_joining_keys = [
    "fs_acct_id"
]

# COMMAND ----------

# DBTITLE 1,partition parameters
df_param_partition_curr = get_lookback_cycle_meta(
    df_calendar=df_global_calendar_meta
    , vt_param_date=vt_param_ssc_reporting_date
    , vt_param_lookback_cycles=vt_param_lookback_cycles
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

# DBTITLE 1,payment base
df_base_payment_curr = (
    df_stg_payment
    .filter(f.col("payment_create_date") <= vt_param_partition_end_date)
    .filter(f.col("payment_create_date") >= vt_param_partition_start_date)
    .drop("reporting_date", "reporting_cycle_type")
    .join(
        df_base_unit_base_curr
        .select(ls_param_payment_joining_keys)
        .distinct()
        , ls_param_payment_joining_keys
        , "inner"
    )
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
    .withColumn("partition_date", f.col("payment_create_date"))
    .join(
        df_param_partition_curr
        .select("partition_date", "cycle_date", "cycle_index")
        , ["partition_date"]
        , "left"
    )
    .withColumn(
        'auto_pay_flag'
        , f.when(
            f.col('payment_event_type').isin(
                '/event/billing/payment/dd'
                , '/event/billing/payment/cc'
            )
            , f.lit('Y')
        )
        .otherwise(f.lit('N'))
    )
    .withColumn(
        'payment_method'
        , f.element_at(
            f.split(f.col('payment_event_type'), '/')
            , f.size(f.split(f.col('payment_event_type'), '/'))
        )
    )
    .withColumn(
        'payment_category'
        , f.when(
            f.col('item_poid_type')=='/item/adjustment'
            , f.lit('adjustment')
        )
        .when( 
            f.col('payment_event_type') == '/event/billing/payment/failed'
            , f.lit('fail payment')
        )
        .when(
            f.col('item_poid_type') == '/item/payment'
            , f.lit('payment')
        )
        .otherwise(f.lit('other'))
    )
    .withColumnRenamed("item_poid_id0", "payment_id")
    .withColumnRenamed("payment_create_date", "payment_date")
    .withColumnRenamed("item_total", "payment_amt")
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_payment_curr
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,output
df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_payment_curr, ls_param_payment_joining_keys, "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(
        "reporting_date"
        , "reporting_cycle_type"
        , "fs_cust_id"
        , "fs_acct_id"
        , "payment_id"
        , "payment_date"
        , "cycle_date"
        , "cycle_index"
        , "payment_amt"
        , "auto_pay_flag"
        , "payment_method"
        , "payment_category"
    )
    #.select(ls_param_export_fields)
)

# COMMAND ----------

display(df_output_curr.limit(10))
