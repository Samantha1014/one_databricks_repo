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
df_stg_bill = spark.read.format("delta").load(os.path.join(dir_data_stg, "d299_src/stg_brm_bill_t"))
df_stg_payment = spark.read.format("delta").load(os.path.join(dir_data_stg, "d299_src/stg_brm_payment_hist"))
df_prm_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base")
df_fsr_field_meta = spark.read.format("delta").load(os.path.join(dir_data_meta, "d004_fsr_meta/fsr_field_meta"))

# COMMAND ----------

# DBTITLE 1,sample data check
print("bill")
display(df_stg_bill.limit(10))

print("payment")
display(df_stg_payment.limit(10))

print('unit base')
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

# COMMAND ----------

# DBTITLE 1,node parameters 02
ls_param_unit_base_fields = [
    "reporting_date", "reporting_cycle_type"
    , "fs_cust_id", "fs_acct_id", "fs_srvc_id"
]

ls_param_aod_joining_keys = [
    "fs_acct_id"
]

# COMMAND ----------

vt_param_export_table = "prm_aod_mobile_oa_consumer"

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
)

# COMMAND ----------

display(df_base_unit_base_curr.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### base bills

# COMMAND ----------

# DBTITLE 1,bill init
df_base_bill_init_curr = (
    df_stg_bill
    .filter(
        (f.col("bill_start_date") <= vt_param_ssc_end_date)
        & (f.col("bill_create_date") <= vt_param_ssc_end_date)
        # & (f.col("due_dttm") <= vt_param_ssc_end_date)
        # & (f.col("invoice_obj_id0") != 0)
        # & (f.col("bill_no").isNotNull())
    )
    .join(
        df_base_unit_base_curr
        .select(ls_param_aod_joining_keys)
        .distinct()
        , ls_param_aod_joining_keys
        , "inner"
    )
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(
                *ls_param_aod_joining_keys
                , "bill_no"
            )
            .orderBy(f.desc("bill_mod_dttm"))
        )
    )
    .filter(f.col("index") == 1)
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(ls_param_aod_joining_keys)
            .orderBy(f.desc("bill_end_dttm"))
        )
    )
    .withColumn('total_charge', f.col('total_due') - f.col('previous_total'))
    .select(
        *ls_param_aod_joining_keys
        , "bill_poid_id0"
        , "bill_no"
        , f.col("index").alias("bill_index")
        , "bill_create_date"
        , "bill_start_date"
        , "bill_end_date"
        , "bill_due_date"
        , "bill_close_date"
        , "previous_total"
        , "total_due"
        , "total_charge"
        , "due"
    )
)

# COMMAND ----------

# DBTITLE 1,bill start date
df_base_bill_calc_start_curr = (
    df_base_bill_init_curr
    .filter(f.col("due") == 0)
    .filter(f.col("bill_index") >= 6)
    .withColumn(
        "temp_index"
        , f.row_number().over(
            Window
            .partitionBy(ls_param_aod_joining_keys)
            .orderBy(f.desc("bill_end_date"))
        )
    )
    .filter(f.col("temp_index") == 1)
    .select(
        *ls_param_aod_joining_keys
        , f.col("bill_index").alias("bill_index_start")
        , f.col("bill_end_date").alias("bill_end_date_start")
    )
)

# COMMAND ----------

# DBTITLE 1,bill base
df_base_bill_curr = (
    df_base_bill_init_curr
    .join(
        df_base_bill_calc_start_curr
        , ls_param_aod_joining_keys
        , "left"
    )
    .filter(
        (f.col("bill_index") <= f.col("bill_index_start"))
        | (f.col("bill_index_start").isNull())
    )
    .drop("bill_end_date_start", "bill_index_start")
)

# COMMAND ----------

# DBTITLE 1,sample check
display(
  df_base_bill_curr
  #.filter(f.col("index") > 6)
  #.filter(f.col('fs_acct_id') == '443477710')
  .filter(f.col('fs_acct_id') == '337838335')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### base payments

# COMMAND ----------

# DBTITLE 1,payment init
df_base_payment_init_curr = (
    df_stg_payment
    .filter(
        (f.col("payment_create_dttm") <= vt_param_ssc_end_date)
        & (f.col("payment_effective_dttm") <= vt_param_ssc_end_date)
    )
    .join(
        df_base_unit_base_curr
        .select(ls_param_aod_joining_keys)
        .distinct()
        , ls_param_aod_joining_keys
        , "inner"
    )
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(
                *ls_param_aod_joining_keys
                , "item_poid_id0"
            )
            .orderBy(f.desc("payment_mod_dttm"))
        )
    )
    .filter(f.col("index") == 1)
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(ls_param_aod_joining_keys)
            .orderBy(f.desc("payment_effective_dttm"))
        )
    )
    .select(
        *ls_param_aod_joining_keys
        , "item_poid_id0"
        , "item_poid_type"
        , "ebit_obj_id0"
        , f.col("index").alias("payment_index")
        , "payment_create_date"
        , "payment_mod_date"
        , "payment_effective_date"
        , f.col("item_total").alias("payment_amt")
    )
)

# COMMAND ----------

# DBTITLE 1,payment base
df_base_payment_curr = (
    df_base_payment_init_curr
    .join(
        df_base_bill_calc_start_curr
        , ls_param_aod_joining_keys
        , "left"
    )
    .filter(f.col("payment_create_date") >= f.col("bill_end_date_start"))
    .filter(f.col("payment_create_date") <= vt_param_ssc_end_date)
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_payment_curr
    .filter(f.col('fs_acct_id') == '337838335')
    .limit(100)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### bill payment matched

# COMMAND ----------

# DBTITLE 1,bill credit aggr
df_base_bill_credit_aggr_curr = (
    df_base_bill_curr
    .filter(f.col('total_charge') < 0)
    .groupBy(ls_param_aod_joining_keys)
    .agg(f.sum('total_charge').alias('total_credit'))
)

# COMMAND ----------

# DBTITLE 1,payment aggr
df_base_payment_aggr_curr = (
    df_base_payment_curr
    .groupBy(ls_param_aod_joining_keys)
    .agg(
        f.sum("payment_amt").alias("total_payment")
        , f.min("payment_create_date").alias("payment_create_date_min")
        , f.max("payment_create_date").alias("payment_create_date_max")
    )
    .join(
        df_base_bill_credit_aggr_curr
        , ls_param_aod_joining_keys
        , "left"
    )
    .withColumn(
        'total_receive'
        , f.coalesce('total_payment', f.lit(0)) + f.coalesce('total_credit', f.lit(0))
    )
)

# COMMAND ----------

# DBTITLE 1,sample output check

display(df_base_payment_aggr_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,combined 01
df_base_combined_01_curr = (
    df_base_bill_curr
    .join(
        df_base_payment_aggr_curr
        , ls_param_aod_joining_keys
        , 'left'
    )
    .withColumn(
        "bill_index_rev"
        , f.row_number().over(
            Window
            .partitionBy(ls_param_aod_joining_keys)
            .orderBy("bill_end_date")
        )
    )
    .withColumn(
        "bill"
        , f.when(
            f.col("bill_index_rev") == 1
            , f.col("total_due")
        )
        .otherwise(
            f.col("total_due") - f.col("previous_total")
        )
    )
    .withColumn(
        "cumulative_bill"
        , f.sum("bill").over(
            Window
            .partitionBy(ls_param_aod_joining_keys)
            .orderBy("bill_end_date")
            .rowsBetween(Window.unboundedPreceding, 0)
        )
    )
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_combined_01_curr
    .filter(f.col('fs_acct_id') == '337838335')
)

# COMMAND ----------

# DBTITLE 1,combined 02
df_base_combined_02_curr = (
    df_base_combined_01_curr
    .filter(f.col('bill') >= 0)
    .withColumn('excess', f.col('total_receive') + f.col('cumulative_bill'))
    .withColumn(
        'bill_cleared_flag'
        , f.when(
            f.col('excess') <= 0
            , 1
        )
        .otherwise(0)
    )
    .withColumn(
        'excess_to_next'
        , f.lag('excess', 1, 0).over(
            Window
            .partitionBy(ls_param_aod_joining_keys)
            .orderBy('bill_end_date')
        )
    )
    .withColumn(
        'true_access'
        # if the overall payment does not cover the earliest overdue bill at all
        , f.when(
            (f.col('excess_to_next') == 0)
            & (f.col('bill') > f.abs(f.col('total_receive')))
            & (f.col('bill_cleared_flag') == 0)
            , f.col('total_receive')
        )  # payment sum cannot pay first bill
        .when(
            f.col('excess_to_next') > 0
            , f.lit(0)
        )
        .otherwise(f.col('excess_to_next'))
    )
    .withColumn(
        'due_remain'
        , f.when(
            f.col('bill_cleared_flag') == 1
            , 0
        )
        .otherwise(
            f.col('bill') + f.col('true_access')
        )
    )
)

# COMMAND ----------

display(
    df_base_combined_02_curr
    .filter(f.col('fs_acct_id') == '337838335')
)

# COMMAND ----------

# DBTITLE 1,combined 03
df_base_combined_03_curr = (
    df_base_combined_02_curr
    .withColumn(
        'od_days'
        , f.datediff(f.lit(vt_param_ssc_reporting_date), f.col('bill_due_date'))
    )
    .withColumn(
        'aod_ind'
        , f.when(
            f.col('od_days') <= 0
            , f.lit('aod_current')
        )
        .when(
            f.col('od_days') < 30
            , f.lit('aod_01to30')
        )
        .when(
            f.col('od_days') < 60
            , f.lit('aod_31to60')
        )
        .when(
            f.col('od_days') < 90
            , f.lit('aod_61to90')
        )
        .when(
            f.col('od_days') < 120
            , f.lit('aod_91to120')
        )
        .when(
            f.col('od_days') < 150
            , f.lit('aod_121to150')
        )
        .when(
            f.col('od_days') < 180
            , f.lit('aod_151to180')
        )
        .when(
            f.col('od_days') >= 180
            , f.lit('aod_181plus')
        )
        .otherwise(f.lit("unknown"))
    )
    .orderBy('od_days')
)

# COMMAND ----------

df_base_aod_aggr_curr = (
    df_base_combined_03_curr
    .groupBy(*ls_param_aod_joining_keys, "aod_ind")
    .agg(f.sum("due_remain").alias("value"))
    .filter(f.col("value") > 0)
    # .groupBy("fs_acct_id")
    # .pivot("aod_ind")
    # .agg(f.sum("due_remain"))
)

# COMMAND ----------

display(df_base_aod_aggr_curr.limit(100))

# COMMAND ----------

display(
    df_base_aod_aggr_curr
    .agg(
        f.count("*")
        , f.countDistinct("fs_acct_id")
    )
)

# COMMAND ----------

df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_aod_aggr_curr, ls_param_aod_joining_keys, "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(ls_param_export_fields)
)

# COMMAND ----------

display(df_output_curr.limit(10))

# COMMAND ----------

ls_check = df_output_curr.columns

for i in ls_check:
    print(i)
