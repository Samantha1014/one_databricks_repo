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

# MAGIC %run "../../utility_functions/fsr"

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
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d301_mobile_oa_consumer")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d401_mobile_oa_consumer")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d501_mobile_oa_consumer")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## s2 data import

# COMMAND ----------

df_prm_payment = spark.read.format("delta").load(os.path.join(dir_data_prm, "prm_payment_cycle_rolling_6"))
df_fea_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base")
df_fsr_field_meta = spark.read.format("delta").load(os.path.join(dir_data_meta, "d004_fsr_meta/fsr_field_meta"))

# COMMAND ----------

print("payment")
display(df_prm_payment.limit(10))

print("unit base")
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
vt_param_payment_flag_field = "payment_cycle_rolling_flag"
vt_param_payment_cycle_type = "cycle"
vt_param_payment_lookback_cycles = 6

# COMMAND ----------

# DBTITLE 1,node parameters 02
vt_param_unit_base_table = "fea_unit_base_mobile_oa_consumer"
vt_param_payment_table = "prm_payment_cycle_rolling_6_mobile_oa_consumer"
vt_param_export_table = "fea_payment_method_cycle_rolling_6_mobile_oa_consumer"

ls_param_unit_base_fields = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_unit_base_table
)["primary_keys"]

ls_param_payment_fields = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_payment_table
    , keep_all=True
)["all"]

ls_param_payment_primary_keys = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_payment_table
    # , keep_all=True
)["primary_keys"]

ls_param_payment_joining_keys = get_joining_keys(
    df_fsr_field_meta
    , vt_param_payment_table
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

# MAGIC %md
# MAGIC ### unit base

# COMMAND ----------

# DBTITLE 1,calc
df_base_unit_base_curr = (
    df_fea_unit_base
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    .select(ls_param_unit_base_fields)
    .distinct()
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(df_base_unit_base_curr.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### payment

# COMMAND ----------

ls_param_payment_fields

# COMMAND ----------

# DBTITLE 1,base
df_base_payment_curr = (
    df_prm_payment
    .filter(
        (f.col("reporting_date") == vt_param_ssc_reporting_date)
        & (f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    )
    .select(ls_param_payment_fields)
    .withColumn(
        "cycle_index"
        , f.col("cycle_index") - 1
    )
    .withColumn(
        "payment_category"
        , f.when(
            f.col("payment_category") == "fail payment"
            , f.lit('payment_fail')
        )
        .when(
            f.col("payment_category") == "adjustment"
            , f.lit("payment_adj")
        )
        .when(
            f.col("payment_category") == "payment"
            , f.lit("payment")
        )
        .otherwise(f.lit("payment_misc"))
    )
    .withColumn(
        "payment_amt"
        , -f.col("payment_amt")
    )
)

# COMMAND ----------

vt_param_field_suffix = str(vt_param_payment_lookback_cycles) + vt_param_payment_cycle_type

# COMMAND ----------

# DBTITLE 1,main payment method
vt_param_payment_method_code_field = "payment_method_main_code_" + vt_param_field_suffix
vt_param_payment_method_type_field = "payment_method_main_type_" + vt_param_field_suffix
vt_param_payment_method_cnt_field = "payment_method_main_cnt_" + vt_param_field_suffix
vt_param_payment_method_pct_field = "payment_method_main_pct_" + vt_param_field_suffix

df_base_payment_method_curr = (
    df_base_payment_curr
    .filter(
        (f.col("payment_category") == "payment")
        & (f.col("cycle_index") <= (vt_param_payment_lookback_cycles - 1))
    )
    .groupBy(*ls_param_payment_joining_keys, "payment_method")
    .agg(f.countDistinct("payment_id").alias("payment_method_cnt"))
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy(ls_param_payment_joining_keys)
            .orderBy(f.desc("payment_method_cnt"))
        )
    )
    .withColumn(
        "payment_method_cnt_tot"
        , f.sum("payment_method_cnt").over(
            Window
            .partitionBy(ls_param_payment_joining_keys)
        )
    )
    .withColumn(
        "payment_method_pct"
        , f.round(f.col("payment_method_cnt")/f.col("payment_method_cnt_tot"), 3)
    )
    .filter(f.col("index") == 1)
    .withColumn(
        "payment_method_type"
        , f.when(
            f.col("payment_method") == 'cc'
            , f.lit("credit card")
        )
        .when(
            f.col("payment_method") == 'dd'
            , f.lit("debit card")
        )
        .when(
            f.col("payment_method") == 'external'
            , f.lit("external")
        )
        .when(
            f.col("payment_method") == 'partner'
            , f.lit("partner")
        )
        .otherwise(f.lit("misc"))
    )
    .withColumnRenamed("payment_method", vt_param_payment_method_code_field)
    .withColumnRenamed("payment_method_type", vt_param_payment_method_type_field)
    .withColumnRenamed("payment_method_cnt", vt_param_payment_method_cnt_field)
    .withColumnRenamed("payment_method_pct", vt_param_payment_method_pct_field)
    .drop("index", "payment_method_cnt_tot")
)

# COMMAND ----------

# DBTITLE 1,sample data check
display(df_base_payment_method_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,auto payment flag
vt_param_payment_auto_field = "payment_auto_flag_" + vt_param_field_suffix
vt_param_payment_auto_cnt_field = "payment_auto_cnt_" + vt_param_field_suffix
vt_param_payment_auto_pct_field = "payment_auto_pct_" + vt_param_field_suffix

df_base_payment_auto_flag_curr = (
    df_base_payment_curr
    .filter(
        (f.col("payment_category") == "payment")
        & (f.col("cycle_index") <= (vt_param_payment_lookback_cycles - 1))
    )
    .withColumn(
        "auto_pay_flag"
        , f.when(
            f.col("auto_pay_flag") == 'Y'
            , f.lit(1)
        )
        .otherwise(f.lit(0))
    )
    .groupBy(*ls_param_payment_joining_keys)
    .agg(
        f.countDistinct("payment_id").alias("payment_cnt")
        , f.sum("auto_pay_flag").alias("payment_auto_cnt")
    )
    .withColumn(
        "payment_auto_pct"
        , f.when(
            f.col('payment_cnt') == 0
            , f.lit(0)
        )
        .otherwise(f.round(f.col("payment_auto_cnt")/f.col("payment_cnt"), 3))
    )
    .withColumn(
        "payment_auto_flag"
        , f.when(
            f.col("payment_auto_pct") >= 0.8
            , f.lit('Y')
        )
        .otherwise(f.lit('N'))
    )
    .drop("payment_cnt")
    .withColumnRenamed("payment_auto_flag", vt_param_payment_auto_field)
    .withColumnRenamed("payment_auto_cnt", vt_param_payment_auto_cnt_field)
    .withColumnRenamed("payment_auto_pct", vt_param_payment_auto_pct_field)
)

# COMMAND ----------

display(
    df_base_payment_auto_flag_curr 
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### output

# COMMAND ----------

# DBTITLE 1,calc
df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_payment_method_curr, ls_param_payment_joining_keys, "left")
    .join(df_base_payment_auto_flag_curr, ls_param_payment_joining_keys, "left")
)

df_output_curr = add_missing_cols_v2(
    df_output_curr
    , ls_param_export_fillna_fields_num
)

df_output_curr = add_missing_cols_v2(
    df_output_curr
    , ls_param_export_fillna_fields_flag
    , vt_datatype="string"
)

df_output_curr = (
    df_output_curr
    .fillna(value=0, subset=ls_param_export_fillna_fields_num)
    .fillna(value='N', subset=ls_param_export_fillna_fields_flag)
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(ls_param_export_fields)
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_output_curr
    .limit(10)
)

# COMMAND ----------

display(
    df_output_curr
    .filter(f.col("payment_auto_flag_6cycle") == 'Y')
    .limit(10)
)
