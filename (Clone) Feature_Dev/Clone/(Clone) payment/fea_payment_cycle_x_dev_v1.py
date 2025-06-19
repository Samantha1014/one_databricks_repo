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
vt_param_payment_lookback_cycles = 3

# COMMAND ----------

# DBTITLE 1,node parameters 02
vt_param_unit_base_table = "fea_unit_base_mobile_oa_consumer"
vt_param_payment_table = "prm_payment_cycle_rolling_6_mobile_oa_consumer"
vt_param_export_table = "fea_payment_cycle_rolling_3_mobile_oa_consumer"

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

# DBTITLE 1,base aggr
df_base_payment_aggr_curr = (
    df_base_payment_curr
    .groupBy(*ls_param_payment_joining_keys, "cycle_index", "payment_category", "payment_method")
    .agg(
        f.sum("payment_amt").alias('payment_amt')
        , f.countDistinct("payment_id").alias("payment_cnt")
    )
)

# COMMAND ----------

# DBTITLE 1,sample data check
display(df_base_payment_aggr_curr.limit(10))

# COMMAND ----------

display(
    df_prm_payment
    .filter(f.col("payment_category") == "fail payment")
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,lookback cycle stats
df_temp_payment_stats_01_curr = (
    df_base_payment_aggr_curr
    .filter(
        (f.col("cycle_index") <= (vt_param_payment_lookback_cycles - 1))
        & (f.col("payment_category").isin("payment", "payment_fail"))
    )
    .groupBy(*ls_param_payment_joining_keys, "cycle_index", "payment_category")
    .agg(
        f.sum("payment_amt").alias('payment_amt')
        , f.sum("payment_cnt").alias("payment_cnt")
    )
    .withColumn("group", f.concat(f.lit("p"), f.col("cycle_index"), f.lit("|"), f.col("payment_category")))
    .groupBy(*ls_param_payment_joining_keys)
    .pivot("group")
    .agg(
        f.sum("payment_amt").alias('|amt_tot_1' + vt_param_payment_cycle_type)
        , f.sum("payment_cnt").alias("|cnt_tot_1" + vt_param_payment_cycle_type)
    )
)

# COMMAND ----------

# DBTITLE 1,sample data check
display(df_temp_payment_stats_01_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,reformat field names
# field name reformat
dict_base_payment_stats_01_col_map = reformat_pivot_col_names(
    df_temp_payment_stats_01_curr.columns
    , vt_pattern_org="(.*)\\|(.*)_\\|(.*)"
    , vt_pattern_present="$2_$3_$1"
    , vt_pattern_suppress="_p0"
)

df_base_payment_stats_01_curr = (
    df_temp_payment_stats_01_curr
    .select(dict_base_payment_stats_01_col_map["col_nm_org"])
    .toDF(*dict_base_payment_stats_01_col_map["col_nm_proc"])
    .withColumn(
        vt_param_payment_flag_field
        , f.when(
            f.col('payment_cnt_tot_1' + vt_param_payment_cycle_type) > 0
            , f.lit('Y')
        )
        .otherwise(f.lit('N'))
    )
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(df_base_payment_stats_01_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,aggr stats
vt_param_stats_02_suffix = str(vt_param_payment_lookback_cycles) + vt_param_payment_cycle_type

df_base_payment_stats_02_curr = (
    df_base_payment_aggr_curr
    .filter((f.col("cycle_index") <= (vt_param_payment_lookback_cycles - 1)))
    .groupBy(*ls_param_payment_joining_keys, "payment_category")
    .agg(
        f.sum("payment_amt").alias('amt_tot')
        , f.sum("payment_cnt").alias("cnt_tot")
        , f.avg("payment_amt").alias('amt_avg')
        , f.avg("payment_cnt").alias("cnt_avg")
    )
    .groupBy(*ls_param_payment_joining_keys)
    .pivot("payment_category")
    .agg(
        f.sum("amt_tot").alias('amt_tot_' + vt_param_stats_02_suffix)
        , f.sum("cnt_tot").alias("cnt_tot_" + vt_param_stats_02_suffix)
        , f.sum("amt_avg").alias('amt_avg_' + vt_param_stats_02_suffix)
        , f.sum("cnt_avg").alias("cnt_avg_" + vt_param_stats_02_suffix)
    )
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(df_base_payment_stats_02_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,payment interval
vt_param_stats_03_suffix = str(vt_param_payment_lookback_cycles) + vt_param_payment_cycle_type

df_base_payment_stats_03_curr = (
    df_base_payment_curr
    .filter(
        (f.col("payment_category") == "payment")
        & (f.col("cycle_index") <= (vt_param_payment_lookback_cycles - 1))
    )
    .withColumn(
        "last_payment_date"
        , f.lag("payment_date").over(
            Window
            .partitionBy(ls_param_payment_primary_keys)
            .orderBy("payment_date")
        )
    )
    .withColumn(
        "payment_interval_days"
        , f.datediff("payment_date", "last_payment_date")
    )
    .groupBy(*ls_param_payment_primary_keys)
    .agg(
        f.round(f.avg("payment_interval_days"), 2).alias("payment_interval_days_avg_" + vt_param_stats_03_suffix)
        , f.round(f.median("payment_interval_days"), 2).alias("payment_interval_days_med_" + vt_param_stats_03_suffix)
        , f.round(f.var_pop('payment_interval_days'), 2).alias('payment_interval_days_var_' + vt_param_stats_03_suffix)
    )
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(df_base_payment_stats_03_curr.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### output

# COMMAND ----------

# DBTITLE 1,calc
df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_payment_stats_01_curr, ls_param_payment_joining_keys, "left")
    .join(df_base_payment_stats_02_curr, ls_param_payment_joining_keys, "left")
    .join(df_base_payment_stats_03_curr, ls_param_payment_joining_keys, "left")
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
    #.select(ls_param_export_fields)
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
    .groupBy("reporting_date")
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.count("*")
        , f.sum("payment_amt_tot_1cycle")
        , f.sum("payment_cnt_tot_1cycle")
        , f.sum("payment_fail_amt_tot_1cycle")
        , f.sum("payment_fail_cnt_tot_1cycle")
    )
)
