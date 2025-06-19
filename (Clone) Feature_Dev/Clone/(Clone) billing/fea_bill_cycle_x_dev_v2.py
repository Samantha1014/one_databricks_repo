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
dir_data_parent = "/mnt/feature-store-prod-lab"
dir_data_parent_shared = os.path.join(dir_data_parent, "")
dir_data_parent_users = os.path.join(dir_data_parent, "")

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

df_prm_bill = spark.read.format("delta").load(os.path.join(dir_data_prm, "prm_bill_cycle_billing_6"))
df_fea_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base")
df_fsr_field_meta = spark.read.format("delta").load(os.path.join(dir_data_meta, "d004_fsr_meta/fsr_field_meta"))

# COMMAND ----------

print("bill")
display(df_prm_bill.filter(f.col("reporting_date") == '2024-04-21').limit(10))

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
vt_param_bill_lookback_billing_cycles = 6

# COMMAND ----------

# DBTITLE 1,node parameters 02
vt_param_unit_base_table = "fea_unit_base_mobile_oa_consumer"
vt_param_bill_table = "prm_bill_cycle_billing_6_mobile_oa_consumer"
vt_param_export_table = "fea_bill_cycle_billing_6_mobile_oa_consumer"

ls_param_unit_base_fields = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_unit_base_table
)["primary_keys"]

ls_param_bill_fields = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_bill_table
    , keep_all=True
)["all"]

ls_param_bill_primary_keys = get_fsr_dict(
    df_fsr_field_meta
    , vt_param_bill_table
    # , keep_all=True
)["primary_keys"]

ls_param_bill_joining_keys = get_joining_keys(
    df_fsr_field_meta
    , vt_param_bill_table
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
# MAGIC ### bill

# COMMAND ----------

# DBTITLE 1,base 00
df_base_bill_00_curr = (
    df_prm_bill
    .filter(
        (f.col("reporting_date") == vt_param_ssc_reporting_date)
        & (f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
        & (f.col("bill_end_date") <= vt_param_ssc_reporting_date)
    )
    .select(ls_param_bill_fields)
    #.withColumn(
    #    "bill_period"
    #    , f.datediff(
    #        f.col("bill_end_date")
    #        , f.col("bill_start_date")
    #    )
    #)
    .withColumn(
        "previous_total_due"
        , f.lag("total_due", 1).over(
            Window
            .partitionBy(ls_param_bill_joining_keys)
            .orderBy("bill_end_date")
        )
    )
    .withColumn(
        "total_due_delta"
        , f.col("total_due")/f.col("previous_total_due")
    )
    .withColumn(
        "current_charge"
        , f.col('total_due') - f.col('previous_total')
    )
    .withColumn(
        "previous_current_charge"
        , f.lag("current_charge", 1).over(
            Window
            .partitionBy(ls_param_bill_joining_keys)
            .orderBy("bill_end_date")
        )
    )
    .withColumn(
        "current_charge_delta"
        , f.col("current_charge")/f.col("previous_current_charge")
    )
    .withColumn(
        'next_record_previous_total'
        , f.lead('previous_total', 1).over(
            Window
            .partitionBy(ls_param_bill_joining_keys)
            .orderBy('bill_end_date'))
    )
    .withColumn(
        'calculated_recvd'
        , f.col('total_due') - f.col('next_record_previous_total')
    )
    .filter(
        (f.col("billing_cycle_finish_flag") == 'Y')
        & (f.col("billing_cycle_index") <= vt_param_bill_lookback_billing_cycles)
    )
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_bill_00_curr
    .filter(f.col("fs_acct_id") == '1003757')
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,base 01
df_base_bill_01_curr = (
    df_base_bill_00_curr
    .withColumn(
        "bill_close_flag"
        , f.when(
            (f.col("bill_close_date") <= '1970-01-31')
            | (f.col("bill_close_date") > vt_param_ssc_reporting_date)
            , f.lit('N')
        )
        .otherwise(f.lit('Y'))
    )
    .withColumn(
        "bill_overdue_days"
        , f.when(
            f.col("bill_close_flag") == 'N'
            , f.datediff("reporting_date", "bill_due_date")
        )
        .otherwise(f.datediff("bill_close_date", "bill_due_date"))
    )
    .withColumn(
        "bill_payment_timeliness_status"
        , f.when(
            (f.col("total_due") <= 0)
            , f.lit("credit_bill")
        )
        .when(
            (f.col("bill_close_flag") == 'N')
            , f.lit("miss")
        )
        .when(
            f.col("bill_overdue_days") > 0
            , f.lit("late")
        )
        .when(
            f.col("bill_overdue_days") < 0
            , f.lit("early")
        )
        .when(
            f.col("bill_overdue_days") == 0
            , f.lit("ontime")
        )
        .otherwise(f.lit("unknown"))
    )
    .withColumn(
        "bill_payment_completion_status"
        , f.when(
            (f.col("total_due") <= 0)
            , f.lit("credit_bill")
        )
        .when(
            (f.col("total_due") == f.col("next_record_previous_total"))
            & (f.col("total_due") > 0)
            , f.lit("unpaid")
        )
        .when(
            (f.col("next_record_previous_total") > 0)
            & (f.col("next_record_previous_total") <= f.col("total_due"))
            & (f.col("total_due") > 0)
            , f.lit("partial")
        )
        .when(
            (f.col("next_record_previous_total") == 0)
            & (f.col("total_due") > 0)
            , f.lit("full")
        )
        .when(
            (f.col("next_record_previous_total") < 0)
            & (f.col("total_due") > 0)
            , f.lit("over")
        )
        .otherwise(f.lit("unknown"))
    )
    .withColumn(
        "bill_payment_completion_status"
        , f.when(
            (f.col("bill_payment_timeliness_status").isin(["early", "ontime"]))
            & (f.col("bill_payment_completion_status").isin("unpaid", "partial"))
            , f.lit("invalid")
        )
        .when(
            (f.col("bill_payment_timeliness_status").isin(["miss"]))
            & (f.col("bill_payment_completion_status").isin("over", "full"))
            , f.lit("invalid")
        )
        .otherwise(f.col("bill_payment_completion_status"))
    )
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_bill_01_curr
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_bill_01_curr
    .groupBy("bill_close_flag", "bill_payment_timeliness_status", "bill_payment_completion_status")
    .agg(
        f.countDistinct("fs_acct_id")
        , f.countDistinct("bill_no").alias("bill_cnt")
    )
    .withColumn(
        "bill_cnt_tot"
        , f.sum("bill_cnt").over(
            Window
            .partitionBy("bill_close_flag", "bill_payment_timeliness_status")
        )

    )
    .withColumn(
        "bill_pct"
        , f.round(f.col("bill_cnt")/f.col("bill_cnt_tot"), 2)
    )
    .orderBy("bill_close_flag", "bill_payment_timeliness_status", f.desc("bill_cnt"))
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_bill_01_curr
    #.filter(f.col("bill_close_flag") == 'N')
    .filter(f.col("bill_payment_timeliness_status") == "early")
    .filter(f.col("bill_payment_completion_status") == "unknown")
    .select('fs_acct_id', "bill_no", "billing_cycle_index", "bill_start_date", "bill_end_date", "bill_due_date", "bill_close_date", "total_due", "previous_total", "current_charge", "next_record_previous_total", "calculated_recvd")
)

# COMMAND ----------

# DBTITLE 1,sample output check
display(
    df_base_bill_01_curr
    #.filter(f.col("fs_acct_id") == '334700082')
    .filter(f.col("fs_acct_id") == '452243987')
    .select(
        'fs_acct_id', "bill_no", "billing_cycle_index"
        , "bill_start_date", "bill_end_date", "bill_due_date", "bill_close_date"
        , "total_due", "previous_total", "current_charge", "next_record_previous_total", "calculated_recvd"
        , "bill_payment_timeliness_status", 'bill_payment_completion_status', "bill_close_flag"
    )
)

# COMMAND ----------

vt_param_field_suffix = str(vt_param_bill_lookback_billing_cycles) + "bmnth"

# COMMAND ----------

# DBTITLE 1,stats 00 - basic
df_temp_bill_stats_00_curr = (
    df_base_bill_01_curr
    .withColumn(
        "billing_cycle_index"
        , f.concat(f.lit("p"), f.col('billing_cycle_index') - 1)
    )
    .groupBy(*ls_param_bill_joining_keys)
    .pivot("billing_cycle_index")
    .agg(
        f.first("bill_no").alias("|bill_no")
        , f.first("bill_start_date").alias("|bill_start_date")
        , f.first("bill_end_date").alias("|bill_end_date")
        , f.first("bill_due_date").alias("|bill_due_date")
        , f.first("bill_close_date").alias("|bill_close_date")
        , f.sum("total_due").alias("|bill_due_amt")
        , f.sum("current_charge").alias("|bill_charge_amt")
        , f.sum("previous_total").alias("|bill_carryover_bal")
        , f.first("bill_close_flag").alias("|bill_close_flag")
        , f.sum("bill_overdue_days").alias("|bill_overdue_days")
        , f.first("bill_payment_timeliness_status").alias("|bill_payment_timeliness_status")
    )
)

# COMMAND ----------

# DBTITLE 1,reformat
# field name reformat
dict_base_bill_stats_00_col_map = reformat_pivot_col_names(
    df_temp_bill_stats_00_curr.columns
    , vt_pattern_org="(.*)_\\|(.*)"
    , vt_pattern_present="$2_$1"
    , vt_pattern_suppress="_p0"
)

df_base_bill_stats_00_curr = (
    df_temp_bill_stats_00_curr
    .select(dict_base_bill_stats_00_col_map["col_nm_org"])
    .toDF(*dict_base_bill_stats_00_col_map["col_nm_proc"])
)

# COMMAND ----------

display(df_base_bill_stats_00_curr.limit(10))

# COMMAND ----------

for i in df_base_bill_stats_00_curr.columns:
    print(i)

# COMMAND ----------

# DBTITLE 1,stats 01 - aggr
df_base_bill_stats_01_curr = (
    df_base_bill_01_curr
    .groupBy(ls_param_bill_joining_keys)
    .agg(
        f.countDistinct("bill_no").alias("bill_cnt_tot_" + vt_param_field_suffix)
        , f.sum("total_due").alias("bill_due_amt_tot_" + vt_param_field_suffix)
        , f.avg("total_due").alias("bill_due_amt_avg_" + vt_param_field_suffix)
        , f.median("total_due").alias("bill_due_amt_med_" + vt_param_field_suffix)
        , f.avg("total_due_delta").alias("bill_due_amt_delta_avg_" + vt_param_field_suffix)
        , f.median("total_due_delta").alias("bill_due_amt_delta_med_" + vt_param_field_suffix)
        , f.max("total_due_delta").alias("bill_due_amt_delta_max_" + vt_param_field_suffix)

        , f.sum("current_charge").alias("bill_charge_amt_tot_" + vt_param_field_suffix)
        , f.avg("current_charge").alias("bill_charge_amt_avg_" + vt_param_field_suffix)
        , f.median("current_charge").alias("bill_charge_amt_med_" + vt_param_field_suffix)
        , f.avg("current_charge_delta").alias("bill_charge_amt_delta_avg_" + vt_param_field_suffix)
        , f.median("current_charge_delta").alias("bill_charge_amt_delta_med_" + vt_param_field_suffix)
        , f.max("current_charge_delta").alias("bill_charge_amt_delta_max_" + vt_param_field_suffix)

        , f.avg("previous_total").alias("bill_carryover_bal_avg_" + vt_param_field_suffix)
        , f.median("previous_total").alias("bill_carryover_bal_med_" + vt_param_field_suffix)
    )
)

display(
    df_base_bill_stats_01_curr
    .limit(100)
)

# COMMAND ----------

for i in df_base_bill_stats_01_curr.columns:
    print(i)

# COMMAND ----------

# DBTITLE 1,stats 02 - payment time
df_temp_bill_stats_02_curr = (
    df_base_bill_01_curr
    .groupBy(*ls_param_bill_joining_keys, "bill_payment_timeliness_status")
    .agg(
        f.countDistinct("bill_no").alias("bill_cnt")
        , f.sum("current_charge").alias("bill_charge_amt")
        , f.avg("bill_overdue_days").alias("bill_overdue_days")
    )
    .withColumn(
        "bill_cnt_tot"
        , f.sum("bill_cnt").over(
            Window
            .partitionBy(ls_param_bill_joining_keys)
        )
    )
    .withColumn(
        "bill_cnt_pct"
        , f.round(f.col("bill_cnt")/f.col("bill_cnt_tot"), 2)
    )
    .withColumn(
        "bill_charge_amt_tot"
        , f.sum("bill_charge_amt").over(
            Window
            .partitionBy(ls_param_bill_joining_keys)
        )
    )
    .withColumn(
        "bill_charge_amt_pct"
        , f.round(f.col("bill_charge_amt")/f.col("bill_charge_amt_tot"), 2)
    )
    .filter(f.col("bill_payment_timeliness_status").isin("early", "ontime", "late", "miss"))
    .groupBy(*ls_param_bill_joining_keys)
    .pivot("bill_payment_timeliness_status")
    .agg(
        f.sum("bill_cnt").alias("|bill_cnt_|tot_" + vt_param_field_suffix)
        , f.sum("bill_cnt_pct").alias("|bill_cnt_|pct_"+ vt_param_field_suffix)
        , f.sum("bill_charge_amt").alias("|bill_charge_amt_|tot_" + vt_param_field_suffix)
        , f.sum("bill_charge_amt_pct").alias("|bill_charge_amt_|pct_" + vt_param_field_suffix)
        , f.avg("bill_overdue_days").alias("|bill_overdue_days_|avg_" + vt_param_field_suffix)
    )
)

# COMMAND ----------

# DBTITLE 1,reformat
# field name reformat
dict_base_bill_stats_02_col_map = reformat_pivot_col_names(
    df_temp_bill_stats_02_curr.columns
    , vt_pattern_org="(.*)_\\|(.*)_\\|(.*)"
    , vt_pattern_present="$2_$1_$3"
    , vt_pattern_suppress="_p0"
)

df_base_bill_stats_02_curr = (
    df_temp_bill_stats_02_curr
    .select(dict_base_bill_stats_02_col_map["col_nm_org"])
    .toDF(*dict_base_bill_stats_02_col_map["col_nm_proc"])
)

# COMMAND ----------

# DBTITLE 1,sample data check
display(
    df_base_bill_stats_02_curr 
    .limit(10)
)

# COMMAND ----------

for i in df_base_bill_stats_02_curr.columns:
    print(i)

# COMMAND ----------

# DBTITLE 1,stats 03 - payment completion
df_temp_bill_stats_03_curr = (
    df_base_bill_01_curr
    .groupBy(*ls_param_bill_joining_keys, "bill_payment_completion_status")
    .agg(
        f.countDistinct("bill_no").alias("bill_cnt")
    )
    .withColumn(
        "bill_cnt_tot"
        , f.sum("bill_cnt").over(
            Window
            .partitionBy(ls_param_bill_joining_keys)
        )
    )
    .withColumn(
        "bill_cnt_pct"
        , f.round(f.col("bill_cnt")/f.col("bill_cnt_tot"), 2)
    )
    .filter(f.col("bill_payment_completion_status").isin("full", "partial", "over", "unpaid"))
    .groupBy(*ls_param_bill_joining_keys)
    .pivot("bill_payment_completion_status")
    .agg(
        f.sum("bill_cnt").alias("|bill_cnt_|tot_" + vt_param_field_suffix)
        , f.sum("bill_cnt_pct").alias("|bill_cnt_|pct_"+ vt_param_field_suffix)
    )
)

# COMMAND ----------

# DBTITLE 1,reformat
# field name reformat
dict_base_bill_stats_03_col_map = reformat_pivot_col_names(
    df_temp_bill_stats_03_curr.columns
    , vt_pattern_org="(.*)_\\|(.*)_\\|(.*)"
    , vt_pattern_present="$2_$1_$3"
    , vt_pattern_suppress="_p0"
)

df_base_bill_stats_03_curr = (
    df_temp_bill_stats_03_curr
    .select(dict_base_bill_stats_03_col_map["col_nm_org"])
    .toDF(*dict_base_bill_stats_03_col_map["col_nm_proc"])
)

# COMMAND ----------

# DBTITLE 1,sample data check
display(df_base_bill_stats_03_curr.limit(10))

# COMMAND ----------

for i in df_base_bill_stats_03_curr.columns:
    print(i)

# COMMAND ----------

df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_bill_stats_00_curr, ls_param_bill_joining_keys, "left")
    .join(df_base_bill_stats_01_curr, ls_param_bill_joining_keys, "left")
    .join(df_base_bill_stats_02_curr, ls_param_bill_joining_keys, "left")
    .join(df_base_bill_stats_03_curr, ls_param_bill_joining_keys, "left")
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

display(
    df_output_curr
    .limit(10)
)
