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

df_prm_bill = spark.read.format("delta").load(os.path.join(dir_data_prm, "prm_bill_cycle_billing_6"))

# COMMAND ----------

print("bill")
display(df_prm_bill.limit(10))

# COMMAND ----------

display(
    df_prm_bill
    .groupBy("reporting_date", "reporting_cycle_type")
    .agg(
        f.count("*").alias("cnt")
        , f.countDistinct("bill_no")
        , f.countDistinct("fs_acct_id")
    )
    .orderBy(f.desc("reporting_date"))
)

# COMMAND ----------

display(
    df_prm_bill
    .groupBy("reporting_date", "reporting_cycle_type", "fs_acct_id")
    .agg(f.countDistinct("bill_no").alias("bill_cnt"))
    .groupBy("reporting_date", "reporting_cycle_type")
    .agg(
        f.avg("bill_cnt").alias("bill_cnt_avg")
        , f.median("bill_cnt").alias("bill_cnt_med")
    )
    .orderBy(f.desc("reporting_date"))
)

# COMMAND ----------

vt_param_cnt_field = "fs_srvc_id"
vt_param_group_field = "bill_payment_timeliness_status"
vt_param_metric_field = "cnt"

df_check_input = (
    df_prm_bill
)

df_check = trend_profile(
    df = df_check_input
    , group_field = vt_param_group_field
    , cnt_field = vt_param_cnt_field
    , metric = vt_param_metric_field
)

display(df_check)

# COMMAND ----------

vt_param_cnt_field = "fs_srvc_id"
vt_param_group_field = "bill_payment_timeliness_status"
vt_param_metric_field = "pct"

df_check_input = (
    df_prm_bill
)

df_check = trend_profile(
    df = df_check_input
    , group_field = vt_param_group_field
    , cnt_field = vt_param_cnt_field
    , metric = vt_param_metric_field
)

display(df_check)

# COMMAND ----------

vt_param_cnt_field = "fs_srvc_id"
vt_param_group_field = "late_flag"
vt_param_metric_field = "pct"

df_check_input = (
    df_prm_bill
    .withColumn(
        "late_flag"
        , f.when(
            (f.col("bill_cnt_late_tot_6bmnth") > 0)
            & (f.col("bill_cnt_late_tot_6bmnth") <= 3)
            , f.lit('M')
        )
        .when(
             (f.col("bill_cnt_late_tot_6bmnth") > 3)
            , f.lit('H')
        )
        .otherwise(f.lit('N'))
    )
)

df_check = trend_profile(
    df = df_check_input
    , group_field = vt_param_group_field
    , cnt_field = vt_param_cnt_field
    , metric = vt_param_metric_field
)

display(df_check)
