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

# DBTITLE 1,import
df_stg_credit_score_itf = spark.read.format("delta").load(os.path.join(dir_data_stg, "d299_src/stg_inteflow_creditreport"))
df_stg_credit_score_sbl = spark.read.format("delta").load(os.path.join(dir_data_stg, "d299_src/stg_sbl_s_finan_prof_hist_credit_score"))
df_int_cust = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d200_intermediate/d201_mobile_oa_consumer/int_ssc_customer/")


# COMMAND ----------

# DBTITLE 1,sample output check
print("itf credit score")
display(df_stg_credit_score_itf.limit(10))

print("sbl credit score")
display(df_stg_credit_score_sbl.limit(10))

print("cust")
display(df_int_cust.limit(10))

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
    , "fs_cust_id", "fs_acct_id"
]

ls_param_payment_joining_keys = [
    "fs_acct_id"
]

# COMMAND ----------

# DBTITLE 1,unit base
df_base_cust_curr = (
    df_int_cust
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    .select(
        "reporting_date"
        , "reporting_cycle_type"
        , "fs_cust_id"
    )
    .distinct()
)


# COMMAND ----------

# DBTITLE 1,sample output check
display(df_base_cust_curr.count())
display(df_base_cust_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,itf credit score
df_base_cs_itf_curr = (
    df_stg_credit_score_itf
    .join(
        df_base_cust_curr
        , ["fs_cust_id"]
        , "inner"
    )
    .filter(
        (f.col("itf_submit_dttm") <= vt_param_ssc_reporting_date)
        & (f.col("itf_credit_score_dnb_final").isNotNull())

    )
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy("fs_cust_id")
            .orderBy(
                f.desc("itf_submit_dttm")
                , f.desc("dss_hist_end_dttm")
                , f.desc("dss_hist_start_dttm")
                , f.desc("dss_update_dttm")
            )
        )
    )
    .filter(f.col("index") == 1)
    .select(
        "fs_cust_id"
        , "itf_id"
        , f.col("itf_credit_score_dnb_final").alias("itf_credit_score")
        , "itf_submit_dttm"
        , "itf_dealer"
        , f.lit("Y").alias("itf_flag")
        , f.lit("itf").alias("itf_source")
    )
)

# COMMAND ----------

# DBTITLE 1,sample output
display(df_base_cs_itf_curr.limit(10))
display(df_base_cs_itf_curr.count())

# COMMAND ----------

# DBTITLE 1,sbl credit score
df_base_cs_sbl_curr = (
    df_stg_credit_score_sbl
    .join(
        df_base_cust_curr
        , ["fs_cust_id"]
        , "inner"
    )
    .filter(
        (f.col("sbl_credit_req_dttm") <= vt_param_ssc_reporting_date)
        & (f.col("sbl_credit_score").isNotNull())
    )
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy("fs_cust_id")
            .orderBy(
                f.desc("sbl_credit_req_dttm")
                , f.desc("sbl_last_update_dttm")
                , f.desc("sbl_create_dttm")
                , f.desc("dss_hist_end_dttm")
                , f.desc("dss_hist_start_dttm")
                , f.desc("dss_update_dttm")
            )
        )
    )
    .filter(f.col("index") == 1)
    .select(
        "fs_cust_id"
        , f.col("sbl_identifier").alias("sbl_id")
        , f.col("sbl_credit_score").alias("sbl_credit_score")
        , f.col("sbl_credit_req_dttm").alias("sbl_submit_dttm")
        , f.col("sbl_dealer_desc").alias("sbl_dealer")
        , f.lit("Y").alias("sbl_flag")
        , f.lit("sbl").alias("sbl_source")
    )
) 

# COMMAND ----------

# DBTITLE 1,sample output check
display(df_base_cs_sbl_curr.limit(10))
display(df_base_cs_sbl_curr.count())

# COMMAND ----------

# DBTITLE 1,combined score
df_base_cs_curr = (
    df_base_cs_itf_curr
    .join(
        df_base_cs_sbl_curr
        , ["fs_cust_id"]
        , "full"
    )
    .withColumn(
        "cs_id"
        , f.coalesce('itf_id', "sbl_id")
    )
    .withColumn(
        "credit_score"
        , f.coalesce("itf_credit_score", "sbl_credit_score")
    )
    .withColumn(
        "cs_submit_dttm"
        , f.coalesce("itf_submit_dttm", "sbl_submit_dttm")
    )
    .withColumn(
        "cs_submit_date"
        , f.col("cs_submit_dttm").cast("date")
    )
    .withColumn(
        "cs_dealer"
        , f.coalesce("itf_dealer", "sbl_dealer")
    )
    .withColumn(
        "cs_source"
        , f.coalesce("itf_source", "sbl_source")
    )
    .select(
        "fs_cust_id"
        , "cs_id"
        , "credit_score"
        , "cs_submit_dttm"
        , "cs_submit_date"
        , "cs_dealer"
        , "cs_source"
    )
)

# COMMAND ----------

# DBTITLE 1,sample check
display(df_base_cs_curr.limit(10))

display(
    df_base_cs_curr
    .groupBy("cs_source")
    .agg(
        f.count("*")
        , f.countDistinct("fs_cust_id")
    )
)

# COMMAND ----------

# DBTITLE 1,output
df_output_curr = (
    df_base_cust_curr
    .join(df_base_cs_curr, ["fs_cust_id"], "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(
        "reporting_date"
        , "reporting_cycle_type"
        , "fs_cust_id"
        , "cs_id"
        , "credit_score"
        , "cs_submit_dttm"
        , "cs_dealer"
        , "cs_source"
        , "data_update_dttm"
        , "data_update_date"
    )
)

# COMMAND ----------

display(df_output_curr.limit(10))

# COMMAND ----------

display(
    df_output_curr
    .filter(f.col("credit_score") < 0)
    .agg(f.count("*"))
)

# COMMAND ----------

display(
    df_output_curr
    .agg(
        f.median("credit_score")
        , f.avg("credit_score")
        , f.min("credit_score")
        , f.max("credit_score")
    )
)
