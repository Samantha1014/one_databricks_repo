# Databricks notebook source
# MAGIC %md ## s000 environment setup

# COMMAND ----------

# MAGIC %md ### s001 libraries

# COMMAND ----------

# libraries
import os

import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md ### s002 sf connectivity

# COMMAND ----------

# MAGIC %run "../utility_functions/spkdf_utils"

# COMMAND ----------

# MAGIC %run "./utility_functions"

# COMMAND ----------

# ------------- Use snowflake utility
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils

# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "SANDBOX",
  "sfWarehouse": "LAB_DS_WH"
}

# COMMAND ----------

# MAGIC %md ### s003 directories

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab"
dir_mls_data_parent = "/mnt/ml-store-prod-lab/classification"

# COMMAND ----------

dir_mls_data_score = os.path.join(dir_mls_data_parent, "d400_model_score")

# COMMAND ----------

dir_fs_data_meta = os.path.join(dir_fs_data_parent, 'd000_meta')
dir_fs_data_raw =  os.path.join(dir_fs_data_parent, 'd100_raw')
dir_fs_data_int =  os.path.join(dir_fs_data_parent, "d200_intermediate")
dir_fs_data_prm =  os.path.join(dir_fs_data_parent, "d300_primary")
dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")
dir_fs_data_target = os.path.join(dir_fs_data_parent, "d500_movement")
dir_fs_data_serv = os.path.join(dir_fs_data_parent, "d600_serving")

# COMMAND ----------

# MAGIC %md ## s100 data import

# COMMAND ----------

vt_param_reporting_date = "2023-11-26"
vt_param_reporting_cycle_type = "rolling cycle"

# COMMAND ----------

# MAGIC %md ### s101 feature store

# COMMAND ----------

df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))

# COMMAND ----------

# MAGIC %md ### s102 ml store

# COMMAND ----------

#df_mls_score_dr_apple = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_apple_pred30d"))
df_mls_score_dr_samsung = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_samsung_pred30d"))
#df_mls_score_dr = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_pred30d"))
df_mls_score_ifp = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_upsell_ifp_pred30d"))
df_mls_score_churn = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_churn_pred30d"))

# COMMAND ----------

# MAGIC %md ### s103 target log

# COMMAND ----------

df_campaign_hist = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select 
                *
                , prim_accs_num as fs_srvc_id

            from lab_ml_store.sandbox.dna_sql_ml_ifp_exp_fact
            where campaign_name in ('DGC-2493', 'DGC-2494', 'DGC-2499')
        """
    )
    .load()
)

df_campaign_hist = lower_col_names(df_campaign_hist)

display(df_campaign_hist.limit(100))

# COMMAND ----------

display(
    df_campaign_hist
    .groupBy("cohort")
    .agg(
        f.count("*")
    )
)

# COMMAND ----------

# MAGIC %md ### s104 global control group

# COMMAND ----------

df_gc_curr = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select 
                *
                , service_id as fs_srvc_id

            from lab_ml_store.sandbox.dna_sql_global_control
            where control_group_id = 'mobile_oa_021'
        """
    )
    .load()
)

df_gc_prev = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select 
                *
                , service_id as fs_srvc_id

            from lab_ml_store.sandbox.dna_sql_global_control
            where control_group_id = 'mobile_oa_020'
        """
    )
    .load()
)

df_gc_curr = lower_col_names(df_gc_curr)
df_gc_prev = lower_col_names(df_gc_prev)

display(df_gc_curr.limit(100))
display(df_gc_prev.limit(100))

# COMMAND ----------

# MAGIC %md ## s200 data processing

# COMMAND ----------

# MAGIC %md ### s201 base candidate

# COMMAND ----------

df_base_full = (
    df_fs_master
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
)

display(
    df_base_full
    .limit(100)
)

# COMMAND ----------

# MAGIC %md ### s202 exclusion flag

# COMMAND ----------

# current global control
df_tmp_excl_01 = (
    df_gc_curr
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "gc_curr_flag"
        , f.lit('Y')
    )
)

# previous global control
df_tmp_excl_02 = (
    df_gc_prev
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "gc_prev_flag"
        , f.lit('Y')
    )
)

# current active campaign
df_tmp_excl_03 = (
    df_campaign_hist
    .filter(f.col("cohort") == 'TARGET')
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "ch_target_flag"
        , f.lit('Y')
    )
)

# current active local control
df_tmp_excl_04 = (
    df_campaign_hist
    .filter(f.col("cohort") == 'LOCAL_CONTROL')
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "ch_control_flag"
        , f.lit('Y')
    )
)

# ifp <= 90 days
df_tmp_excl_05 = (
    df_fs_ifp_srvc
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .filter(f.col("ifp_srvc_dvc_flag") == 'Y')
    .withColumn("datediff", f.datediff(f.current_date(), f.col("ifp_srvc_dvc_term_start_date")))
    .filter(f.col("datediff") <= 90)
    .select(
        "fs_srvc_id"
        , "ifp_srvc_dvc_model"
        , "ifp_srvc_dvc_term_start_date"
    )
    .distinct()
    .withColumn(
        "ifp_90d_srvc_flag"
        , f.lit('Y')
    )
)

df_tmp_excl_06 = (
    df_fs_ifp_bill
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .filter(f.col("ifp_bill_dvc_flag") == 'Y')
    .withColumn("datediff", f.datediff(f.current_date(), f.col("ifp_bill_dvc_term_start_date")))
    .filter(f.col("datediff") <= 90)
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy("fs_acct_id")
            .orderBy("datediff")
        )
    )
    .filter(f.col("index") == 1)
    .select(
        "fs_acct_id"
        , "ifp_bill_dvc_model"
        , "ifp_bill_dvc_term_start_date"
    )
    .distinct()
    .withColumn(
        "ifp_90d_bill_flag"
        , f.lit('Y')
    )
)

df_tmp_excl_07 = (
    df_fs_master
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .filter(f.col("network_dvc_model_brand") == 'Apple')
    .select("fs_srvc_id")
    .withColumn("dvc_brand_flag", f.lit('Y'))
    .distinct()
)

# COMMAND ----------

# MAGIC %md ### s203 ML score

# COMMAND ----------

df_base_score_ifp = (
    df_mls_score_ifp
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .select(
        "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , f.col("propensity_score").alias("ifp_score")
        , f.col("propensity_segment_qt").alias("ifp_segment")
        , f.col("propensity_top_ntile").alias("ifp_top_ntile")
    )
)

df_base_score_dr_samsung = (
    df_mls_score_dr_samsung
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .select(
        "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , f.col("propensity_score").alias("dr_samsung_score")
        , f.col("propensity_segment_qt").alias("dr_samsung_segment")
        , f.col("propensity_top_ntile").alias("dr_samsung_top_ntile")
    )
)


df_base_score_churn = (
    df_mls_score_churn
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .select(
        "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , f.col("propensity_score").alias("churn_score")
        , f.col("propensity_segment_qt").alias("churn_segment")
        , f.col("propensity_top_ntile").alias("churn_top_ntile")
    )
)

# COMMAND ----------

# MAGIC %md ### s204 proc candidate

# COMMAND ----------

df_proc_full = (
    df_base_full
    .select(
        "reporting_date"
        , "reporting_cycle_type"
        , "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , 'srvc_privacy_flag'
        , "plan_name_std"
        , "plan_legacy_flag"
        , "plan_amt"
        , "network_dvc_imei"
        , "network_dvc_tac"
        , "network_dvc_tenure"
        , "network_dvc_brand"
        , "network_dvc_model_marketing"
        , "network_dvc_os"
        , "network_dvc_first_used_date"
        , "network_dvc_release_tenure_year"
        , "network_dvc_release_tenure_year_group"
        , "network_dvc_first_used_from_release_year"
        , "network_dvc_first_used_from_release_year_group"
        , "ifp_prm_dvc_flag"
        , "ifp_prm_dvc_level"
        , "ifp_prm_dvc_type"
        , "ifp_prm_dvc_model"
        , "ifp_prm_dvc_term_start_date"
    )
    .withColumn(
        "network_dvc_brand_std"
        , f.when(
            f.lower(f.col("network_dvc_brand")).isin(["apple", "samsung"])
            , f.lower(f.col("network_dvc_brand"))
        ).otherwise(
            f.lit("others")
        )
    )
    .join(
        df_base_score_ifp
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_base_score_dr_samsung
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_base_score_churn
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_01
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_02
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_03
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_04
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_05
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_06
        , ["fs_acct_id"]
        , "left"
    )
    .join(
        df_tmp_excl_07
        , ["fs_srvc_id"]
        , 'left'
    )
    .fillna(
        value='N'
        , subset=['gc_curr_flag', 'gc_prev_flag', 'ch_target_flag', 'ch_control_flag', 'ifp_90d_srvc_flag', 'ifp_90d_bill_flag', 'dvc_brand_flag']
    )
    
    .withColumn(
        "target_segment"
        , f.when(
            f.col('srvc_privacy_flag') == 'N'
            , f.lit("z1.opt out")
        )
        .when(
            (
                (f.col("gc_curr_flag") == 'Y')
            )
            , f.lit("z2.global control - curr")
        )
        .when(
            (
                (f.col("gc_prev_flag") == 'Y')
            )
            , f.lit("z2.global control - prev")
        )
        .when(
            (
                f.col("ch_target_flag") == 'Y'
            )
            , f.lit("z3.campaign target")
        )
        .when(
            (
                f.col("ch_control_flag") == 'Y'
            )
            , f.lit("z4.campaign control")
        )
        .when(
            (
                (f.col('ifp_90d_srvc_flag') == 'Y')
                | (f.col('ifp_90d_bill_flag') == 'Y')
            )
            , f.lit("z5.<= ifp 90d")
        )
        .when(
            (
                f.col("dvc_brand_flag") == 'Y'
            )
            , f.lit('z6.apple')
        )
        .otherwise(f.lit("a.target"))
    )
)

# COMMAND ----------

display(
    df_proc_full
    .groupBy("target_segment")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
    )
    .orderBy("target_segment")
)

display(df_proc_full.limit(100))

# COMMAND ----------

display(
    df_base_full
    .filter(f.lower(f.col("network_dvc_model_marketing")).rlike("find"))
    .select('network_dvc_model_brand', "network_dvc_model_marketing")
    .distinct()
)

# COMMAND ----------

# MAGIC %md ### s205 campaign base

# COMMAND ----------

df_campaign_full = (
    df_proc_full
    .withColumn(
        "ifp_rank"
        , f.row_number().over(
            Window
            .partitionBy(f.lit(1))
            .orderBy(f.desc("ifp_score"))
        )
    )
    .withColumn(
        "campaign_segment_lvl1"
        , f.when(
            f.col("network_dvc_brand").isin(['Apple'])
            , f.lit("Apple")
        )
        .otherwise(f.lit("Android"))
    )
    .withColumn(
        "campaign_segment_lvl2"
        , f.when(
            (f.col("campaign_segment_lvl1") == 'Android')
            & (f.lower(f.col("network_dvc_model_marketing")).rlike("galaxy s23|galaxy z fold5|galaxy z flip5|reno 10|find n"))
            , f.lit("N")
        )
        .when(
            (f.col("campaign_segment_lvl1") == 'Android')
            , f.lit("Y")
        )
        .otherwise(f.lit("unknown"))
    )
    .withColumn(
        "campaign_segment_lvl3"
        , f.col("ifp_segment")
    )
    .withColumn(
        "campaign_segment_lvl4"
        , f.col("dr_samsung_segment")
    )
    .withColumn(
        "campaign_segment_lvl5"
        , f.col("churn_segment")
    )
    .withColumn(
        "campaign_cohort"
        , f.concat(f.col("campaign_segment_lvl1"), f.lit("-"), f.col("campaign_segment_lvl2"), f.lit("-"), f.col("campaign_segment_lvl3"))
    )
)


# COMMAND ----------

display(
    df_campaign_full
    .filter(f.col("campaign_segment_lvl2") == 'N')
    .groupBy("campaign_segment_lvl1", "network_dvc_model_marketing")
    .agg(f.countDistinct("fs_srvc_id"))
    .orderBy("campaign_segment_lvl1", "network_dvc_model_marketing")
)

# COMMAND ----------

display(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("campaign_segment_lvl2") == 'Y')
    .groupBy("campaign_cohort")
    .agg(f.countDistinct("fs_srvc_id"))
    .orderBy("campaign_cohort")
)

display(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("campaign_segment_lvl2") == 'Y')
    .groupBy("campaign_segment_lvl1", "campaign_segment_lvl2", "campaign_segment_lvl3")
    .agg(f.countDistinct("fs_srvc_id"))
    .orderBy("campaign_segment_lvl1", "campaign_segment_lvl2", "campaign_segment_lvl3")
)

# COMMAND ----------

(
    df_campaign_full
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_el/ml_campaigns/ifp/202312_dgc_2501_android') 
)

# COMMAND ----------

df_campaign_full = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_el/ml_campaigns/ifp/202312_dgc_2501_android")

# COMMAND ----------

display(
    df_campaign_full
    .limit(100)
)

# COMMAND ----------

# MAGIC %md ### s206 local control - Android H

# COMMAND ----------

# 1.5% conversion
vt_param_sample_req = 6600

df_campaign_cand_target = (
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("campaign_cohort") == 'Android-Y-H')
)

df_sample_extractor = get_sample_extractor(
    df_campaign_cand_target
    , vt_param_sample_req
)

df_sample_extractor = (
    df_sample_extractor
    .withColumn("sample_target", f.col("sample_req"))
)

df_campaign_cand_control = get_local_control(
    df_campaign_full
    .filter(f.col("target_segment").isin(['z1.opt out', 'z2.global control - curr', "z3.global - prev"]))
    .filter(f.col("campaign_cohort") == 'Android-Y-H')
    , df_sample_extractor
    , seed = 12
)

display(
    df_sample_extractor
    .join(
        df_campaign_cand_control
        .groupBy("ifp_top_ntile")
        .agg(f.countDistinct("fs_srvc_id").alias("sample_extract"))
        , ["ifp_top_ntile"]
        , "left"
    )
    .withColumn(
        "delta"
        , f.col("sample_req") - f.col("sample_extract")
    )
)

display(
    df_campaign_cand_control
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)

display(
    df_campaign_cand_target
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)

# COMMAND ----------

df_campaign_cand_target_01 = df_campaign_cand_target
df_campaign_cand_control_01 = df_campaign_cand_control

# COMMAND ----------

# MAGIC %md ### s209 local control - Android M

# COMMAND ----------

# 1% conversion
vt_param_sample_req = 8000

df_campaign_cand_target = (
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("campaign_cohort") == 'Android-Y-M')
)

df_sample_extractor = get_sample_extractor(
    df_campaign_cand_target
    , vt_param_sample_req
)

df_sample_extractor = (
    df_sample_extractor
    .withColumn("sample_target", f.col("sample_req"))
)

df_campaign_cand_control = get_local_control(
    df_campaign_full
    .filter(f.col("target_segment").isin(['z1.opt out', 'z2.global control - curr', "z3.global - prev"]))
    .filter(f.col("campaign_cohort") == 'Android-Y-M')
    , df_sample_extractor
    , seed = 12
)

display(
    df_sample_extractor
    .join(
        df_campaign_cand_control
        .groupBy("ifp_top_ntile")
        .agg(f.countDistinct("fs_srvc_id").alias("sample_extract"))
        , ["ifp_top_ntile"]
        , "left"
    )
    .withColumn(
        "delta"
        , f.col("sample_req") - f.col("sample_extract")
    )
)

display(
    df_campaign_cand_control
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)

display(
    df_campaign_cand_target
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)

# COMMAND ----------

df_campaign_cand_target_02 = df_campaign_cand_target
df_campaign_cand_control_02 = df_campaign_cand_control

# COMMAND ----------

# MAGIC %md ### s209 local control - Android L

# COMMAND ----------

# 1%
vt_param_sample_req = 15000
vt_param_target_req = 20000

df_campaign_cand_target = (
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("campaign_cohort") == 'Android-Y-L')
    #.filter(f.col("churn_segment").isin(["H", "M"]))
    .withColumn(
        "rank"
        , f.row_number().over(
            Window
            .partitionBy(f.lit(1))
            .orderBy(f.desc('dr_samsung_score'), f.desc("ifp_score"))
        )
    )
    .filter(f.col("rank") <= vt_param_target_req)
)

df_sample_extractor = get_sample_extractor(
    df_campaign_cand_target
    , vt_param_sample_req
)

df_sample_extractor = (
    df_sample_extractor
    .withColumn("sample_target", f.col("sample_req"))
)

df_campaign_cand_control = get_local_control(
    df_campaign_full
    .filter(f.col("target_segment").isin(['z1.opt out', 'z2.global control - curr', "z3.global - prev"]))
    .filter(f.col("campaign_cohort") == 'Android-Y-L')
    , df_sample_extractor
    , seed = 12
)

display(
    df_sample_extractor
    .join(
        df_campaign_cand_control
        .groupBy("ifp_top_ntile")
        .agg(f.countDistinct("fs_srvc_id").alias("sample_extract"))
        , ["ifp_top_ntile"]
        , "left"
    )
    .withColumn(
        "delta"
        , f.col("sample_req") - f.col("sample_extract")
    )
)

display(
    df_campaign_cand_control
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)

display(
    df_campaign_cand_target
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)

# COMMAND ----------

df_campaign_cand_target_03 = df_campaign_cand_target
df_campaign_cand_control_03 = df_campaign_cand_control

# COMMAND ----------

# MAGIC %md ### s210 campaign target finalisation

# COMMAND ----------

display(
    df_campaign_cand_target_01
    .limit(100)
)

# COMMAND ----------

def get_campaign_attr(
    df
):
    df_out = (
        df
        .withColumn(
            "campaign_segment_lvl4"
            ,f.col('dr_samsung_segment')
        )
        .select(
            f.col("reporting_date").alias("car_date")
            , "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , f.col("plan_name_std").alias("rateplan")
            , f.col("network_dvc_brand").alias("hand_brand")
            , f.col("network_dvc_model_marketing").alias("hand_prodtitle")
            , f.col("network_dvc_first_used_date").alias("hand_first_using_dt")
            , f.col("ifp_prm_dvc_term_start_date").alias("oa_if_start_dt")
            , f.col("ifp_prm_dvc_model").alias("oa_if_device")
            , f.lit("OA_IFP").alias("model")
            , f.lit("mobile_oa_consumer_srvc_upsell_ifp_v1").alias("model_version")
            , f.lit("2023-10-02").alias("model_base_date")
            , f.col("ifp_score").alias("propensity_score")
            , f.col("ifp_segment").alias("propensity_segment_qt")
            , f.col("ifp_rank").alias("propensity_rank")
            , "cohort"
            , "campaign_name"
            , "campaign_cohort"
            , "campaign_segment_lvl1"
            , "campaign_segment_lvl2"
            , "campaign_segment_lvl3"
            , "campaign_segment_lvl4"
            , "campaign_segment_lvl5"
        )
    )

    return df_out

# COMMAND ----------

def get_campaign_output(
    df_target
    , df_control
    , campaign_cohort
    , campaign_name
):
    df_out = (
        get_campaign_attr(
            df_target
            .withColumn(
                "cohort"
                , f.lit("TARGET")
            )
            .withColumn(
                "campaign_cohort"
                , f.lit(campaign_cohort)
            )
            .withColumn(
                "campaign_name"
                , f.lit(campaign_name)
            )
        )
        .unionByName(
            get_campaign_attr(
                df_control
                .withColumn(
                    "cohort"
                    , f.lit("LOCAL_CONTROL")
                )
                .withColumn(
                    "campaign_cohort"
                    , f.lit(campaign_cohort)
                )
                .withColumn(
                    "campaign_name"
                    , f.lit(campaign_name)
                )
            )
        )
    )

    return df_out
    

# COMMAND ----------

# MAGIC %md #### Android H

# COMMAND ----------

vt_param_campaign_cohort = 'Android-HP'
vt_param_campaign_name = 'DGC-2501'

df_input_target = df_campaign_cand_target_01
df_input_control = df_campaign_cand_control_01

df_output_campaign = get_campaign_output(
    df_input_target
    , df_input_control
    , vt_param_campaign_cohort
    , vt_param_campaign_name
)

df_output_campaign_01 = (
    df_output_campaign
    .withColumn(
        "campaign_name_load"
        , f.concat(
            f.col("campaign_name")
            , f.lit('-'), f.col("campaign_segment_lvl1")
            , f.lit("-"), f.col("campaign_segment_lvl2")
            , f.lit("-"), f.col("campaign_segment_lvl3")
            , f.lit("-"), f.col("campaign_segment_lvl4")
            , f.lit("-"), f.col("campaign_segment_lvl5")
        )
    )
)

display(
    df_output_campaign
    .groupBy("cohort")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
)

display(
    df_output_campaign
    .limit(100)
)

# COMMAND ----------

# MAGIC %md #### Android M

# COMMAND ----------

vt_param_campaign_cohort = 'Android-MP'
vt_param_campaign_name = 'DGC-2501'

df_input_target = df_campaign_cand_target_02
df_input_control = df_campaign_cand_control_02

df_output_campaign = get_campaign_output(
    df_input_target
    , df_input_control
    , vt_param_campaign_cohort
    , vt_param_campaign_name
)

df_output_campaign_02 = (
    df_output_campaign
    .withColumn(
        "campaign_name_load"
        , f.concat(
            f.col("campaign_name")
            , f.lit('-'), f.col("campaign_segment_lvl1")
            , f.lit("-"), f.col("campaign_segment_lvl2")
            , f.lit("-"), f.col("campaign_segment_lvl3")
            , f.lit("-"), f.col("campaign_segment_lvl4")
            , f.lit("-"), f.col("campaign_segment_lvl5")
        )
    )
)

display(
    df_output_campaign
    .groupBy("cohort")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
)

display(
    df_output_campaign
    .limit(100)
)

# COMMAND ----------

# MAGIC %md #### Android L

# COMMAND ----------

vt_param_campaign_cohort = 'Android-LP'
vt_param_campaign_name = 'DGC-2501'

df_input_target = df_campaign_cand_target_03
df_input_control = df_campaign_cand_control_03

df_output_campaign = get_campaign_output(
    df_input_target
    , df_input_control
    , vt_param_campaign_cohort
    , vt_param_campaign_name
)

df_output_campaign_03 = (
    df_output_campaign
    .withColumn(
        "campaign_name_load"
        , f.concat(
            f.col("campaign_name")
            , f.lit('-'), f.col("campaign_segment_lvl1")
            , f.lit("-"), f.col("campaign_segment_lvl2")
            , f.lit("-"), f.col("campaign_segment_lvl3")
            , f.lit("-"), f.col("campaign_segment_lvl4")
            , f.lit("-"), f.col("campaign_segment_lvl5")
        )
    )
)

display(
    df_output_campaign
    .groupBy("cohort")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
)

display(
    df_output_campaign
    .limit(100)
)

# COMMAND ----------

# MAGIC %md #### combine all

# COMMAND ----------

df_output_campaign = (
    df_output_campaign_01
    .unionByName(df_output_campaign_02)
    .unionByName(df_output_campaign_03)
)

display(
    df_output_campaign
    .groupBy("campaign_name", "campaign_cohort", "cohort")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
    .orderBy(f.desc("campaign_name"), "cohort")
)

display(
    df_output_campaign
    .groupBy("campaign_name", "campaign_cohort", "cohort", "campaign_segment_lvl3")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
    .orderBy(f.desc("campaign_name"), "cohort")
)

display(
    df_output_campaign
    .filter(f.col("cohort") == 'TARGET')
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
)

display(
    df_output_campaign
    .filter(f.col("cohort") == 'LOCAL_CONTROL')
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
)

# COMMAND ----------

display(
    df_output_campaign
    .groupBy("campaign_name", "campaign_name_load", "campaign_cohort", "cohort", "campaign_segment_lvl3")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
    .orderBy(f.desc("campaign_name"), "cohort")
)

# COMMAND ----------

display(
    df_output_campaign
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.count("*")
    )
)

# COMMAND ----------

display(
    df_output_campaign
    .limit(100)
)

# COMMAND ----------

display(
    df_output_campaign
    .filter(f.col("cohort") == 'TARGET')
    .groupBy("campaign_name", "campaign_cohort")
    .agg(
        f.countDistinct("fs_srvc_id")
    )
)

# COMMAND ----------

display(
    df_output_campaign
    .filter(f.col("fs_srvc_id") == '642108070960')
)

# COMMAND ----------

(
    df_output_campaign
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "ml_campaign_ifp_exp_202312_2501")
    .mode("overwrite")
    .save()
)

# COMMAND ----------

# MAGIC %md ## s300 marketo upload

# COMMAND ----------

# MAGIC %md ### comm upload

# COMMAND ----------

'''
INSERT INTO campaigncontact.mko.Marketo_BAU_Campaign_Load ([Source_CONCATRENATED],[Campaign_name],[campaign_date],[Variable_23])
SELECT 
	concat(TAB_A.CONN_ID, '|', TAB_A.PRIM_ACCS_NUM) AS SOURCE_CONCATRENATED
	, 'DGC-2501' AS CAMPAIGN_NAME
	, '2023-12-05' AS CAMPAIGN_DATE
	, concat(TAB_A.CAMPAIGN_NAME, '-', TAB_A.CAMPAIGN_SEGMENT_LVL1, '-', TAB_A.CAMPAIGN_SEGMENT_LVL4) AS VARIABLE_23
FROM STAGING.DBO.ML_IFP_EXP_202312_2501 TAB_A
LEFT JOIN campaigncontact.dbo.Marketo_BAU_Campaign_Load_read TAB_B
	ON concat(TAB_A.CONN_ID, '|', TAB_A.PRIM_ACCS_NUM) = TAB_B.Source_CONCATRENATED
	AND TAB_B.campaign_date = '2023-12-05'
WHERE TAB_B.Source_CONCATRENATED IS NULL AND TAB_A.COHORT = 'TARGET'
'''

# COMMAND ----------

'''

SELECT COUNT(DISTINCT SOURCE_CONCATRENATED), COUNT(*)
from campaigncontact.mko.Marketo_BAU_Campaign_Load
where Campaign_name = 'DGC-2501'

SELECT VARIABLE_23, COUNT(DISTINCT SOURCE_CONCATRENATED), COUNT(*)
from campaigncontact.mko.Marketo_BAU_Campaign_Load
where Campaign_name = 'DGC-2501'
GROUP BY VARIABLE_23

select 
	Source_CONCATRENATED
	, Campaign_name
	, campaign_date
	, Variable_23
from campaigncontact.mko.Marketo_BAU_Campaign_Load
where Campaign_name = 'DGC-2501'

'''

# COMMAND ----------

'''
INSERT INTO STAGING.DBO.ML_IFP_EXP_FACT
SELECT 
   CAR_DATE
    , CONN_ID
    , CUST_PARTY_ID
    , BILLING_ACCT_NUM
    , PRIM_ACCS_NUM
    , RATEPLAN
    , HAND_BRAND
    , HAND_PRODTITLE
    , HAND_FIRST_USING_DT
    , OA_IF_START_DT
    , OA_IF_DEVICE
    , MODEL
    , MODEL_VERSION
    , MODEL_BASE_DATE
    , PROPENSITY_SCORE
    , PROPENSITY_SEGMENT_QT
    , PROPENSITY_RANK
    , concat(CONN_ID, '|', PRIM_ACCS_NUM) AS SOURCE_CONCATRENATED
    , CAST('2022-11-24' as date) AS CAMPAIGN_DATE
    , CAMPAIGN_NAME
    , COHORT
    , GETDATE() AS DATA_CREATE_DATE
FROM STAGING.DBO.ML_IFP_EXP_202311_2501;
'''
