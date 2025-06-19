# Databricks notebook source
# MAGIC %md ### s000 environment setup

# COMMAND ----------

# DBTITLE 1,libraries
# libraries
import os

import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

# DBTITLE 1,utility functions 01
# MAGIC %run "../s04_resources/s01_utility_functions/utils_spark_df"

# COMMAND ----------

# DBTITLE 1,utility functions 02
# MAGIC %run "../s04_resources/s01_utility_functions/utils_stratified_sampling"

# COMMAND ----------

# DBTITLE 1,sf connection
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

# DBTITLE 1,directories 01
dir_fs_data_parent = "/mnt/feature-store-prod-lab"
dir_mls_data_parent = "/mnt/ml-store-prod-lab/classification"

# COMMAND ----------

# DBTITLE 1,directories 02
dir_mls_data_score = os.path.join(dir_mls_data_parent, "d400_model_score")

# COMMAND ----------

# DBTITLE 1,directories 03
dir_fs_data_meta = os.path.join(dir_fs_data_parent, 'd000_meta')
dir_fs_data_raw =  os.path.join(dir_fs_data_parent, 'd100_raw')
dir_fs_data_int =  os.path.join(dir_fs_data_parent, "d200_intermediate")
dir_fs_data_prm =  os.path.join(dir_fs_data_parent, "d300_primary")
dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")
dir_fs_data_target = os.path.join(dir_fs_data_parent, "d500_movement")
dir_fs_data_serv = os.path.join(dir_fs_data_parent, "d600_serving")

# COMMAND ----------

# MAGIC %md ### s100 data import

# COMMAND ----------

# DBTITLE 1,parameters 01
vt_param_reporting_date = "2024-09-29"
vt_param_reporting_cycle_type = "rolling cycle"

# COMMAND ----------

# DBTITLE 1,feature store 01
df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))

# COMMAND ----------

# DBTITLE 1,feature store 02
df_fs_id_master = (
    spark
    .read
    .format("delta")
    .load(os.path.join(dir_fs_data_prm, "d301_mobile_oa_consumer/prm_unit_base"))
    .select(
        "reporting_date"
        , "reporting_cycle_type"
        , "fs_cust_id"
        , "fs_acct_id"
        , "fs_acct_src_id"
        , "fs_srvc_id"
    )
)


# COMMAND ----------

# DBTITLE 1,ml store
#df_mls_score_dr_apple = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_apple_pred30d"))
#df_mls_score_dr_apple = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_apple_pred30d"))
df_mls_score_dr = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_pred30d"))
df_mls_score_ifp = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_upsell_ifp_pred30d"))
df_mls_score_churn = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_churn_pred30d"))
df_mls_score_ar = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_writeoff_pred120d"))

# COMMAND ----------

# DBTITLE 1,target log 01
df_campaign_hist_01 = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_el/ml_campaigns/wallet/240911-DH-MOBPM-Apple-NPI")

display(
    df_campaign_hist_01
    #.groupBy("campaign_name",'cohort')
    .agg(f.countDistinct("fs_srvc_id"))
)

display(df_campaign_hist_01.limit(100))

# COMMAND ----------

df_campaign_hist_02 = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_el/ml_campaigns/wallet/240906-DH-MOBPM-Apple-Stock-Clearance")

display(
    df_campaign_hist_02
    #.groupBy("campaign_name",'cohort')
    .agg(f.countDistinct("fs_srvc_id"))
)

display(df_campaign_hist_02.limit(100))

# COMMAND ----------

df_campaign_hist_03 = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_el/ml_campaigns/wallet/240924-DH-MOBPM-Apple-NPI-Small")

display(
    df_campaign_hist_03
    #.groupBy("campaign_name",'cohort')
    .agg(f.countDistinct("fs_srvc_id"))
)

display(df_campaign_hist_03.limit(100))

# COMMAND ----------

# MAGIC %md ### s104 global control group

# COMMAND ----------

df_gc_curr = spark.read.format("delta").load("/mnt/ml-store-dev/dev_users/dev_el/marketing_programs/global_control/mobile_oa_consumer")

df_gc_curr = (
    df_gc_curr
    .select("fs_srvc_id")
    .distinct()
)

print(df_gc_curr.count())

display(
    df_gc_curr
    .limit(10)
)

# COMMAND ----------

df_wallet_program_control = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select distinct service_id_sha2 as fs_srvc_id_sha2
            from prod_ml_store.serving.wallet_service_classification_propensity_score_fact
            where model_version in (
                'wallet_experiment_program_control_endless_small_20240924'
                , 'wallet_experiment_program_control_endless_medium_plus_20240924'
            )
        """
    )
    .load()
)

df_wallet_program_control = lower_col_names(df_wallet_program_control)


display(
    df_wallet_program_control
    #.groupBy("campaign_name",'cohort')
    .agg(f.countDistinct("fs_srvc_id_sha2"))
)

display(df_wallet_program_control.limit(100))


# COMMAND ----------

df_wallet_srvc = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select distinct
                wallet_id as fs_srvc_id
                , wallet_eligibility_flag
            from PROD_WALLET.MODELLED.D_WALLET_CUSTOMER
            where current_record_ind = 1
            and wallet_eligibility_flag = 'Y'
            and service_status_name = 'Active'
        """
    )
    .load()
)

df_wallet_srvc = lower_col_names(df_wallet_srvc)

display(df_wallet_srvc.limit(10))

display(
    df_wallet_srvc
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
    )
)


# COMMAND ----------

df_wallet_srvc_bal = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select wallet_id as fs_srvc_id
                , wallet_balance_earned
                , d_date_key as wallet_balance_date_key
            from PROD_WALLET.SERVING.E_SFMC_CUSTOMER_DAILY_SS
            qualify row_number() over (
                partition by wallet_id
                order by d_date_key desc
            ) = 1
        """
    )
    .load()
)

df_wallet_srvc_bal = lower_col_names(df_wallet_srvc_bal)

display(df_wallet_srvc_bal.limit(10))

display(df_wallet_srvc_bal.count())

display(
    df_wallet_srvc_bal
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
    )
)

# COMMAND ----------

# MAGIC %md ## s200 data processing

# COMMAND ----------

# MAGIC %md ### s201 base candidate

# COMMAND ----------

df_base_full = (
    df_fs_master
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .drop("srvc_privacy_flag")
    .join(
        df_fs_master
        .filter(f.col("reporting_date") == '2024-08-31')
        .filter(f.col('reporting_cycle_type') == "calendar cycle")
        .select("fs_cust_id", "fs_acct_id", "fs_srvc_id", "srvc_privacy_flag")
        , ["fs_cust_id", 'fs_acct_id', "fs_srvc_id"]
        , "left"
    )
    
    .join(
        df_fs_id_master
        , ["reporting_date", "reporting_cycle_type", "fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
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

# current active campaign
df_tmp_excl_02_01 = (
    df_campaign_hist_01
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "ch01_flag"
        , f.lit('Y')
    )
)

# current active campaign
df_tmp_excl_02_02 = (
    df_campaign_hist_02
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "ch02_flag"
        , f.lit('Y')
    )
)

df_tmp_excl_02_03 = (
    df_campaign_hist_03
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "ch03_flag"
        , f.lit('Y')
    )
)

df_tmp_excl_03 = (
    df_wallet_program_control
    .select("fs_srvc_id_sha2")
    .distinct()
    .withColumn(
        "wpc_flag"
        , f.lit('Y')
    )
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

df_base_score_dr = (
    df_mls_score_dr
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .select(
        "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , f.col("propensity_score").alias("dr_score")
        , f.col("propensity_segment_qt").alias("dr_segment")
        , f.col("propensity_top_ntile").alias("dr_top_ntile")
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

df_base_score_ar = (
    df_mls_score_ar
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .select(
        "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , f.col("propensity_score").alias("risk_score")
        , f.col("propensity_segment_qt").alias("risk_segment")
        , f.col("propensity_top_ntile").alias("risk_top_ntile")
    )
)

# COMMAND ----------

# MAGIC %md ### s204 proc candidate

# COMMAND ----------

display(df_base_full.limit(15))

# COMMAND ----------

df_proc_full = (
    df_base_full
    .select(
        "reporting_date"
        , "reporting_cycle_type"
        , "fs_cust_id"
        , "fs_acct_id"
        , 'fs_acct_src_id'
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
            f.lower(f.col("network_dvc_brand")).isin(["apple"])
            , f.lower(f.col("network_dvc_brand"))
        ).otherwise(
            f.lit("others")
        )
    )
    .withColumn(
        "fs_srvc_id_sha2"
        , f.sha2(f.col("fs_srvc_id"), 256)
    )
    .join(
        df_base_score_ifp
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_base_score_dr
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )

    .join(
        df_base_score_churn
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_base_score_ar
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_01
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_02_01
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_02_02
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_02_03
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_03
        , ["fs_srvc_id_sha2"]
        , "left"
    )
    .join(
        df_wallet_srvc
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_wallet_srvc_bal
        , ["fs_srvc_id"]
        , "left"
    )
    .fillna(
        value='N'
        , subset=['gc_curr_flag', 'ch01_flag', 'ch02_flag', 'ch03_flag', 'wpc_flag', 'wallet_eligibility_flag']
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
                (f.col("wpc_flag") == 'Y')
            )
            , f.lit("z3.wallet program control")
        )
        .when(
            (
                (f.col("ch01_flag") == 'Y')
                | (f.col("ch02_flag") == 'Y')
                | (f.col("ch03_flag") == 'Y')
            )
            , f.lit("z4.campaign")
        )
        .when(
            (
                f.col("wallet_eligibility_flag") == 'N'
            )
            , f.lit("z5.Non Wallet")
        )
        .when(
            (
                f.col("risk_top_ntile") >= 98
            )
            , f.lit("z6.High Bad Debt Risk")
        )
       
        .otherwise(f.lit("a.target"))
        
    )
    .withColumn("churn_top_ntile", f.ntile(30).over(Window.orderBy(f.desc("churn_score"))))
    .withColumn("dr_top_ntile", f.ntile(30).over(Window.orderBy(f.desc("dr_score"))))
    .withColumn("ifp_top_ntile", f.ntile(30).over(Window.orderBy(f.desc("ifp_score"))))
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
        "dr_rank"
        , f.row_number().over(
            Window
            .partitionBy(f.lit(1))
            .orderBy(f.desc("dr_score"))
        )
    )
    .withColumn(
        "churn_rank"
        , f.row_number().over(
            Window
            .partitionBy(f.lit(1))
            .orderBy(f.desc("churn_score"))
        )
    )
)


# COMMAND ----------

display(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy("churn_segment")
    .agg(f.countDistinct("fs_srvc_id").alias("conn"))
    .withColumn(
        "conn_tot"
        , f.sum("conn").over(Window.partitionBy(f.lit(1)))
    )
    .withColumn(
        "pct"
        , f.col("conn") / f.col("conn_tot")
    )
)

display(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy("dr_segment")
    .agg(f.countDistinct("fs_srvc_id").alias("conn"))
    .withColumn(
        "conn_tot"
        , f.sum("conn").over(Window.partitionBy(f.lit(1)))
    )
    .withColumn(
        "pct"
        , f.col("conn") / f.col("conn_tot")
    )
)

display(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy("ifp_segment")
    .agg(f.countDistinct("fs_srvc_id").alias("conn"))
    .withColumn(
        "conn_tot"
        , f.sum("conn").over(Window.partitionBy(f.lit(1)))
    )
    .withColumn(
        "pct"
        , f.col("conn") / f.col("conn_tot")
    )
)

# COMMAND ----------

# DBTITLE 1,interim export
(
    df_campaign_full
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/xx/ml_campaigns/wallet/xx202410_mobpm_non_apple_samsung_boost') 
)

# COMMAND ----------

df_campaign_full = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_el/ml_campaigns/wallet/xx202410_mobpm_non_apple_samsung_boost")

# COMMAND ----------

display(
    df_campaign_full
    .limit(100)
)

# COMMAND ----------

# MAGIC %md ### s206 local control - H

# COMMAND ----------

display(
    df_campaign_full
    .groupBy("srvc_privacy_flag")
    .agg(f.countDistinct("fs_srvc_id"))
)

# COMMAND ----------

# 1.5% conversion
vt_param_target_size = 35000
vt_param_control_size = 6000
ls_param_strata = ["churn_top_ntile", "ifp_top_ntile"]


df_sample_target = create_sample_target(
     df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(
        (f.col("churn_segment") == 'H')
        | (
            (f.col("churn_segment") == 'M')
            #& (f.col("dr_segment").isin(['H', "M"]))
        ) 
    )
    .filter(f.col("network_dvc_brand_std") != 'apple')
    #.filter(f.col("wallet_eligibility_flag") == 'Y')
    , ls_param_strata
)

#display(df_sample_target.limit(10))

df_campaign_cand_control = find_similar_sample(
    df_campaign_full
    .filter(f.col("target_segment") != 'a.target')
    .filter(f.col("ch01_flag") != 'Y')
    .filter(f.col("ch02_flag") != 'Y')
    .filter(f.col("ch03_flag") != 'Y')
    , size = vt_param_control_size
    , strata = ls_param_strata
    , df_target = df_sample_target
)

df_campaign_cand_target = find_similar_sample(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("network_dvc_brand_std") != 'apple')
    #.filter(f.col("wallet_eligibility_flag") == 'Y')
    , size = vt_param_target_size
    , strata = ls_param_strata
    , df_target = df_sample_target
)


print("control")
display(
    df_campaign_cand_control
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("churn_score")
        , f.avg("churn_score")
        , f.median("dr_score")
        , f.avg("dr_score")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)


print("target")
display(
    df_campaign_cand_target
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("churn_score")
        , f.avg("churn_score")
        , f.median("dr_score")
        , f.avg("dr_score")
        , f.median("ifp_score")
        , f.avg("ifp_score")
    )
)

# COMMAND ----------

# MAGIC %md ### s210 campaign target finalisation

# COMMAND ----------

def get_campaign_attr (
    df
):
    df_out = (
        df
        .select(
            "reporting_date"
            , "reporting_cycle_type"
            , "fs_cust_id"
            , "fs_acct_id"
            , "fs_acct_src_id"
            , "fs_srvc_id"
            , "plan_name_std"
            , "ifp_prm_dvc_flag"
            , "ifp_score"
            , "ifp_segment"
            , "ifp_top_ntile"
            , "dr_score"
            , "dr_segment"
            , "dr_top_ntile"
            , "churn_score"
            , "churn_segment"
            , "churn_top_ntile"
            , "risk_score"
            , "risk_segment"
            , "risk_top_ntile"
            , "wallet_eligibility_flag"
            , "wallet_balance_earned"
            , "wallet_balance_date_key"
            , "campaign"
            , "offer"
            , "campaign_cohort"
            , "target_cohort"
        )
    )

    return df_out

# COMMAND ----------


df_output_campaign_01 = get_campaign_attr(
    df_campaign_cand_target
    .withColumn(
        "offer"
        , f.lit(700)
    )
    .withColumn(
        "campaign",
        f.lit("241003-RM-MOBPM-Samsung-LTO-Offer")
    )
    .withColumn(
        "campaign_cohort",
        f.lit("241003-RM-MOBPM-Samsung-LTO-Offer")
    )
    .withColumn(
        "target_cohort",
        f.lit("Target")
    )
)


df_output_campaign_02 = get_campaign_attr(
    df_campaign_cand_control
    .withColumn(
        "offer"
        , f.lit(0)
    )
    .withColumn(
        "campaign"
        , f.lit("241003-RM-MOBPM-Samsung-LTO-Offer")
    )
    .withColumn(
        "campaign_cohort"
        , f.lit("Local Control")
    )
    .withColumn(
        "target_cohort"
        , f.lit("Local Control")
    )
)

df_output_campaign = (
    df_output_campaign_01
    .unionByName(df_output_campaign_02)
)

display(
    df_output_campaign
    .groupBy("campaign", "campaign_cohort", "offer")
    .agg(
        f.count("*"),
        f.countDistinct("fs_srvc_id"),
        f.median("churn_score"),
        f.avg("churn_score"),
        f.median("dr_score"),
        f.avg("dr_score"),
        f.median("ifp_score"),
        f.avg("ifp_score")
    )
    .orderBy("campaign", f.desc("offer"))
)


# COMMAND ----------

# MAGIC %md #### combine all

# COMMAND ----------

df_output_campaign = (
    df_output_campaign
)

display(df_output_campaign.limit(10))

display(
    df_output_campaign
    .groupBy("campaign", "target_cohort",  "offer", "campaign_cohort")
    .agg(
        f.count("*"),
        f.countDistinct("fs_srvc_id"),
        f.median("churn_score"),
        f.avg("churn_score"),
        f.median("dr_score"),
        f.avg("dr_score"),
        f.median("ifp_score"),
        f.avg("ifp_score")
    )
    .orderBy("campaign", f.desc("offer"))
)

display(
    df_output_campaign
    .groupBy("campaign")
    .agg(
       f.count("*"),
        f.countDistinct("fs_srvc_id"),
        f.median("churn_score"),
        f.avg("churn_score"),
        f.median("dr_score"),
        f.avg("dr_score"),
        f.median("ifp_score"),
        f.avg("ifp_score")
    )
    .orderBy("campaign")
)


# COMMAND ----------

# DBTITLE 1,export lake file
(
    df_output_campaign
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/xx/ml_campaigns/wallet/241003-RM-MOBPM-Samsung-LTO-Offer') 
)

# COMMAND ----------

# DBTITLE 1,export sf
(
    df_output_campaign
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "xxml_campaign_wallet_exp_202410_rm_mobpm_samsung_lto_offer")
    .mode("overwrite")
    .save()
)

# COMMAND ----------

# MAGIC %md ## s300 SFMC upload

# COMMAND ----------

df_output_campaign = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_el/ml_campaigns/wallet/241003-RM-MOBPM-Samsung-LTO-Offer")

# COMMAND ----------

df_sfmc_export = (
    df_output_campaign
    .filter(f.col("target_cohort") != "Local Control")
    .withColumn(
        "wallet_balance_final"
        , f.col("wallet_balance_earned") + f.col("offer")
    )
    .select(
        f.col("fs_cust_id").alias("contact_key")
        , f.col("fs_acct_id").alias("billing_account_number")
        , f.col('fs_srvc_id').alias("service_id")
        , f.col("campaign")
        , f.current_date().alias("data_update_date")
    )
)

display(
    df_sfmc_export
    .groupBy("campaign")
    .agg(
        f.count("*")
        , f.countDistinct("service_id")
    )
    .orderBy("campaign")
)

display(df_sfmc_export.limit(10))

# COMMAND ----------

# DBTITLE 1,export sfmc
dir_data_sfmc = "/mnt/prod_sfmc/imports/DataAnalytics/"

(
    df_sfmc_export
    .toPandas()
    .to_csv(f"/dbfs{dir_data_sfmc}/xx241003-RM-MOBPM-Samsung-LTO-Offer.csv", index=False)
)

# COMMAND ----------

f"/dbfs{dir_data_sfmc}/241003-RM-MOBPM-Samsung-LTO-Offer.csv"

# COMMAND ----------

df_check = spark.read.options(header=True).csv(f"{dir_data_sfmc}/241003-RM-MOBPM-Samsung-LTO-Offer.csv")

# COMMAND ----------

display(df_check.limit(100))
