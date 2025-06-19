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

# MAGIC %run "./spkdf_utils"

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

vt_param_reporting_date = "2024-06-02"
vt_param_reporting_cycle_type = "rolling cycle"

# COMMAND ----------

# MAGIC %md ### s101 feature store

# COMMAND ----------

df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
# df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
# df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))

# COMMAND ----------

df_fs_non_bb_base = spark.read.format("delta").load("/mnt/feature-store-dev/dev_users/dev_rz_pp/d999_tmp/xsell_bb_base")

display(
    df_fs_non_bb_base
    .filter(f.col("fs_cust_id") == "1-1D0ZXX25")
)

# COMMAND ----------

df_fs_unit_base = (
    spark
    .read
    .format("delta")
    .load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_unit_base"))
    .filter(f.col("reporting_cycle_type") == 'rolling cycle')
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .select("reporting_date","reporting_cycle_type","fs_cust_id","fs_acct_id","fs_srvc_id","product_holding_desc")
)
display(df_fs_unit_base.limit(3))

# COMMAND ----------

display(df_fs_unit_base.filter(f.col("fs_cust_id") == "1-1D0ZXX25").limit(3))

# COMMAND ----------

display(
    df_fs_unit_base
    .agg(
        f.count("*")
        , f.countDistinct('fs_srvc_id')
        , f.countDistinct('fs_cust_id')
        , f.countDistinct('fs_acct_id')
    )
)

# COMMAND ----------

# MAGIC %md ### s102 ml store

# COMMAND ----------

#df_mls_score_dr_apple = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_apple_pred30d"))
df_mls_score_upsell_plan = spark.read.format("delta").load("/mnt/feature-store-dev/dev_users/dev_rz/d999_testing/score_data_params_mobile_oa_consumer_srvc_upsell_plan_endless_pred30d_202405_exp2_v1_20240602")
df_mls_score_xsell_bb = spark.read.format("delta").load("/mnt/feature-store-dev/dev_users/dev_rz/d999_testing/score_data_params_mobile_oa_consumer_srvc_xsell_bb_pred30d_202405_exp1_v1_20240602")
# df_mls_score_ifp = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_upsell_ifp_pred30d"))
df_mls_score_churn = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_churn_pred30d"))

# COMMAND ----------

# MAGIC %md ### s103 target log

# COMMAND ----------

df_oreo = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select *
            , service_id as fs_srvc_id 
            , contact_key as fs_cust_id
            from LAB_ML_STORE.SANDBOX.LEADS_PROJECT_OREO
        """
    )
    .load()
)

df_oreo = lower_col_names(df_oreo)

display(df_oreo.limit(10))


# COMMAND ----------

df_campaign_upsell = (
    spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_rz/ml_campaigns/plan_upsell_endless/202406_plan_upsell")
    .filter(f.col("target_segment") == 'a.target')
)

display(df_campaign_upsell.groupBy("target_segment").count().limit(10))

# COMMAND ----------

df_survey = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select *
            , customer_id as fs_cust_id
            from LAB_ML_STORE.SANDBOX.LEADS_PRAPHAN_SURVEY
        """
    )
    .load()
)

df_survey = lower_col_names(df_survey)

display(df_survey.limit(10))


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
                , customer_source_id as fs_cust_id

            from lab_ml_store.sandbox.MOBILE_GLOBAL_CONTROL_202406
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
                , customer_source_id as fs_cust_id

            from lab_ml_store.sandbox.MOBILE_GLOBAL_CONTROL_202405
        """
    )
    .load()
)

df_gc_curr = lower_col_names(df_gc_curr)
df_gc_prev = lower_col_names(df_gc_prev)

display(df_gc_curr.limit(100))
display(df_gc_prev.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### s105 wallet customer

# COMMAND ----------

df_wallet_cust = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select 
                billing_account_number as fs_acct_id
                , MAX(wallet_id) as wallet_id
                , 'Y' as wallet_eligible_flag
            from PROD_WALLET.MODELLED.D_WALLET_CUSTOMER
            where current_record_ind = 1
            and wallet_eligibility_flag = 'Y'
            and service_status_name = 'Active'
            group by 1
        """
    )
    .load()
)

df_wallet_cust = lower_col_names(df_wallet_cust)

display(df_wallet_cust.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ### s106 email address filter

# COMMAND ----------

df_martech_cust = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select *
            , customer_id as fs_cust_id
            from PROD_MAR_TECH.SERVING.EXPORT_CUSTOMER
        """
    )
    .load()
)

df_martech_cust = lower_col_names(df_martech_cust)

display(df_martech_cust.limit(10))

display(df_martech_cust.groupBy("customer_id").count().filter(f.col("count")>1))

# COMMAND ----------

df_martech_bill = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select *
            , customer_id as fs_cust_id
            , billing_account_number as fs_acct_id
            from PROD_MAR_TECH.SERVING.EXPORT_BILL_TO
        """
    )
    .load()
)

df_martech_bill = lower_col_names(df_martech_bill)

display(
    df_martech_bill
    .filter(f.col("fs_acct_id") == "413426010")
    .limit(10)
)

display(
    df_martech_bill
    .groupBy("fs_acct_id")
    .agg(
        f.count('*').alias("count")
        , f.collect_list("customer_id").alias("customer_ids")
    )
    .filter(f.col("count")>1)
)

# COMMAND ----------

df_martech_service = (
     spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select *
            , customer_id as fs_cust_id
            from PROD_MAR_TECH.SERVING.EXPORT_SERVICE
        """
    )
    .load()
)

df_martech_service = lower_col_names(df_martech_service)

display(
    df_martech_service
    # .groupBy("service_id")
    # .count()
    # .filter(f.col("count")>1)
    .limit(10)
)

# COMMAND ----------

df_martech = (
    df_martech_bill
    .groupBy("fs_acct_id")
    .agg(
        f.max("fs_cust_id").alias("fs_cust_id")
        , f.max("billing_profile_email").alias("billing_profile_email")
    )
    .distinct()
    .join(
        df_martech_cust
        .groupBy("fs_cust_id")
        .agg(
            f.max("customer_primary_contact_email").alias("customer_primary_contact_email")
        )
        , ["fs_cust_id"]
        , "left"
    )
    .filter(f.col("fs_acct_id").isNotNull())
    .withColumn(
        "email_address"
        , f.when(f.col("customer_primary_contact_email") == "Unknown", f.col("billing_profile_email"))
        .otherwise(
            f.coalesce(f.col("customer_primary_contact_email"),f.col("billing_profile_email"))
        )
    )
)

display(df_martech.limit(100))

display(df_martech.filter(f.col("email_address").isNull()).count())

display(df_martech.groupBy("fs_acct_id").count().filter(f.col("count")>1))

# COMMAND ----------

# MAGIC %md ## s200 data processing

# COMMAND ----------

# MAGIC %md ### s201 base candidate

# COMMAND ----------

display(df_fs_non_bb_base.limit(10))

# COMMAND ----------

df_base_full = (
    df_fs_master
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .filter(f.col("plan_discount_flag_lop") == "N")
    .join(
        df_fs_non_bb_base
        , ['reporting_date','fs_acct_id','fs_cust_id','fs_srvc_id']
        , 'inner'
    )
    .join(
        df_fs_unit_base
        , ['reporting_date','reporting_cycle_type','fs_acct_id','fs_cust_id','fs_srvc_id']
        , 'inner'
    )
    .filter(~f.col("product_holding_desc").ilike("%broadband%"))
    
    # # check small
    # .filter(f.col("plan_name_std").isin("small mobile"))
)

display(
    df_base_full
    .limit(100)
)

display(
    df_base_full
    .groupBy('product_holding_desc')
    .count()
)

# COMMAND ----------

# MAGIC %md ### s202 exclusion flag

# COMMAND ----------

# current global control
df_tmp_excl_01 = (
    df_gc_curr
    .select("fs_cust_id")
    .distinct()
    .withColumn(
        "gc_curr_flag"
        , f.lit('Y')
    )
)

# previous global control
df_tmp_excl_02 = (
    df_gc_prev
    .select("fs_cust_id")
    .distinct()
    .withColumn(
        "gc_prev_flag"
        , f.lit('Y')
    )
)

# exclude audience from discount campaign
df_tmp_excl_03 = (
    df_oreo
    .select("fs_cust_id")
    .distinct()
    .withColumn(
        "oreo_campaign_flag"
        , f.lit('Y')
    )
)

# exclude audience from upsell campaign
df_tmp_excl_04 = (
    df_campaign_upsell
    .select("fs_srvc_id")
    .distinct()
    .withColumn(
        "upsell_campaign_flag"
        , f.lit('Y')
    )
)

# current H churn propensity
df_tmp_excl_05 = (
    df_mls_score_churn
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .filter(f.col("propensity_segment_qt").isin("H"))
    .select("fs_acct_id")
    .distinct()
    .withColumn(
        "churn_h_flag"
        , f.lit('Y')
    )
)

# customer without emails
df_tmp_excl_06 = (
    df_martech
    .filter(f.col("email_address") == "Unknown")
    .select("fs_acct_id")
    .distinct()
    .withColumn(
        "no_email_flag"
        , f.lit('Y')
    )
)

# customer from survey
df_tmp_excl_07 = (
    df_survey
    .select("fs_cust_id")
    .distinct()
    .withColumn(
        "survey_flag"
        , f.lit('Y')
    )
)



# COMMAND ----------

# display(df_mls_score_churn
#     .filter(f.col("reporting_date") == vt_param_reporting_date)
#     .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
#     .filter(f.col("propensity_segment_qt").isin("H"))
#     .count())

# COMMAND ----------

# MAGIC %md ### s203 ML score

# COMMAND ----------

df_base_score_xsell_bb = (
    df_mls_score_xsell_bb
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .select(
        "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , f.col("propensity_score").alias("xsell_score")
        , f.col("propensity_segment_qt").alias("xsell_segment")
        , f.col("propensity_top_ntile").alias("xsell_top_ntile")
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

df_base_score_upsell_plan = (
    df_mls_score_upsell_plan
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .select(
        "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , f.col("propensity_score").alias("plan_upsell_score")
        , f.col("propensity_segment_qt").alias("plan_upsell_segment")
        , f.col("propensity_top_ntile").alias("plan_upsell_top_ntile")
    )
)

# COMMAND ----------

display(df_base_score_churn.groupBy('churn_segment').count())

# COMMAND ----------

display(df_base_full.limit(10))

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
        , "num_of_active_srvc_cnt"
        , "plan_name_std"
        , "plan_legacy_flag"
        , "plan_amt"
        , "plan_discount_flag"
        , "plan_discount_cnt"
        , "plan_discount_amt"
 
    )
    .join(
        df_base_score_xsell_bb
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_base_score_upsell_plan
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )

    .join(
        df_base_score_churn
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_wallet_cust
        , "fs_acct_id"
        , "left"
    )
    .join(
        df_tmp_excl_01
        , ["fs_cust_id"]
        , "left"
    )
    .join(
        df_tmp_excl_02
        , ["fs_cust_id"]
        , "left"
    )
    .join(
        df_tmp_excl_03
        , ["fs_cust_id"]
        , "left"
    )
    .join(
        df_tmp_excl_04
        , ["fs_srvc_id"]
        , "left"
    )
    .join(
        df_tmp_excl_05
        , ["fs_acct_id"]
        , "left"
    )
    .join(
        df_tmp_excl_06
        , ["fs_acct_id"]
        , "left"
    )
    .join(
        df_tmp_excl_07
        , ["fs_cust_id"]
        , "left"
    )
    .fillna(
        value='N'
        , subset=[
            'gc_curr_flag', 'gc_prev_flag'
            , 'upsell_campaign_flag','oreo_campaign_flag'
            , 'churn_h_flag'
            , 'wallet_eligible_flag'  #,'wallet_member_flag'
            , "no_email_flag"
            , "survey_flag"
        ]
    )
    .fillna(
        value=0
        , subset=['plan_upsell_score','plan_upsell_top_ntile']
    )
    .fillna(
        value='L'
        , subset=['plan_upsell_segment']
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
                (f.col("oreo_campaign_flag") == 'Y')
            )
            , f.lit("z3.oreo campaign")
        )
        .when(
            (
                (f.col("upsell_campaign_flag") == 'Y')
            )
            , f.lit("z4.plan upsell campaign")
        )
        .when(
            (
                (f.col("churn_h_flag") == 'Y')
            )
            , f.lit("z5.churn risk - h")
        )
        .when(
            (
                (f.col("no_email_flag") == 'Y')
            )
            , f.lit("z6.no primary email")
        )
        .when(
            (
                (f.col("survey_flag") == 'Y')
            )
            , f.lit("z7.has survey")
        )
        .otherwise(f.lit("a.target"))
    )
    .filter(f.col("xsell_segment").isNotNull())
)

# COMMAND ----------

# TODO: add converge group
display(
    df_proc_full
    .groupBy("target_segment")
    .pivot("wallet_eligible_flag")
    .agg(
        # f.count("*")
        # , f.countDistinct("fs_srvc_id")
        f.countDistinct("fs_acct_id")
        # , f.countDistinct("fs_cust_id")
    )
    .orderBy("target_segment")
)

display(
    df_proc_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy('wallet_eligible_flag')
    .agg(
        # f.count("*")
        # , f.countDistinct("fs_srvc_id")
        f.countDistinct("fs_acct_id")
        # , f.countDistinct("fs_cust_id")
    )
    .orderBy('wallet_eligible_flag')
)

display(df_proc_full.limit(100))

# COMMAND ----------

# TODO: add converge group
display(
    df_proc_full
    .groupBy("target_segment")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.countDistinct("fs_acct_id")
        , f.countDistinct("fs_cust_id")
    )
    .orderBy("target_segment")
)

display(
    df_proc_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy('wallet_eligible_flag')
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.countDistinct("fs_acct_id")
        , f.countDistinct("fs_cust_id")
    )
    .orderBy('wallet_eligible_flag')
)

display(df_proc_full.limit(100))

# COMMAND ----------

# MAGIC %md ### s205 campaign base

# COMMAND ----------

df_campaign_full = (
    df_proc_full
    .withColumn(
        "xsell_rank"
        , f.row_number().over(
            Window
            .partitionBy()
            .orderBy(f.desc("xsell_score"))
        )
    )
    .withColumn(
        "campaign_cohort"
        # , f.col("wallet_eligible_flag")
        , f.concat(f.lit("wallet_eligible"), f.lit('-'), f.col("wallet_eligible_flag"))
    )
    .withColumn(
        "primary_rank"
        , f.row_number().over(
            Window
            .partitionBy("fs_acct_id","campaign_cohort")
            .orderBy(f.desc("target_segment"), f.desc("xsell_score"), f.desc("plan_amt"))
        )
    )
)


# COMMAND ----------

display(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy("campaign_cohort","xsell_segment")
    .agg(
        f.countDistinct("fs_acct_id").alias("account_count")
        , f.avg("xsell_score").alias("xsell_score")
    )
    .withColumn("Pct", 100*f.col("account_count")/f.sum(f.col("account_count")).over(Window.partitionBy("campaign_cohort")))
    .orderBy("campaign_cohort")
)


# COMMAND ----------

(
    df_campaign_full
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_rz/ml_campaigns/xsell/202405_xsell_bb') 
)

# COMMAND ----------

df_campaign_full = (
    spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_rz/ml_campaigns/xsell/202405_xsell_bb")
    # .filter(f.col("xsell_top_ntile")>= 78)
)

# COMMAND ----------

# check
display(
    df_campaign_full
    .filter(f.col("fs_acct_id") == "501471149")
    .limit(100)
)

# COMMAND ----------

# df_campaign_full = (
#     df_campaign_full
#     .filter(f.col("primary_rank") == 1)
# )

# COMMAND ----------

# MAGIC %md ### s206 local control - Wallet Eligible

# COMMAND ----------

# 1.08% conversion; 2.7 times uplift
vt_param_sample_req = 18000

df_campaign_cand_target = (
    df_campaign_full
    .filter(f.col("campaign_cohort") == "wallet_eligible-Y")
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("primary_rank") == 1)
    .filter(f.col("xsell_top_ntile")> 56)
)

df_sample_extractor = get_sample_extractor(
    df_campaign_cand_target
    , 'xsell_top_ntile'
    , vt_param_sample_req
)

df_sample_extractor = (
    df_sample_extractor
    .withColumn("sample_target", f.col("sample_req"))
)

df_campaign_cand_control = get_local_control(
    df_campaign_full
    .filter(f.col("target_segment").isin(['z1.opt out', 'z2.global control - curr', "z3.global - prev"]))
    .filter(f.col("campaign_cohort") == "wallet_eligible-Y")
    .filter(f.col("xsell_top_ntile")> 56)
    .filter(f.col("primary_rank") == 1)
    , df_sample_extractor
    , 'xsell_top_ntile'
    , seed = 12
)

display(
    df_sample_extractor
    .join(
        df_campaign_cand_control
        .groupBy("xsell_top_ntile")
        .agg(f.countDistinct("fs_srvc_id").alias("sample_extract"))
        , ["xsell_top_ntile"]
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
        , f.countDistinct("fs_acct_id")
        , f.median("xsell_score")
        , f.avg("xsell_score")
        , f.median("xsell_top_ntile")
    )
)

display(
    df_campaign_cand_target
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.countDistinct("fs_acct_id")
        , f.median("xsell_score")
        , f.avg("xsell_score")
        , f.median("xsell_top_ntile")
    )
)


# COMMAND ----------

df_campaign_cand_target_01 = df_campaign_cand_target
df_campaign_cand_control_01 = df_campaign_cand_control

# COMMAND ----------

# check
display(
    df_campaign_cand_target_01
    .join(
        df_campaign_cand_control_01
        , 'fs_acct_id'
        , 'inner'
    )
    .count()
)

# COMMAND ----------

# MAGIC %md ### s209 local control - Wallet Ineligible

# COMMAND ----------

display(df_campaign_cand_target)

# COMMAND ----------

# 0.8 % conversion; 2 times uplift
vt_param_sample_req = 20000

df_campaign_cand_target = (
    df_campaign_full
    .filter(f.col("campaign_cohort") == "wallet_eligible-N")
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("primary_rank") == 1)
    .filter(f.col("xsell_top_ntile")> 17)
)

df_sample_extractor = get_sample_extractor(
    df_campaign_cand_target
    , 'xsell_top_ntile'
    , vt_param_sample_req
)

df_sample_extractor = (
    df_sample_extractor
    .withColumn("sample_target", f.col("sample_req"))
)

df_campaign_cand_control = get_local_control(
    df_campaign_full
    .filter(f.col("target_segment").isin(['z1.opt out', 'z2.global control - curr', "z3.global - prev"]))
    .filter(f.col("campaign_cohort") == "wallet_eligible-N")
    .filter(f.col("primary_rank") == 1)
    .filter(f.col("xsell_top_ntile")> 17)
    , df_sample_extractor
    , 'xsell_top_ntile'
    , seed = 12
)

display(
    df_sample_extractor
    .join(
        df_campaign_cand_control
        .groupBy("xsell_top_ntile")
        .agg(f.countDistinct("fs_srvc_id").alias("sample_extract"))
        , ["xsell_top_ntile"]
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
        , f.median("xsell_score")
        , f.avg("xsell_score")
        , f.median("xsell_top_ntile")
    )
)

display(
    df_campaign_cand_target
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.median("xsell_score")
        , f.avg("xsell_score")
        , f.median("xsell_top_ntile")
    )
)

# COMMAND ----------

df_campaign_cand_target_02 = df_campaign_cand_target
df_campaign_cand_control_02 = df_campaign_cand_control

# COMMAND ----------

display(
    df_campaign_cand_target_01
    .filter(f.col("fs_acct_id") == "1031589")
    .limit(100)
)

display(
    df_campaign_cand_target_02
    .filter(f.col("fs_acct_id") == "1031589")
    .limit(100)
)

# COMMAND ----------

# check
display(
    df_campaign_cand_target_02
    .select("fs_acct_id", f.lit(1).alias("ind"))
    .union(df_campaign_cand_control_02.select("fs_acct_id", f.lit(2).alias("ind")))
    .union(df_campaign_cand_target_01.select("fs_acct_id", f.lit(3).alias("ind")))
    .union(df_campaign_cand_control_01.select("fs_acct_id", f.lit(4).alias("ind")))
    .groupBy("fs_acct_id")
    .count()
    .filter(f.col("count")>1)
)

# COMMAND ----------

# MAGIC %md ### s210 campaign target finalisation

# COMMAND ----------

display(
    df_campaign_cand_target_01
    .limit(100)
)
display(
    df_campaign_cand_target_02
    .limit(100)
)

# COMMAND ----------

def get_campaign_attr(
    df
):
    df_out = (
        df
        .select(
            f.col("reporting_date")
            , "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , f.col("plan_name_std")
            , 'churn_segment'
            , 'plan_upsell_segment'
            , f.lit("OA_XSELL_BB").alias("model")
            , f.lit("mobile_oa_consumer_srvc_xsell_bb_pred30d_v0").alias("model_version")
            , f.lit(vt_param_reporting_date).alias("model_base_date")
            , f.col("xsell_score").cast('double').alias("propensity_score")
            , f.col("xsell_top_ntile").cast('double').alias("propensity_top_ntile")
            , f.col("xsell_rank").cast('double').alias("propensity_rank")
            , f.col("xsell_segment")
            , "cohort"
            , "campaign_name"
            , "campaign_cohort"
            , "wallet_id"
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

# MAGIC %md #### Xsell Wallet Eligible
# MAGIC

# COMMAND ----------

vt_param_campaign_cohort = 'wallet_eligible-Y'
vt_param_campaign_name = 'XS-20240605'

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
        )
    )
)

# display(
#     df_output_campaign
#     .groupBy("cohort")
#     .agg(
#         f.count("*")
#         , f.countDistinct("fs_srvc_id")
#         , f.median("propensity_score")
#         , f.avg("propensity_score")
#         , f.median("propensity_top_ntile")
#     )
# )

display(
    df_output_campaign
    .limit(100)
)

display(
    df_output_campaign
    .groupBy("cohort")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
        , f.median("propensity_top_ntile")
        , f.min("propensity_top_ntile")
    )
)

# COMMAND ----------

# MAGIC %md #### Wallet not eligible

# COMMAND ----------

vt_param_campaign_cohort = 'wallet_eligible-N'
vt_param_campaign_name = 'XS-20240605'

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
        )
    )
)

# display(
#     df_output_campaign
#     .groupBy("cohort")
#     .agg(
#         f.count("*")
#         , f.countDistinct("fs_srvc_id")
#         , f.median("propensity_score")
#         , f.avg("propensity_score")
#     )
# )

display(
    df_output_campaign
    .limit(100)
)

display(
    df_output_campaign
    .groupBy("cohort")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
        , f.median("propensity_top_ntile")
        , f.min("propensity_top_ntile")
    )
)

# COMMAND ----------

# MAGIC %md #### combine all

# COMMAND ----------

df_output_campaign = (
    df_output_campaign_01
    .unionByName(df_output_campaign_02)
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
    .orderBy(f.desc("campaign_name"), "cohort","campaign_cohort")
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
    .groupBy("campaign_name", "campaign_name_load", "campaign_cohort", "cohort")
    .agg(
        f.count("*")
        , f.countDistinct("fs_acct_id")
        , f.countDistinct("fs_srvc_id")
        , f.median("propensity_score")
        , f.avg("propensity_score")
    )
    .orderBy(f.desc("campaign_name"), "campaign_cohort", f.desc("cohort"))
)

# COMMAND ----------

display(
    df_output_campaign
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.countDistinct("fs_acct_id")
        , f.count("*")
    )
)

# COMMAND ----------

display(
    df_output_campaign
    .limit(10)
)

display(
    df_output_campaign
    .groupBy("churn_segment","plan_upsell_segment","cohort")
    .agg(
        f.countDistinct("fs_srvc_id")
        , f.count("*")
    )
)

# COMMAND ----------

# check
display(
    df_output_campaign
    .groupBy('fs_srvc_id')
    .count()
    .filter(f.col('fs_srvc_id')>1)
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
    .filter(f.col("cohort") == 'TARGET')
    .filter(~f.col("plan_name_std").ilike("%small%"))
    .groupBy("campaign_name", "campaign_cohort", "plan_name_std")
    .agg(
        f.countDistinct("fs_acct_id").alias("count")
    )
    .orderBy("campaign_cohort",f.desc("count"))
)

# COMMAND ----------

(
    df_output_campaign
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "ml_campaign_xsell_bb_20240605")
    .mode("overwrite")
    .save()
)

# COMMAND ----------

# (
#     df_output_upload
#     .write
#     .format("snowflake")
#     .options(**options)
#     .option("dbtable", "ml_campaign_xsell_bb_20240529_leads")
#     .mode("overwrite")
#     .save()
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## s300 upload audience

# COMMAND ----------

# DBTITLE 1,upload audience
# MAGIC %fs ls /mnt/prod_sfmc/imports/DataAnalytics/xsell_bb_20240529.csv

# COMMAND ----------

# dbutils.fs.rm("/mnt/prod_sfmc/imports/DataAnalytics/xsell_bb_20240529.csv", True)

# COMMAND ----------

# # check dup
# display(
#     df_output_upload
#     .agg(
#         f.countDistinct("contact_key")
#         , f.countDistinct("billing_account_number")
#         , f.countDistinct("service_id")
#         , f.count('*')

#     )
# )

# display(df_output_upload.filter(f.col("contact_key").isNull()).count())
# display(df_output_upload.filter(f.col("billing_account_number").isNull()).count())

# COMMAND ----------

df_output_upload_eligible = (
    df_output_campaign
    .filter(f.col("cohort") == "TARGET")
    .filter(f.col("campaign_cohort") == "wallet_eligible-Y")
    .select(
        f.col("fs_cust_id").alias("contact_key")
        , f.col("fs_acct_id").alias("billing_account_number")
        ,f.coalesce(f.col("wallet_id"), f.col("fs_srvc_id")).alias("service_id")
        , "campaign_cohort"
    )
)
display(df_output_upload_eligible.limit(100))
display(df_output_upload_eligible.count())
display(df_output_upload_eligible.filter(f.col("service_id").isNull()).count())
display(df_output_upload_eligible.agg(f.count('*'), f.countDistinct("service_id"), f.countDistinct("billing_account_number")))

# COMMAND ----------

df_output_upload_noneligible = (
    df_output_campaign
    .filter(f.col("cohort") == "TARGET")
    .filter(f.col("campaign_cohort") == "wallet_eligible-N")
    .select(
        f.col("fs_cust_id").alias("contact_key")
        , f.col("fs_acct_id").alias("billing_account_number")
        ,f.col("fs_srvc_id").alias("service_id")
        , "campaign_cohort"
    )
)
display(df_output_upload_noneligible.limit(100))
display(df_output_upload_noneligible.count())
display(df_output_upload_noneligible.filter(f.col("service_id").isNull()).count())
display(df_output_upload_noneligible.agg(f.count('*'), f.countDistinct("service_id"), f.countDistinct("billing_account_number")))

# COMMAND ----------

# MAGIC %fs ls /mnt/prod_sfmc/imports/DataAnalytics/

# COMMAND ----------

# dir_path = "/mnt/feature-store-dev/dev_users/dev_rz/d999_testing/"
dir_path = "/mnt/prod_sfmc/imports/DataAnalytics/"
output_filename_y = "xsell_bb_20240605_wallet_eligible.csv"
file_dir_output_y = f"{dir_path}{output_filename_y}"

df_output_upload_eligible.toPandas().to_csv(f"/dbfs{file_dir_output_y}", header=True)


# COMMAND ----------

dir_path = "/mnt/prod_sfmc/imports/DataAnalytics/"
output_filename_n = "xsell_bb_20240605_wallet_ineligible.csv"
file_dir_output_n = f"{dir_path}{output_filename_n}"

df_output_upload_noneligible.toPandas().to_csv(f"/dbfs{file_dir_output_n}", header=True)

# COMMAND ----------

# dbutils.fs.rm("dbfs:/mnt/prod_sfmc/imports/DataAnalytics/xsell_bb_20240605_wallet_eligible.csv", True)

# COMMAND ----------

df_test_y = spark.read.format("csv").load(file_dir_output_y)
df_test_n = spark.read.format("csv").load(file_dir_output_n)

display(df_test_y.limit(10))
display(df_test_y.count())

display(df_test_n.limit(10))
display(df_test_n.count())

# COMMAND ----------

# MAGIC %fs ls /mnt/feature-store-dev/dev_users/dev_rz/d999_testing/

# COMMAND ----------

# # Source file path
# source_path = "dbfs:/FileStore/xsell_bb_20240529.csv"

# # Destination file path
# destination_path = "dbfs:/mnt/prod_sfmc/imports/DataAnalytics/xsell_bb_20240529.csv"

# # Copy the file
# dbutils.fs.cp(source_path, destination_path)

# COMMAND ----------

# MAGIC %fs ls /mnt/feature-store-prod-lab/d400_feature/d402_mobile_pp/fea_unit_base/

# COMMAND ----------

df_tmp = (
    spark.read.format("delta").load("/mnt/feature-store-prod-lab/d400_feature/d402_mobile_pp/fea_unit_base")
    .filter(f.col("reporting_date") == '2024-05-26')
    .select('fs_cust_id','fs_acct_id','fs_srvc_id','srvc_privacy_flag')
)

display(
    df_tmp
    # .slect('fs_cust_id','fs_acct_id','fs_srvc_id','srvc_privacy_flag')
    .limit(10)
    )

# COMMAND ----------

(
    df_tmp
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "pp_optin_20240607")
    .mode("overwrite")
    .save()
)
