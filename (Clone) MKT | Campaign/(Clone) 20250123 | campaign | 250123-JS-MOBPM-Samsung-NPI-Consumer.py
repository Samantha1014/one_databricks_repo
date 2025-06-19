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
# MAGIC %run "./utils_spark_df"

# COMMAND ----------

# DBTITLE 1,utility functions 02
# MAGIC %run "./utils_stratified_sampling"

# COMMAND ----------

# MAGIC %run "./sc_utils_stratified_sampling"

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
# vt_param_reporting_date = "2024-09-29"
vt_param_reporting_date = "2025-01-19"
vt_param_reporting_cycle_type = "rolling cycle"

# COMMAND ----------

# DBTITLE 1,feature store 01
df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))
df_mvnt_ifp_srvc = spark.read.format('delta').load(os.path.join(dir_fs_data_target, "d501_mobile_oa_consumer/mvmt_ifp_upsell_on_service"))
df_mvnt_ifp_bill = spark.read.format('delta').load(os.path.join(dir_fs_data_target, "d501_mobile_oa_consumer/mvmt_ifp_upsell_on_bill"))

# COMMAND ----------

df_output_campaign_v1 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_Email')
df_output_campaign_v2 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_SMS')

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

# DBTITLE 1,sfmc query for leads
query_sfmc = """ 
select  * from PROD_MAR_TECH.SERVING.SFMC_EMAIL_PERFORMANCE
where campaignname in ('250123-JS-MOBPM-EML-M-DEV-P4-Samsung-NPI-Consumer_Email')
"""


df_sfmc = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_sfmc
    )
    .load()
)


query_sfmc_sms = """
select * from PROD_MAR_TECH.SERVING.SFMC_ON_NET_SMS_MESSAGE
where sms_name in ('250123-JS-MOBPM-SMS-M-DEV-P4-Samsung-NPI-Consumer');
"""

df_sfmc_sms = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_sfmc_sms
    )
    .load()
)


# COMMAND ----------

df_sfmc_export_v1 = (
    df_output_campaign_v1
    .filter(f.col("target_cohort") != "Control")
    .select(
        f.col("fs_cust_id").alias("contact_key")
        , f.col("fs_acct_id").alias("billing_account_number")
        , f.col('fs_srvc_id').alias("service_id")
        , f.col("campaign").alias('target_group')
        , f.current_date().alias("data_update_date")
    )
)

display(
    df_sfmc.alias('a')
        .join(
            df_sfmc_export_v1.alias('b')
            , (f.col('a.CUSTOMER_ID') == f.col('b.contact_key')) 
              #(f.col('a.BILLING_ACCOUNT_NUMBER') == f.col('b.billing_account_number'))
            , 'anti'
        )
)

# COMMAND ----------

display(df_sfmc_export_v1
        .select('contact_key')
        .distinct()
        .count()
        )

# COMMAND ----------

display(df_sfmc_email.count())

display(df_sfmc_sms.count())

# COMMAND ----------

display(
        df_sfmc_export_v1
        .count()
)

# COMMAND ----------

display(df_sfmc_export_v1
        .join(df_sfmc_email, f.col('contact_key') == f.col('CONTACTKEY'), 'anti')
        .limit(10)
)



#display(df_output_campaign_v2.count())

# COMMAND ----------

display(df_sfmc_email.count())

# COMMAND ----------

# DBTITLE 1,one upgrade data
query_one_upgrade = """select * from PROD_MAR_TECH.SERVING.export_bill_ifp where IFP_PHONE_UPGRADE_FLAG = 'Y';"""

df_one_upgrade = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_one_upgrade
    )
    .load()
)



# COMMAND ----------

# DBTITLE 1,check samsung/ android count
# samsung/android customers check 

display(
    df_fs_master
    .filter(f.col('reporting_date') == vt_param_reporting_date)
    #.limit(100)
    .groupBy('network_dvc_model_brand_std', 'network_dvc_os')
    .agg(
        f.countDistinct('fs_acct_id')
         , f.countDistinct('fs_srvc_id')
    )        
)

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

# DBTITLE 1,wallet program control
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
    .join(
        df_fs_id_master
        , ["reporting_date", "reporting_cycle_type", "fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    # add condition for all andrioid customers 
    # .filter(f.col('network_dvc_os').isin('Android')) 
)

display(
    df_base_full
    .limit(100)
)

display(
    df_base_full
    .count()
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

# df_tmp_excl_03 = (
#     df_wallet_program_control
#     .select("fs_srvc_id_sha2")
#     .distinct()
#     .withColumn(
#         "wpc_flag"
#         , f.lit('Y')
#     )
# )

# exclude one one grade 
df_tmp_excl_04 = (
    df_one_upgrade
    .select('IFP_LINKED_SERVICE_ID')
    .distinct()
    .withColumn(
        'one_upgrade_flag'
        , f.lit('Y')
    )
    .withColumnRenamed('IFP_LINKED_SERVICE_ID', 'fs_srvc_id')
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
        df_tmp_excl_04
        , ["fs_srvc_id"]
        , "left"
    )
    .fillna(
        value='N'
        , subset=['gc_curr_flag', 'one_upgrade_flag']
    )
    .withColumn(
        "target_segment"
        ,
        f.when(
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
                (f.col("one_upgrade_flag") == 'Y')
            )
            , f.lit("z4.one_upgrade")
        )
        .when(
            (
                f.col("risk_top_ntile") >= 98
            )
            , f.lit("z6.High Bad Debt Risk")
        )
       .when(
           (
               f.col('network_dvc_brand_std').isin('apple')
           )
           , f.lit('z7.Apple Device')
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
    .groupBy('srvc_privacy_flag')
    .agg(f.countDistinct('fs_srvc_id'))
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
display(df_proc_full.count())

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
    # base on requirement of 1. samsumg/android phone 2. not in one upgrade 
    #.filter(f.col("one_upgrade_flag") != "Y")
    #.filter(f.col('network_dvc_os') == 'Android')
)


# COMMAND ----------

display(df_campaign_full.count())

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
# (
#     df_campaign_full
#     .write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     #.option("partitionOverwriteMode", "dynamic")
#     .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/samsung20240121') 
# )

# COMMAND ----------

# df_campaign_full = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/samsung20240121')

# COMMAND ----------

# MAGIC %md ### s206 local control - H

# COMMAND ----------

display(
    df_campaign_full
    .groupBy("srvc_privacy_flag")
    .agg(f.countDistinct("fs_srvc_id"))
)

# COMMAND ----------

# DBTITLE 1,check ifp uptake count
display(
    df_mvnt_ifp_bill
    .filter(f.col('ifp_type') == 'device')
    #.select('reporting_date', 'reporting_cycle_type', 'fs_ifp_id')
    .filter(f.col('reporting_date') >= '2023-01-01')
    .filter(f.col('reporting_cycle_type') == 'rolling cycle')
    .groupBy('reporting_date', 'reporting_cycle_type')
    .agg(f.countDistinct('fs_ifp_id'))
)


display(
    df_mvnt_ifp_srvc
    .filter(f.col('reporting_date') >= '2023-01-01')
    .filter(f.col('reporting_cycle_type') == 'rolling cycle')
    .groupBy('reporting_date', 'reporting_cycle_type')
    .agg(f.countDistinct('fs_ifp_id'))
)


display(
    df_fs_id_master
    .filter(f.col('reporting_date') >= '2023-01-01')
    .filter(f.col('reporting_cycle_type') == 'rolling cycle')
    .groupBy('reporting_date')
    .agg(f.countDistinct('fs_srvc_id'))
)

# COMMAND ----------

# DBTITLE 1,control and target
# 0.8% conversion, take 150 as significance size, 150/0.8% = 18750 for control 
vt_param_control_size = 25000
# vt_param_priority_field = "wallet_control_flag"
# ls_param_priority_values = ['Y']

ls_param_strata = ["churn_top_ntile", "ifp_top_ntile"]


df_control = generate_sample_v2(
    df= df_campaign_full.filter(f.col("target_segment") == 'a.target')
    , size = vt_param_control_size
    , strata =  ls_param_strata
    , seed = 124
    
)

# export 
(
    df_control
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/samsung20240121_df_control') 
)

# re-import 
df_control = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/samsung20240121_df_control')

# calculate target 
df_target = (
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .join(
        df_control
        .select("fs_cust_id", "fs_acct_id", "fs_srvc_id")
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left_anti"
    )
)


print("control")
display(
    df_control
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
    df_target
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

# DBTITLE 1,check
# no dup between target and control 
display(
    df_control
    .join(df_target, ["fs_cust_id", "fs_acct_id", "fs_srvc_id"], "inner")
)

# COMMAND ----------

# DBTITLE 1,evaluate sample
evaluate_sample(
    df_control
    , df_target
    , ["churn_score", "dr_score", "ifp_score"]
)

# COMMAND ----------

# MAGIC %md ### s210 campaign target finalisation

# COMMAND ----------

# DBTITLE 1,function
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
           # , "wallet_eligibility_flag"
           # , "wallet_balance_earned"
           # , "wallet_balance_date_key"
            , "campaign"
           # , "offer"
          #  , "campaign_cohort"
            , "target_cohort"
        )
    )

    return df_out

# COMMAND ----------

# DBTITLE 1,campaign version 1

#campagin verison 1 

# 1.	Samsung NPI pre-order email: 
# a.	All Samsung/Android customers (H/M/L)
# b.	Exclude all One Upgrade customers 
# c.	Create relevant control group for the campaign
# d.	There is only one eDM version for this campaign 


df_output_campaign_01 = get_campaign_attr(
    df_target
    .withColumn(
        "campaign",
        f.lit("250123-JS-MOBPM-Samsung-NPI-Consumer_Email")
    )
    .withColumn(
        "target_cohort",
        f.lit("Target")
    )
)


df_output_campaign_02 = get_campaign_attr(
    df_control
    .withColumn(
        "campaign"
        , f.lit("250123-JS-MOBPM-Samsung-NPI-Consumer_Email")
    )
    .withColumn(
        "target_cohort"
        , f.lit("Control")
    )
)

df_output_campaign_v1 = (
    df_output_campaign_01
    .unionByName(df_output_campaign_02)
)

display(
    df_output_campaign_v1
    .groupBy("campaign", "target_cohort")
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
   # .orderBy("campaign", f.desc("offer"))
)


# COMMAND ----------

# DBTITLE 1,campaign version 2
# campaign verison 2 
# a.	All Samsung/Android One Upgrade customers 
# b.	No control needed 
# c.	This group will receive an SMS to let them know that they can’t redeem One Upgrade during pre-order

df_output_campaign_v2 = (
    df_fs_master
    .filter(f.col("reporting_date") == vt_param_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_reporting_cycle_type)
    .join(
        df_fs_id_master
        , ["reporting_date", "reporting_cycle_type", "fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    # add condition for all andrioid customers 
    .filter(f.col('network_dvc_os').isin('Android')) 
    # make sure service privacy is not N 
    #.filter(f.col('srvc_privacy_flag') != 'N')
    # with one upgrade customers
    .join(
        df_one_upgrade
        , f.col('fs_srvc_id') == f.col('IFP_LINKED_SERVICE_ID')
        , "inner"  
    )
    .withColumn(
        "campaign",
        f.lit("250123-JS-MOBPM-Samsung-NPI-Consumer_SMS")
    )
)

display(df_output_campaign_v2.count())

display(df_output_campaign_v2.limit(100))

# COMMAND ----------

# MAGIC %md #### combine all

# COMMAND ----------

df_output_campaign_v1 = (
    df_output_campaign_v1
)

display(df_output_campaign_v1.limit(10))

df_output_campaign_v2  = (
    df_output_campaign_v2
)


display(df_output_campaign_v1.limit(10))
display(df_output_campaign_v2.count())

# COMMAND ----------

# DBTITLE 1,export lake file
(
    df_output_campaign_v1
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_Email') 
)



(
    df_output_campaign_v2
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_SMS') 
)



# COMMAND ----------

# DBTITLE 1,export sf
# (
#     df_output_campaign
#     .write
#     .format("snowflake")
#     .options(**options)
#     .option("dbtable", "xxml_campaign_wallet_exp_202410_rm_mobpm_samsung_lto_offer")
#     .mode("overwrite")
#     .save()
# )

# COMMAND ----------

# MAGIC %md ## s300 SFMC upload

# COMMAND ----------

df_output_campaign_v1 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_Email')
df_output_campaign_v2 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_SMS')

# COMMAND ----------

# DBTITLE 1,sfmc export 1
df_sfmc_export_v1 = (
    df_output_campaign_v1
    .filter(f.col("target_cohort") != "Control")
    .select(
        f.col("fs_cust_id").alias("contact_key")
        , f.col("fs_acct_id").alias("billing_account_number")
        , f.col('fs_srvc_id').alias("service_id")
        , f.col("campaign").alias('target_group')
        , f.current_date().alias("data_update_date")
    )
)

display(
    df_sfmc_export_v1
    .groupBy("target_group")
    .agg(
        f.count("*")
        , f.countDistinct("service_id")
    )
    .orderBy("target_group")
)


display(df_sfmc_export_v1.limit(10))

# COMMAND ----------

# DBTITLE 1,sfmc export 2
df_sfmc_export_v2 = (
    df_output_campaign_v2
    .withColumn('target_group', f.lit('250123-JS-MOBPM-Samsung-NPI-Consumer_SMS'))
    #.filter(f.col("target_cohort") != "Local Control")
    # .withColumn(
    #     "wallet_balance_final"
    #     , f.col("wallet_balance_earned") + f.col("offer")
    # )
    .select(
        f.col("fs_cust_id").alias("contact_key")
        , f.col("fs_acct_id").alias("billing_account_number")
        , f.col('fs_srvc_id').alias("service_id")
        , f.col("target_group")
        , f.current_date().alias("data_update_date")
    )
    .distinct()
)

display(df_sfmc_export_v2.limit(10))
display(df_sfmc_export_v2.count())

# COMMAND ----------

display(
        df_sfmc_export_v1
        .union(df_sfmc_export_v2)
        .groupBy('target_group')
        .agg(f.countDistinct('service_id'))
)

# COMMAND ----------

# DBTITLE 1,export sfmc
dir_data_sfmc = "/mnt/prod_sfmc/imports/DataAnalytics/"

(
    df_sfmc_export_v1
    .toPandas()
    .to_csv(f"/dbfs{dir_data_sfmc}/250123-JS-MOBPM-Samsung-NPI-Consumer_Email_v2.csv", index=False)
)

# COMMAND ----------

(
    df_sfmc_export_v2
    .toPandas()
    .to_csv(f"/dbfs{dir_data_sfmc}/250123-JS-MOBPM-Samsung-NPI-Consumer_SMS_v2.csv", index=False)
)

# COMMAND ----------

f"/dbfs{dir_data_sfmc}/250123-JS-MOBPM-Samsung-NPI-Consumer_Email.csv"

# COMMAND ----------

df_check_v1 = spark.read.options(header=True).csv(f"{dir_data_sfmc}/250123-JS-MOBPM-Samsung-NPI-Consumer_Email_v2.csv")

df_check_v2 = spark.read.options(header=True).csv(f"{dir_data_sfmc}/250123-JS-MOBPM-Samsung-NPI-Consumer_SMS_v2.csv")

# COMMAND ----------

print('v2')
display(df_check_v2.limit(100))
display(df_check_v2.count())

print('v1')
display(df_check_v1.limit(100))
display(df_check_v1.count())

# COMMAND ----------

display(
        df_sfmc_export_v1
        .withColumn('sfmc_file_name', f.lit('250123-JS-MOBPM-Samsung-NPI-Consumer_Email_v2.csv'))
        .union(
            df_sfmc_export_v2
            .withColumn('sfmc_file_name', f.lit('250123-JS-MOBPM-Samsung-NPI-Consumer_SMS_v2.csv'))
               )
        .groupBy('target_group', 'sfmc_file_name')
        .agg(f.countDistinct('service_id')
             , f.count('*')
             )
)

# COMMAND ----------

display(df_sfmc.limit(10))

display(df_sfmc_export_v1.limit(10))

# COMMAND ----------

display(
    df_sfmc.alias('a')
        .join(
            df_sfmc_export_v1.alias('b')
            , (f.col('a.CUSTOMER_ID') == f.col('b.contact_key')) 
              #(f.col('a.BILLING_ACCOUNT_NUMBER') == f.col('b.billing_account_number'))
            , 'anti'
        )
        )

# COMMAND ----------

display(
    df_sfmc
    .count()        
)

# COMMAND ----------

display(
    df_sfmc_export_v1
    .filter(f.col('contact_key') == '1-10AW1A0H')
    )

# COMMAND ----------

display(
    df_output_campaign_v1.alias('a')
    .filter(f.col('target_cohort') == 'Control')
    .join(
            df_sfmc.alias('b')
            , f.col('b.CUSTOMER_ID') == f.col('a.fs_cust_id')
            , 'inner'
          )
    .count()
)
