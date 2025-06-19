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
vt_param_reporting_date = "2025-03-02"
vt_param_reporting_cycle_type = "rolling cycle"

# COMMAND ----------

# DBTITLE 1,feature store 01
df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))

# COMMAND ----------

display(
    df_fs_master
    .filter(f.col('fs_cust_id') == '1-169U613W')
    .filter(f.col('fs_acct_id') == '490640330')
    .filter(f.col('fs_srvc_id') == '64212484188')
)

# COMMAND ----------

display(
        df_fs_master
        .filter(f.col('network_dvc_model').isNull())
        .filter(f.col('reporting_date') >= '2024-03-02')
        .groupBy('network_dvc_model', 'reporting_date')
        .agg(f.countDistinct('fs_srvc_id'))
        )

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
df_campaign_hist_01 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/sc/ml_campaigns/wallet/250228-JS-MOBPM-iPhone-SE-Launch')

display(
    df_campaign_hist_01
    #.groupBy("campaign_name",'cohort')
    .agg(f.countDistinct("fs_srvc_id"))
)

display(df_campaign_hist_01.limit(100))

# COMMAND ----------

# df_campaign_hist_02 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_Email')

# display(
#     df_campaign_hist_02
#     #.groupBy("campaign_name",'cohort')
#     .agg(f.countDistinct("fs_srvc_id"))
# )

# display(df_campaign_hist_02.limit(100))

# COMMAND ----------

# df_campaign_hist_03 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/250123-JS-MOBPM-Samsung-NPI-Consumer_SMS')

# display(
#     df_campaign_hist_03
#     #.groupBy("campaign_name",'cohort')
#     .agg(f.countDistinct("fs_srvc_id"))
# )

# display(df_campaign_hist_03.limit(100))

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
# df_tmp_excl_02_02 = (
#     df_campaign_hist_02
#     .select("fs_srvc_id")
#     .distinct()
#     .withColumn(
#         "ch02_flag"
#         , f.lit('Y')
#     )
# )

# df_tmp_excl_02_03 = (
#     df_campaign_hist_03
#     .select("fs_srvc_id")
#     .distinct()
#     .withColumn(
#         "ch03_flag"
#         , f.lit('Y')
#     )
# )

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
        , "network_dvc_hardware_rating"
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
    # .join(
    #     df_tmp_excl_02_02
    #     , ["fs_srvc_id"]
    #     , "left"
    # )
    # .join(
    #     df_tmp_excl_02_03
    #     , ["fs_srvc_id"]
    #     , "left"
    # )
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
        , subset=['gc_curr_flag', 'ch01_flag',
                  #'ch02_flag', 'ch03_flag', 
                  'wpc_flag', 'wallet_eligibility_flag']
    )
    
    .withColumn(
        "target_segment", 
        f.when(
            f.col('srvc_privacy_flag') == 'N'
            , f.lit("z1.opt out")
        ) 
        .when(
            ~f.col('plan_name_std').isin('small mobile','unlimited mobile','endless plus','medium mobile')
            , f.lit("z6.plan_non_eligible")
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
                # | (f.col("ch02_flag") == 'Y')
                # | (f.col("ch03_flag") == 'Y')
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
        .when(
            f.col('network_dvc_os').isin('iOS')
            , f.lit('z7.iOS Device')
        )
        .otherwise(f.lit("a.target"))
        
    )
    .withColumn("churn_top_ntile", f.ntile(30).over(Window.orderBy(f.desc("churn_score"))))
    .withColumn("dr_top_ntile", f.ntile(30).over(Window.orderBy(f.desc("dr_score"))))
    .withColumn("ifp_top_ntile", f.ntile(30).over(Window.orderBy(f.desc("ifp_score"))))
)

# COMMAND ----------

# DBTITLE 1,check eligible plans
display(
    df_proc_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy('plan_name_std')
    .agg(f.count('*'))
)

# COMMAND ----------

display(
    df_proc_full
    .filter(f.col("target_segment") == 'a.target')
    .groupBy('')
    .agg(f.count('*'))
)

# COMMAND ----------

# DBTITLE 1,check wallet balance distributtion
display(
    df_proc_full
    .filter(f.col("target_segment") == 'a.target')
    .withColumn(
        'proensity_combine'
        , f.when(
            ~(
                (f.col('ifp_segment') == 'L')
                & (f.col('dr_segment') == 'L')
                ) , f.lit('H/M')
        )
        .otherwise('L')
    )
    .filter(f.col('proensity_combine') == 'H/M')
    .withColumn( 
        'wallence_bal_bucket', 
        f.floor(f.col('wallet_balance_earned') / 100)
    )
    .groupBy('wallence_bal_bucket',  (f.col('wallence_bal_bucket')*100 +100).alias('wallence_bal_category'), 'proensity_combine')
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.min('wallet_balance_earned')
        , f.max('wallet_balance_earned')
    ) 
)

# COMMAND ----------


display(
    df_proc_full
    .filter(f.col("target_segment") == 'a.target')
    .withColumn(
        'proensity_combine'
        , f.when(
            ~(
                (f.col('ifp_segment') == 'L')
                & (f.col('dr_segment') == 'L')
                ) , f.lit('H/M')
        )
        .otherwise('L')
    )
   # .filter(f.col('proensity_combine') == "L")
    .withColumn( 
        'wallence_bal_bucket', 
        f.floor(f.col('wallet_balance_earned') / 50)
    )
    .withColumn('wallence_bal_category', f.col('wallence_bal_bucket')*50 +50)
    .groupBy('wallence_bal_category', 'proensity_combine')
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.min('wallet_balance_earned')
        , f.max('wallet_balance_earned')
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
    .withColumn( 
        'wallence_bal_bucket', 
        f.floor(f.col('wallet_balance_earned') / 100)
    )
    .withColumn('wallence_bal_category', f.col('wallence_bal_bucket')*100 +100)
    .withColumn(
        'propensity_combine'
        , f.when(
            ~(
                (f.col('ifp_segment') == 'L')
                & (f.col('dr_segment') == 'L')
                ) , f.lit('H/M')
        )
        .otherwise('L')
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
    .groupBy("propensity_combine")
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
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsung_cofound') 
)

# COMMAND ----------

df_campaign_full = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsung_cofound")

# COMMAND ----------

display(
    df_campaign_full
    .limit(100)
)

display(
    df_campaign_full
    .filter(f.col('propensity_combine') == 'H/M')
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col('wallence_bal_category') <=500)
    .groupBy(f.col('network_dvc_release_tenure_year'))
    #.count()
    # .groupBy('network_dvc_release_tenure_year_group')
    .agg(f.countDistinct('fs_srvc_id'))
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

 display(
     df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(
        f.col('propensity_combine') == 'H/M'
        ) 
    .filter(f.col('wallence_bal_category')<=500)
    .filter(f.col('network_dvc_tenure') >= 10)
    # .filter(f.col('network_dvc_first_used_from_release_year') >=1)
    .groupBy('wallence_bal_category')
    .agg(f.count('*'))
)

# COMMAND ----------

# DBTITLE 1,check s24fe target
# if device rating is low or mid tier, go for FE 

df_s24fe_target=(
    df_campaign_full
    .filter(f.col('target_segment') == 'a.target')
    .filter(f.col('propensity_combine') == 'H/M')
    .filter(~f.col('network_dvc_hardware_rating').isin('Premium', 'High Tier'))
    .filter(f.col('wallence_bal_category')<=200)
)

display(df_s24fe_target
        .groupBy('network_dvc_hardware_rating', 'wallence_bal_category')
        .agg(f.countDistinct('fs_srvc_id'))
        )

# COMMAND ----------

# DBTITLE 1,S24, wallet dollar<=500

# estimated conversion 1.5% 

vt_param_target_size = 47000
vt_param_control_size = 6500
ls_param_strata = [ "dr_top_ntile","ifp_top_ntile"]


df_sample_target = create_sample_target(
     df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(
        ~((f.col('ifp_segment') == 'L')
          & (f.col('dr_segment') == 'L')
        ) 
    )
    .filter(f.col('wallence_bal_category')<=500)
    .join(df_s24fe_target, ['fs_acct_id', 'fs_cust_id', 'fs_srvc_id'], 'anti')
    #.filter(f.col("network_dvc_brand_std") != 'apple')
    #.filter(f.col("wallet_eligibility_flag") == 'Y')
    , ls_param_strata
    )

#display(df_sample_target.limit(10))

df_campaign_cand_control = find_similar_sample(
    df_campaign_full
    .filter(f.col("target_segment") != 'a.target')
    .filter(f.col("network_dvc_brand_std") != 'apple')
    .filter(f.col("ch01_flag") != 'Y')
    #.filter(f.col('wallence_bal_category')<=500)
    .filter(
        ~((f.col('ifp_segment') == 'L')
          & (f.col('dr_segment') == 'L')
        ) 
    )
    .join(df_s24fe_target, ['fs_acct_id', 'fs_cust_id', 'fs_srvc_id'], 'anti')
    #.filter(f.col("ch02_flag") != 'Y')
    #.filter(f.col("ch03_flag") != 'Y')
    , size = vt_param_control_size
    , strata = ls_param_strata
    , df_target = df_sample_target
)

df_campaign_cand_target = find_similar_sample(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("network_dvc_brand_std") != 'apple')
    .filter(f.col('wallence_bal_category')<=500)
    .filter(
        ~((f.col('ifp_segment') == 'L')
          & (f.col('dr_segment') == 'L')
        ) 
    )
    .join(df_s24fe_target, ['fs_acct_id', 'fs_cust_id', 'fs_srvc_id'], 'anti')
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

# DBTITLE 1,check duplicate
display(df_campaign_cand_target
        .join(df_campaign_cand_control, ['fs_acct_id', 'fs_cust_id', 'fs_srvc_id'], 'inner')      
        )

# COMMAND ----------

display(
    df_campaign_cand_target
    # .withColumn(
    #     'device_release_grp'
    #     , f.when(
    #         f.col('network_dvc_first_used_from_release_year') >=1
    #         , f.lit('1+year_from_release')   
    #     )
    #     .otherwise('<1year_from_release')
    # )
    .groupBy('network_dvc_release_tenure_year_group')
    .agg(f.countDistinct('fs_srvc_id'))
    
)

# COMMAND ----------

# DBTITLE 1,check device release & rating
# check wallet credit and device relase and device rating 

display(
    df_campaign_cand_target
    .withColumn(
        'device_release_grp'
        , f.when(
            f.col('network_dvc_first_used_from_release_year') >=1
            , f.lit('1+year_from_release')   
        )
        .otherwise('<1year_from_release')
    )
    .groupBy('device_release_grp')
    .agg(f.countDistinct('fs_srvc_id'))
    
)


display(
    df_campaign_cand_control
    .withColumn(
        'device_release_grp'
        , f.when(
            f.col('network_dvc_first_used_from_release_year') >=1
            , f.lit('1+year_from_release')   
        )
        .otherwise('<1year_from_release')
    )
    .groupBy('device_release_grp')
    .agg(f.countDistinct('fs_srvc_id'))
    
)

display(
    df_campaign_cand_target
    .groupBy('wallence_bal_category')
    .agg(f.countDistinct('fs_srvc_id'))
)


display(
    df_campaign_cand_target
    .groupBy('wallence_bal_category', 'network_dvc_hardware_rating')
    .agg(f.countDistinct('fs_srvc_id'))
)

display(
    df_campaign_cand_control
    .groupBy('wallence_bal_category', 'network_dvc_hardware_rating')
    .agg(f.countDistinct('fs_srvc_id'))
)



# COMMAND ----------

# MAGIC %md ### s210 campaign target finalisation s24

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
        'campaign'
        , f.lit('SamsungS24')
    )
    .withColumn('campaign_cohort', f.col('wallence_bal_category'))
    .withColumn(
        'offer'
        , f.when(
            (
                (f.col('wallence_bal_category') == 100)
                &   (f.col('campaign') == 'SamsungS24')
            )
            , f.lit('500')
        )
        .when(
            (
                (f.col('wallence_bal_category') == 200)
                & (f.col('campaign') == 'SamsungS24')
            )
            , f.lit('400')
        )
        .when(
                (f.col('wallence_bal_category') == 300)
                & (f.col('campaign') == 'SamsungS24')
            , f.lit('300')
        )
        .when(
            (
                (f.col('wallence_bal_category') == 400)
                & (f.col('campaign') == 'SamsungS24')
            )
            , f.lit('200')
        )
        .when(
            (
                (f.col('wallence_bal_category') == 500)
                & (f.col('campaign') == 'SamsungS24')
            )
            , f.lit('100')
        )
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
        , f.lit('SamsungS24') 
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


display(
    df_output_campaign
    .groupBy("target_cohort", "offer")
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
    #.orderBy("campaign", f.desc("offer"))
)


# COMMAND ----------

# MAGIC %md #### combine all

# COMMAND ----------

# DBTITLE 1,check propensity
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
    .orderBy("campaign")
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
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24') 
)

# COMMAND ----------

df_s24_campaign = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24')

df_s24_target = (
    df_s24_campaign
    .filter(f.col('target_cohort')=='Target')
)

df_s24_control = (
    df_s24_campaign
    .filter(f.col('target_cohort')!='Target')
)

# COMMAND ----------

display(df_s24_control.limit(10))
display(df_s24_target.limit(10))

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

# MAGIC %md
# MAGIC ### s220  Campaign target for S24FE

# COMMAND ----------

display(df_s24fe_target.limit(10))


df_s24fe_target=(
    df_campaign_full
    .filter(f.col('target_segment') == 'a.target')
    .filter(f.col('propensity_combine') == 'H/M')
    .filter(~f.col('network_dvc_hardware_rating').isin('Premium', 'High Tier'))
    .filter(f.col('wallence_bal_category')<=200)
)

# COMMAND ----------

# DBTITLE 1,S24FE Target

vt_param_target_size = 33000
vt_param_control_size = 6500
ls_param_strata = ["dr_top_ntile", "ifp_top_ntile"]


df_sample_target = create_sample_target(
     df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col('propensity_combine') == 'H/M')
    .filter(~f.col('network_dvc_hardware_rating').isin('Premium', 'High Tier'))
    .filter(f.col('wallence_bal_category')<=200)
    #.filter(f.col("network_dvc_brand_std") != 'apple')
    #.filter(f.col("wallet_eligibility_flag") == 'Y')
    , ls_param_strata
)

#display(df_sample_target.limit(10))

df_campaign_cand_control = find_similar_sample(
    df_campaign_full
    .filter(f.col("target_segment") != 'a.target')
    .filter(f.col("ch01_flag") != 'Y')
    .filter(f.col('propensity_combine') == 'H/M')
    .filter(f.col("network_dvc_brand_std") != 'apple')
    #.filter(~f.col('network_dvc_hardware_rating').isin('Premium', 'High Tier'))
    .filter(f.col('wallence_bal_category')<=200)
    #.join(df_s24_control, ['fs_acct_id', 'fs_srvc_id', 'fs_cust_id'], 'anti')
    #.filter(f.col("ch02_flag") != 'Y')
    #.filter(f.col("ch03_flag") != 'Y')
    , size = vt_param_control_size
    , strata = ls_param_strata
    , df_target = df_sample_target
)

df_campaign_cand_target = find_similar_sample(
    df_campaign_full
    .filter(f.col("target_segment") == 'a.target')
    .filter(f.col("network_dvc_brand_std") != 'apple')
    .filter(f.col('propensity_combine') == 'H/M')
    .filter(~f.col('network_dvc_hardware_rating').isin('Premium', 'High Tier'))
    .filter(f.col('wallence_bal_category')<=200)
    .join(df_s24_target, ['fs_acct_id', 'fs_srvc_id', 'fs_cust_id'], 'anti')
    #.filter(f.col("wallet_eligibility_flag") == 'Y')
    , size = vt_param_target_size
    , strata = ls_param_strata
    , df_target = df_sample_target
)

# export 
(
    df_campaign_cand_target
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24fe_target') 
)

(
    df_campaign_cand_control
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24fe_control') 
)

# import 
df_campaign_cand_target = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24fe_target')
df_campaign_cand_control = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24fe_control')


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

# DBTITLE 1,check duplicate
display(
    df_campaign_cand_control
    .join(df_campaign_cand_target, ['fs_acct_id', 'fs_srvc_id', 'fs_cust_id'], 'inner')
)

# COMMAND ----------

display(df_campaign_cand_control.count())
display(df_campaign_cand_target.count())

# COMMAND ----------


df_output_campaign_01 = get_campaign_attr(
    df_campaign_cand_target
    # .withColumn(
    #     "offer"
    #     , f.lit(700)
    # )
    .withColumn(
        "campaign",
        f.lit("SamsungS24FE")
    )
    .withColumn('campaign_cohort', f.col('wallence_bal_category'))
    .withColumn(
        'offer'
        , f.when(
            (
                (f.col('wallence_bal_category') == 100)
                &   (f.col('campaign') == 'SamsungS24FE')
            )
            , f.lit('200')
        )
        .when(
            (
                (f.col('wallence_bal_category') == 200)
                & (f.col('campaign') == 'SamsungS24FE')
            )
            , f.lit('100')
        )
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
        , f.lit("SamsungS24FE")
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

# DBTITLE 1,check propensity
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
    .orderBy("campaign")
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

# DBTITLE 1,export lake file s24fe
(
    df_output_campaign
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    #.option("partitionOverwriteMode", "dynamic")
    .save('/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24fe') 
)

# COMMAND ----------

# MAGIC %md ## s300 SFMC upload

# COMMAND ----------

df_output_campaign_24fe = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24fe")
df_output_campaign_24 = spark.read.format("delta").load("/mnt/ml-lab/dev_users/dev_sc/ml_campaigns/wallet/20250305_samsungs24")

# COMMAND ----------

# DBTITLE 1,check duplicate
display(df_output_campaign_24fe.alias('a')
        .filter(f.col("target_cohort") != "Local Control")
        .join(df_output_campaign_24.alias('b')
              .filter(f.col("target_cohort") != "Local Control")
              , ['fs_acct_id', 'fs_srvc_id', 'fs_cust_id'], 'inner')
        .select('a.ifp_segment'
                , 'a.dr_segment'
                , 'b.ifp_segment'
                , 'b.dr_segment'
                )
        )

# COMMAND ----------

df_output_campaign = (
    df_output_campaign_24
    .union(df_output_campaign_24fe)
)

# COMMAND ----------

display(
    df_output_campaign
    .filter(f.col('target_cohort') != "Local Control")
    .groupBy('target_cohort', 'campaign', 'offer')
    .agg(
        f.count('*')
        , f.max('wallet_balance_earned') 
        , f.min('wallet_balance_earned')
        )    

)

# COMMAND ----------

# DBTITLE 1,check propensity again
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
    .orderBy("campaign")
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

display(df_output_campaign_24fe.limit(10))

display(df_output_campaign_24.limit(10))

# COMMAND ----------

df_sfmc_export = (
    df_output_campaign
    .filter(f.col("target_cohort") != "Local Control")
    # .withColumn(
    #     "wallet_balance_final"
    #     , f.col("wallet_balance_earned") + f.col("offer")
    # )
    #.withColumn('offer', f.col('offer'))
    # .withColumn(
    #     'experiment_segment_version'
    #     , f.when(
    #         (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 500)
    #         , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$500')
    #     )
    #     .when(
    #         (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 450)
    #         , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$450')
    #     )  
    #     .when(
    #         (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 300)
    #         , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$300')
    #     ) 
    #     .when(
    #         (f.col('campaign') == 'SamsungS24FE') & (f.col('offer') == 200)
    #         , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24FE_$200')
    #     )
    #     .when(
    #         (f.col('campaign') == 'SamsungS24FE') & (f.col('offer') == 150)
    #         , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24FE_$150')
    #     ) 
    #     .when(
    #         (f.col('campaign') == 'SamsungS24FE') & (f.col('offer') == 100)
    #         , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24FE_$100')
    #     )             
    # )
    .select(
        f.col("fs_cust_id").alias("contact_key")
        , f.col("fs_acct_id").alias("billing_account_number")
        , f.col('fs_srvc_id').alias("service_id")
        , f.col("campaign").alias('device')
        , f.col('offer')
        #, f.col('experiment_segment_version')
        , f.current_date().alias("data_update_date")
    )
)

display(
    df_sfmc_export
    .groupBy("device", "offer")
    .agg(
        f.count("*")
        , f.countDistinct("service_id")
    )
   # .orderBy("campaign")
)

display(df_sfmc_export.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### export sfmc

# COMMAND ----------

# DBTITLE 1,export sfmc
dir_data_sfmc = "/mnt/prod_sfmc/imports/DataAnalytics/"

(
    df_sfmc_export
    .toPandas()
    .to_csv(f"/dbfs{dir_data_sfmc}/250306-RM-MOBPM-Wallet-Samsung-Galaxy.csv", index=False)
)

# COMMAND ----------

f"/dbfs{dir_data_sfmc}/250306-RM-MOBPM-Wallet-Samsung-Galaxy.csv"

# COMMAND ----------

df_check = spark.read.options(header=True).csv(f"{dir_data_sfmc}/250306-RM-MOBPM-Wallet-Samsung-Galaxy.csv")

# COMMAND ----------

display(
    df_check
    .groupBy('device', 'offer')
    .agg(f.count('*')
         , f.countDistinct(f.concat(f.col('contact_key'), f.col('billing_account_number'), f.col('service_id') ))
         , f.countDistinct('service_id')
         )    
)

# COMMAND ----------

display(df_check.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### export to snowflake

# COMMAND ----------

df_snowflake_export = (
    df_output_campaign
    #.filter(f.col("target_cohort") != "Local Control")
    # .withColumn(
    #     "wallet_balance_final"
    #     , f.col("wallet_balance_earned") + f.col("offer")
    # )
    #.withColumn('offer', f.col('offer'))
    .withColumn(
        'experiment_segment_version'
        , f.when(
            (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 500)
            , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$500')
        )
        .when(
            (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 400)
            , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$400')
        )  
        .when(
            (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 300)
            , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$300')
        ) 
        .when(
            (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 200)
            , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$200')
        )
        .when(
            (f.col('campaign') == 'SamsungS24') & (f.col('offer') == 100)
            , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24_$100')
        ) 
        .when(
            (f.col('campaign') == 'SamsungS24FE') & (f.col('offer') == 200)
            , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24FE_$200')
        )
        .when(
            (f.col('campaign') == 'SamsungS24FE') & (f.col('offer') == 100)
            , f.lit('250305-RM-MOBPM-Wallet-Samsung-Galaxy-S24FE_$100')
        )               
    )
    
    # .select(
    #     f.col("fs_cust_id").alias("contact_key")
    #     , f.col("fs_acct_id").alias("billing_account_number")
    #     , f.col('fs_srvc_id').alias("service_id")
    #     , f.col("campaign").alias('device')
    #     , f.col('offer')
    #     , f.col('experiment_segment_version')
    #     , f.current_date().alias("data_update_date")
    # )
)

# COMMAND ----------

display(df_snowflake_export
        .filter(f.col('target_cohort')=='Target')
        .count())

display(df_snowflake_export.limit(10))

# COMMAND ----------

(
    df_snowflake_export
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "ml_campaign_wallet_exp_20250305_rm_mobpm_samsung_galaxy")
    .mode("overwrite")
    .save()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check Wash Records

# COMMAND ----------

dir_data_sfmc = "/mnt/prod_sfmc/imports/DataAnalytics/"
df_check_1 = spark.read.options(header=True).csv(f"{dir_data_sfmc}/250305-RM-MOBPM-Wallet-Samsung-Galaxy-SEND-S24-WASH.csv")
df_check_2 = spark.read.options(header = True).csv(f"{dir_data_sfmc}/250305-RM-MOBPM-Wallet-Samsung-Galaxy-SEND-S24-WASHV2.csv")
df_check_3 = spark.read.options(header = True).csv(f"{dir_data_sfmc}/250305-RM-MOBPM-Wallet-Samsung-Galaxy-SEND-S24FE-WASH.csv")
#250305-RM-MOBPM-Wallet-Samsung-Galaxy-SEND-S24FE-WASH

# COMMAND ----------

dir_data_sfmc = "/mnt/prod_sfmc/imports/DataAnalytics/"

# COMMAND ----------

 print(f"{dir_data_sfmc}/250305-RM-MOBPM-Wallet-Samsung-Galaxy-SEND-S24-WASHV2.csv")
 print(f"{dir_data_sfmc}/250305-RM-MOBPM-Wallet-Samsung-Galaxy-SEND-S24FE-WASH.csv")

# COMMAND ----------

df_output_campaign = (
    df_output_campaign_24
    .union(df_output_campaign_24fe)
)

# COMMAND ----------

df_wash_target_2 = (
        df_output_campaign
        .filter(f.col('target_cohort') == 'Target')
        .filter(f.col('campaign') == 'SamsungS24FE')
        .join(df_check_3
              , (f.col('Contact_Key') == f.col('fs_cust_id'))
                & (f.col('fs_acct_id') == f.col('Billing_Account_Number'))
                & (f.col('fs_srvc_id') == f.col('Service_ID'))
              , 'inner' )
)

df_wash_target_1 = (
        df_output_campaign
        .filter(f.col('target_cohort') == 'Target')
        .filter(f.col('campaign') == 'SamsungS24')
        .join(df_check_1
              , (f.col('Contact_Key') == f.col('fs_cust_id'))
                & (f.col('fs_acct_id') == f.col('Billing_Account_Number'))
                & (f.col('fs_srvc_id') == f.col('Service_ID'))
              , 'inner' )
)

# COMMAND ----------

print('target')
display(
    df_wash_target_2
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

print('control')
display(
    df_output_campaign
    .filter(f.col('campaign') == 'SamsungS24FE')
    .filter(f.col('target_cohort') == 'Local Control')
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

display(
    df_output_campaign
    .filter(f.col('campaign') == 'SamsungS24')
    .filter(f.col('target_cohort') == 'Local Control')
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

display(df_check.count())

display(df_check.limit(10))
