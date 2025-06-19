# Databricks notebook source
# MAGIC %md
# MAGIC ## s000 environment setup

# COMMAND ----------

# DBTITLE 1,libraries
import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f, Window
from pyspark.sql import Window

# COMMAND ----------

#dbutils.widgets.text("param_sf_host", "", "Snowflake Host")
#dbutils.widgets.text("param_sf_sp", "", "Snowflake Service Principal")
#dbutils.widgets.text("param_keyvault_scope", "", "Key Vault Scope")
#dbutils.widgets.text("param_keyvault_key", "", "Key Vault Key")
#dbutils.widgets.text("param_sf_wh", "", "Snowflake Warehouse")
#dbutils.widgets.text("param_sf_db", "", "Snowflake Database")
#dbutils.widgets.text("param_sf_schema", "", "Snowflake Schema")
#dbutils.widgets.text("param_date", "", "Run Date")
#dbutils.widgets.text("param_control_group_id", "", "Control Group ID")

# COMMAND ----------

# DBTITLE 1,parameters 01
vt_param_sf_host   = dbutils.widgets.get("param_sf_host")
vt_param_sf_sp     = dbutils.widgets.get("param_sf_sp")
vt_param_kv_scope  = dbutils.widgets.get("param_keyvault_scope")
vt_param_kv_key    = dbutils.widgets.get("param_keyvault_key")
vt_param_sf_wh     = dbutils.widgets.get("param_sf_wh")
vt_param_sf_db     = dbutils.widgets.get("param_sf_db")
vt_param_sf_schema = dbutils.widgets.get("param_sf_schema")

# COMMAND ----------

# DBTITLE 1,parameters 02
#vt_param_date = "2024-10-02"
#vt_param_date_key = "20241002"
#vt_param_control_group_id = "global_control_moac_00"
vt_param_date             = dbutils.widgets.get("param_date")
vt_param_date_key         = vt_param_date.replace("-", "")
vt_param_control_group_id = dbutils.widgets.get("param_control_group_id")

# COMMAND ----------

# DBTITLE 1,utility functions 01
# MAGIC %run "../utility_functions/utils_stratified_sampling"

# COMMAND ----------

# DBTITLE 1,utility functions 02
# MAGIC %run "../utility_functions/utils_spark_df"

# COMMAND ----------

# DBTITLE 1,db connection
# ------------- Use snowflake utility
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils

# ------------ login to snowflake
password = dbutils.secrets.get(scope = vt_param_kv_scope, key = vt_param_kv_key)

options = {
  "sfUrl": vt_param_sf_host, 
  "sfUser": vt_param_sf_sp,
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": vt_param_sf_db,
  "sfSchema": vt_param_sf_schema,
  "sfWarehouse": vt_param_sf_wh
}

# COMMAND ----------

# DBTITLE 1,directories 01
dir_fs_data_parent  = "/mnt/feature-store-prod-lab"
dir_mls_data_parent = "/mnt/ml-store-prod-lab/classification"

# COMMAND ----------

# DBTITLE 1,directories 02
dir_fs_data_meta   = os.path.join(dir_fs_data_parent, 'd000_meta')
dir_fs_data_raw    = os.path.join(dir_fs_data_parent, 'd100_raw')
dir_fs_data_int    = os.path.join(dir_fs_data_parent, "d200_intermediate")
dir_fs_data_prm    = os.path.join(dir_fs_data_parent, "d300_primary")
dir_fs_data_fea    = os.path.join(dir_fs_data_parent, "d400_feature")
dir_fs_data_target = os.path.join(dir_fs_data_parent, "d500_movement")
dir_fs_data_serv   = os.path.join(dir_fs_data_parent, "d600_serving")

# COMMAND ----------

# DBTITLE 1,directories 03
dir_mls_data_score = os.path.join(dir_mls_data_parent, "d400_model_score")

# COMMAND ----------

# DBTITLE 1,directories 04
dir_mls_marketing_parent = "/mnt/ml-store-dev/dev_users/dev_el/marketing_programs"

# COMMAND ----------

# DBTITLE 1,directories 05
dir_mls_marketing_wip = os.path.join(dir_mls_marketing_parent, "wip")
dir_mls_marketing_output = os.path.join(dir_mls_marketing_parent, "global_control")

# COMMAND ----------

# MAGIC %md
# MAGIC ## s100 data import

# COMMAND ----------

# DBTITLE 1,feature store
df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))

# COMMAND ----------

display(df_fs_master.limit(10))

# COMMAND ----------

# DBTITLE 1,ml store
df_mls_score_churn = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_churn_pred30d"))
df_mls_score_dr = spark.read.format("delta").load(os.path.join(dir_mls_data_score, "mobile_oa_consumer_srvc_device_replacement_pred30d"))

# COMMAND ----------

# DBTITLE 1,additional features - age
df_x_age = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
           select
                *
                , siebel_row_id as fs_cust_id
            from prod_aws_prod_masked.stage_perm.ds_edw_party_customer_ml_360
            qualify row_number() over (
                partition by siebel_row_id
                order by party_deact_dt desc, party_start_dt desc, last_upd_dt desc
            ) = 1
        """
    )
    .load()
    .withColumn(
        "age_group"
        , f.when(
            f.col("age") < 0
            , f.lit("z.unknown")
        )
        .otherwise(f.col("age_group"))
    )
    .withColumn(
        "age"
        , f.when(
            f.col("age") < 0
            , f.lit(None)
        )
        .otherwise(f.col("age"))
    )
)

df_x_age = lower_col_names(df_x_age)

display(df_x_age.limit(10))

export_data(
    df = df_x_age
    , export_path = os.path.join(dir_mls_marketing_wip, "x_age")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_x_age = spark.read.format("delta").load(os.path.join(dir_mls_marketing_wip, "x_age"))

# COMMAND ----------

# DBTITLE 1,additional features - converged
df_x_converged = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , f"""
            with tmp as (
                select 
                    d_snapshot_date_key
                    , s.service_id as fs_srvc_id
                    , billing_account_number as fs_acct_id
                    , c.customer_source_id as fs_cust_id
                    , service_access_type_name
                    , s.plan_name
                    , s.proposition_product_name
                    , s.broadband_discount_oa_msisdn as converged_oa_msisdn
                from prod_pdb_masked.modelled.f_service_activity_daily_snapshot f 
                inner join prod_pdb_masked.modelled.d_service_curr s 
                    on f.service_source_id = s.service_source_id 
                    and s.current_record_ind = 1 
                inner join prod_pdb_masked.modelled.d_billing_account_curr b 
                    on b.billing_account_source_id = s.billing_account_source_id 
                    and b.current_record_ind = 1
                inner join prod_pdb_masked.modelled.d_customer_curr c 
                    on c.customer_source_id = b.customer_source_id
                    and c.current_record_ind = 1
                where  
                    s.service_type_name in ('Broadband')
                    and 
                    ( 
                        (service_access_type_name is not null)
                        or 
                        ( s.service_type_name in ('Broadband')
                        and service_access_type_name is null   
                        and s.proposition_product_name in ('home phone wireless broadband discount proposition'
                                                            , 'home phone plus broadband discount proposition')
                        )
                    )
                    and c.market_segment_name in ('Consumer')
                    and f.d_snapshot_date_key in ({vt_param_date_key})
            )
            select distinct
                fs_cust_id
                , fs_acct_id
                , 'Y' as converged_flag
            from tmp
           
        """
    )
    .load()
)

df_x_converged = lower_col_names(df_x_converged)

display(df_x_converged.limit(5))

export_data(
    df = df_x_converged
    , export_path = os.path.join(dir_mls_marketing_wip, "x_converged")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_x_converged = spark.read.format("delta").load(os.path.join(dir_mls_marketing_wip, "x_converged"))

# COMMAND ----------

# DBTITLE 1,additional features - wallet flag
df_x_wallet_flag = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            select
                billing_account_number as fs_acct_id
                , wallet_id as fs_srvc_id
                , 'Y' as wallet_flag
            from PROD_WALLET.MODELLED.D_WALLET_CUSTOMER
            where 1 = 1  
                and current_record_ind = 1
                and wallet_eligibility_flag = 'Y'
                and service_status_name = 'Active'
            qualify row_number() over (
                partition by billing_account_number, wallet_id
                order by record_end_date_time desc, record_start_date_time desc
            ) = 1

        """
    )
    .load()
)

df_x_wallet_flag = lower_col_names(df_x_wallet_flag)

display(df_x_wallet_flag.limit(5))

export_data(
    df = df_x_wallet_flag
    , export_path = os.path.join(dir_mls_marketing_wip, "x_wallet_flag")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_x_wallet_flag = spark.read.format("delta").load(os.path.join(dir_mls_marketing_wip, "x_wallet_flag"))

# COMMAND ----------

# DBTITLE 1,additional features - wallet balance
df_x_wallet_balance = (
    spark.read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , f"""
            select
                billing_account_number as fs_acct_id
                , wallet_id as fs_srvc_id
                , sum(balance_earned) as wallet_balance
            from prod_wallet.serving.ds_yt_wallet_daily_ss
            where
                1 = 1
                and (balance_earned > 0)
                and d_date_key = {vt_param_date_key}
                and campaign_type <> 'Trade-in'
            group by billing_account_number, wallet_id
        """
    )
    .load()
)

df_x_wallet_balance = lower_col_names(df_x_wallet_balance)

display(df_x_wallet_balance.limit(5))

export_data(
    df = df_x_wallet_balance
    , export_path = os.path.join(dir_mls_marketing_wip, "x_wallet_balance")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_x_wallet_balance = spark.read.format("delta").load(os.path.join(dir_mls_marketing_wip, "x_wallet_balance"))

# COMMAND ----------

# DBTITLE 1,wallet program control
df_wallet_control_01 = spark.read.format("delta").load("/mnt/ml-lab/dev_shared/wallet/20240924_wallet_program_control_endless_medium_plus")
df_wallet_control_02 = spark.read.format("delta").load("/mnt/ml-lab/dev_shared/wallet/20240924_wallet_program_control_endless_small")

display(df_wallet_control_01.limit(10))
display(df_wallet_control_02.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## s200 data transformation

# COMMAND ----------

vt_param_reporting_date_max = (
    df_fs_master
    .filter(
        (f.col("reporting_date") <= vt_param_date)
        & (f.col("reporting_cycle_type") == "rolling cycle")
    )
    .agg(f.max("reporting_date")).collect()[0][0]
)

# COMMAND ----------

# DBTITLE 1,base creation
df_base = (
    df_fs_master
    .filter(
        (f.col("reporting_date") == vt_param_reporting_date_max)
        & (f.col("reporting_cycle_type") == "rolling cycle")
    )
    .select(
        "reporting_date"
        , "reporting_cycle_type"
        , "fs_cust_id"
        , "fs_acct_id"
        , "fs_srvc_id"
        , "market_segment_type"
        , "srvc_privacy_flag"
        , "srvc_tenure"
        , "plan_family"
        , "plan_name_std"
        , "rev_mmc_tot_1cycle"
        , "rev_tot_1cycle"
        , "usg_data_avg_3cycle"
        , "ifp_prm_dvc_flag"
        , "ifp_acct_dvc_flag"
        , "plan_discount_flag"
        , "network_dvc_brand"
    )
    .withColumn(
        "plan_name_std"
        , f.when(
            f.col("plan_family").isin(["unlimited data", "endless data"])
            , f.col("plan_name_std")
        )
        .otherwise(f.col("plan_family"))
    )
    .withColumn(
        "network_dvc_brand"
        , f.when(
            f.col("network_dvc_brand").isin(["Apple", "Samsung"])
            , f.col("network_dvc_brand")
        )
        .otherwise(f.lit("Misc"))
    )
    .join(
        df_x_age
        .select(
            "fs_cust_id"
            , "age"
            , "age_group"
        )
        , ["fs_cust_id"]
        , "left"
    )
    .join(
        df_x_converged
        , ["fs_cust_id", "fs_acct_id"]
        , "left"
    )
    .join(
        df_x_wallet_flag
        .select(
            "fs_acct_id"
            , "fs_srvc_id"
            , "wallet_flag"
        )
        , ["fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_x_wallet_balance
        .select(
            "fs_acct_id"
            , "fs_srvc_id"
            , "wallet_balance"
        )
        , ["fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .fillna(
        value=0
        , subset=["wallet_balance"]
    )
    .fillna(
        value="N"
        , subset=["converged_flag", "wallet_flag"]
    )
    .fillna(
        value="z.unknown"
        , subset=["age_group"]
    )
    .join(
        df_mls_score_churn
        .select(
            "reporting_date"
            , "reporting_cycle_type"
            , "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , f.col("propensity_score").alias("churn_score")
            , f.col("propensity_top_ntile").alias("churn_top_ntile")
            , f.col("propensity_segment_qt").alias("churn_segment")
            , f.struct(
                  f.col("model").alias("model")
                , f.col("model_version").alias("model_version")
                , f.col('predict_date').alias("predict_date")
                , f.col("propensity_score").cast('double').alias("propensity_score")
                , f.col("propensity_segment_qt").alias("propensity_segment")
                , f.col("propensity_top_ntile").cast('double').alias("propensity_top_ntile")
            ).alias('churn_model_array')
        )
        , ["reporting_date", "reporting_cycle_type", "fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_mls_score_dr
        .select(
            "reporting_date"
            , "reporting_cycle_type"
            , "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , f.col("propensity_score").alias("dr_score")
            , f.col("propensity_top_ntile").alias("dr_top_ntile")
            , f.col("propensity_segment_qt").alias("dr_segment")
            , f.struct(
                  f.col("model").alias("model")
                , f.col("model_version").alias("model_version")
                , f.col('predict_date').alias("predict_date")
                , f.col("propensity_score").cast('double').alias("propensity_score")
                , f.col("propensity_segment_qt").alias("propensity_segment")
                , f.col("propensity_top_ntile").cast('double').alias("propensity_top_ntile")
            ).alias('dr_model_array')
        )
        , ["reporting_date", "reporting_cycle_type", "fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .withColumn("tenure_top_ntile", f.ntile(10).over(Window.orderBy(f.desc("srvc_tenure"))))
    .withColumn("mmc_top_ntile", f.ntile(10).over(Window.orderBy(f.desc("rev_mmc_tot_1cycle"))))
    .withColumn("usg_top_ntile", f.ntile(10).over(Window.orderBy(f.desc("usg_data_avg_3cycle"))))
    .withColumn("wallet_top_ntile", f.ntile(10).over(Window.orderBy(f.desc("wallet_balance"))))
    .withColumn("churn_top_ntile", f.ntile(10).over(Window.orderBy(f.desc("churn_score"))))
    .withColumn("dr_top_ntile", f.ntile(10).over(Window.orderBy(f.desc("dr_score"))))
    .join(
        df_wallet_control_01
        .select(
            "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , f.lit("Y").alias("wallet_control_flag_01")
        )
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .join(
        df_wallet_control_02
        .select(
            "fs_cust_id"
            , "fs_acct_id"
            , "fs_srvc_id"
            , f.lit("Y").alias("wallet_control_flag_02")
        )
        , ["fs_cust_id", "fs_acct_id", "fs_srvc_id"]
        , "left"
    )
    .withColumn(
        "wallet_control_flag"
        , f.when(
            (f.col("wallet_control_flag_01") == "Y") 
            | (f.col("wallet_control_flag_02") == "Y")
            , "Y"
        )
        .otherwise("N")
    )
    .drop("wallet_control_flag_01", "wallet_control_flag_02")
)

export_data(
    df = df_base
    , export_path = os.path.join(dir_mls_marketing_wip, "sample_input")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_base = spark.read.format("delta").load(os.path.join(dir_mls_marketing_wip, "sample_input"))

# COMMAND ----------

# DBTITLE 1,base data check
display(
    df_base
    .groupBy("reporting_date")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
    )
)

display(df_base.limit(10))

# COMMAND ----------

# DBTITLE 1,sample creation 01
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 40000
ls_param_strata_fields = [
    "churn_top_ntile"
    , "mmc_top_ntile"
    , "dr_top_ntile"
    #, "plan_family"
    #, "wallet_top_ntile"
    #, "tenure_top_ntile", "usg_top_ntile", "wallet_top_ntile"
    #, "plan_family", "ifp_acct_dvc_flag", "plan_discount_flag", "wallet_flag"
    #, "age_group", "network_dvc_brand", "srvc_privacy_flag"
]
vt_param_priority_field = "wallet_control_flag"
dict_param_priority_groups = {'Y':2}

df_base_control, df_base_target = generate_sample(
    df=df_base
    , size=vt_param_sample_req
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=15
)

# COMMAND ----------

# DBTITLE 1,sample creation 02
export_data(
    df = df_base_control
    , export_path = os.path.join(dir_mls_marketing_wip, "control")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_base_control = spark.read.format("delta").load(os.path.join(dir_mls_marketing_wip, "control"))

export_data(
    df = df_base_target
    , export_path = os.path.join(dir_mls_marketing_wip, "target")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = True
    , flag_dynamic_partition = False    
)

df_base_target = spark.read.format("delta").load(os.path.join(dir_mls_marketing_wip, "target"))

# COMMAND ----------

# DBTITLE 1,sample data check
display(
    df_base_control
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
    )
)

display(
    df_base_control
    .groupBy("reporting_date", "wallet_control_flag")
    .agg(
        f.count("*")
        , f.countDistinct("fs_srvc_id")
    )
)

# COMMAND ----------

# DBTITLE 1,sample evaluation 01
evaluate_sample(
    df_base_control
    , df_base_target
    , ["churn_score", "dr_score", "srvc_tenure", "rev_mmc_tot_1cycle", "usg_data_avg_3cycle", "wallet_balance"]
)

# COMMAND ----------

# DBTITLE 1,sample evaluation 02
evaluate_sample(
    df_base_control
    , df_base_target
    , [
        "plan_family", "plan_name_std"
        , "ifp_acct_dvc_flag", "plan_discount_flag"
        , "converged_flag"
        , "wallet_flag", "network_dvc_brand" 
        , "srvc_privacy_flag"
        , "age_group"
    ]
)

# COMMAND ----------

display(
    df_base_control
    .groupBy("churn_segment")
    .agg(f.countDistinct("fs_srvc_id").alias("srvc"))
    .withColumn(
        "srvc_tot"
        , f.sum("srvc").over(Window.partitionBy(f.lit(1)))
    )
    .withColumn(
        "pct"
        , f.round(f.col("srvc") / f.col("srvc_tot") * 100, 2)
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## s300 data export

# COMMAND ----------

ls_param_final_fields = [
    "reporting_date", "reporting_cycle_type"
    , "fs_cust_id", "fs_acct_id", "fs_srvc_id"
    , "control_group_id", "control_group_start_date", "segment"
    , "churn_model", "churn_score"
    , "dr_model", "dr_score"
    , "srvc_privacy_flag", "srvc_tenure", "age", "age_group"
    , "plan_family", "plan_name_std", "plan_discount_flag"
    , "rev_mmc_tot_1cycle", "rev_tot_1cycle", "usg_data_avg_3cycle"
    , "converged_flag"
    , "wallet_flag", "wallet_balance"
    , "ifp_prm_dvc_flag", "ifp_acct_dvc_flag", "network_dvc_brand"
    , "data_update_date", "data_update_dttm"
]

# COMMAND ----------

# DBTITLE 1,export data grooming
df_output_control = (
  df_base_control
  .select(
    "reporting_date"
    , "reporting_cycle_type"
    , "fs_cust_id"
    , "fs_acct_id"
    , "fs_srvc_id"
    , f.lit(vt_param_control_group_id).alias("control_group_id")
    #, f.current_date().alias("control_group_start_date")
    , f.col("reporting_date").alias("control_group_start_date")
    , "market_segment_type"
    , "churn_score"
    , "churn_model_array"
    , "dr_score"
    , "dr_model_array"
    , "srvc_privacy_flag", "srvc_tenure", "age", "age_group"
    , "plan_family", "plan_name_std", "plan_discount_flag", "rev_mmc_tot_1cycle"
    , "rev_tot_1cycle", "usg_data_avg_3cycle", "converged_flag", "wallet_flag"
    , "wallet_balance", "ifp_prm_dvc_flag", "ifp_acct_dvc_flag", "network_dvc_brand"
    , f.current_date().alias("data_update_date")
    , f.current_timestamp().alias("data_update_dttm")
  )
)

# COMMAND ----------

display(df_output_control.limit(10))

# COMMAND ----------

dbutils.fs.rm(os.path.join(dir_mls_marketing_output, "mobile_oa_consumer"), True)

# COMMAND ----------

export_data(
    df = df_output_control
    , export_path = os.path.join(dir_mls_marketing_output, "mobile_oa_consumer")
    , export_format = "delta"
    , export_mode = "overwrite"
    , flag_overwrite_schema = False
    , flag_dynamic_partition = True  
    , ls_dynamic_partition = ["control_group_id", "control_group_start_date"] 
)

# COMMAND ----------

os.path.join(dir_mls_marketing_output, "mobile_oa_consumer")
