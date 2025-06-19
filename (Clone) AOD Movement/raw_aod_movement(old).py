# Databricks notebook source
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window 
from pyspark.sql.functions import lag, date_add

# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "prod_edw",
  "sfSchema": "raw",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

df_aod = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
       select 
       
       
       
       from prod_edw.raw.edw2prd_stageperm_s_inf_aod
         """
    )
    .load()
)

# COMMAND ----------

display(df_aod)

# COMMAND ----------

df_aod = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
         with extract_target as(
        select  
            --top 10
            regexp_replace(a.account_no, '^0+', '') as fs_acct_id
            , to_timestamp_ltz(v.created_t) as create_dttm
            , to_date(create_dttm) as create_date
            ,  v.aod as aod_current
            , v."30DAYS" as aod_30
            , v."60DAYS" as aod_60
            , v."90DAYS" as aod_90
            , v."120DAYS" as aod_120
            , v."150DAYS" as aod_150
            , v."180DAYS"  as aod_180
            , "180+DAYS" as aod_180plus
            , "60+DAYS" as aod_60plus 
            , dss_hist_start_dttm
            , dss_hist_end_dttm
            , dss_current_flag
            , dss_insert_dttm
            ,dss_update_dttm
        from prod_aws_prod_masked.stage_perm.ds_brm_vf_aging_bucket_t_hist v  
        inner join 
        prod_brm.raw.pinpap_account_t a 
        on v.account_obj_id0  = a.poid_id0
        where a.account_no not like 'S%'
        and a._is_latest = 1 and a._is_deleted = 0 
        ) 
        select *  from extract_target

         """
    )
    .load()
)

# COMMAND ----------

df_aod = df_aod.toDF(*[c.lower() for c in df_aod.columns])

# COMMAND ----------

display(df_aod.count()) # 397M 

# COMMAND ----------

df_prm_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle')

# COMMAND ----------

ls_param_unit_base_fields = [
    "reporting_date", "reporting_cycle_type"
    , "fs_cust_id", "fs_acct_id", "fs_srvc_id"
]

ls_param_aod_joining_keys = ['fs_acct_id']

vt_param_ssc_reporting_date = '2024-03-31'

# COMMAND ----------

df_prm_oa = (
    df_prm_oa
    .select(ls_param_unit_base_fields)
    .filter(f.col('reporting_date') == f.lit(vt_param_ssc_reporting_date))
)

# COMMAND ----------

# DBTITLE 1,# narrow down
df_aod_oa = (
    df_prm_oa
    .select('reporting_date', 'fs_acct_id')
    .distinct()
    .join( 
          df_aod
          .filter(f.col('create_date') >= 
            date_add(f.lit(vt_param_ssc_reporting_date), -180) 
            )
          .filter(f.col('create_date') <= f.lit(vt_param_ssc_reporting_date))
          , 
          ls_param_aod_joining_keys, 'left'
          )
)

# COMMAND ----------

display(df_aod_oa.count())  #44316028

# COMMAND ----------

# DBTITLE 1,export data to layer
(
    df_aod_oa
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("create_date")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d999_tmp/test_aod_movement")
)
