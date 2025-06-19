# Databricks notebook source
# MAGIC %md
# MAGIC ## s1 environment setup

# COMMAND ----------

# DBTITLE 1,library
import pyspark
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md ### directories

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/dev_sc/")

# COMMAND ----------

dbutils.fs.ls(dir_data_parent_users)

# COMMAND ----------

dir_data_raw =  os.path.join(dir_data_parent_shared, 'd100_raw')
#dir_data_meta = os.path.join(dir_data_parent_users, 'd000_meta')
#dir_data_stg = os.path.join(dir_data_parent_users, "d200_staging")
dir_data_int =  os.path.join(dir_data_parent_users, "d200_intermediate")
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d301_mobile_oa_consumer")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d401_mobile_oa_consumer")
#dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d501_mobile_oa_consumer")
#dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
#dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## s1 data import

# COMMAND ----------

df_stg_creditscore = spark.read.format('delta').load(os.path.join(dir_data_int, "stg_credit_score"))
df_prm_unit_base = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base")

# COMMAND ----------

# DBTITLE 1,sample data check
display(df_stg_creditscore.limit(100))

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
    , "fs_cust_id", "fs_acct_id", "fs_srvc_id"
]

ls_param_credit_joining_keys = [
    "fs_cust_id"
]

# COMMAND ----------

# DBTITLE 1,unit base
df_base_unit_base_curr = (
    df_prm_unit_base
    .filter(f.col("reporting_date") == vt_param_ssc_reporting_date)
    .filter(f.col("reporting_cycle_type") == vt_param_ssc_reporting_cycle_type)
    .select(ls_param_unit_base_fields)
)

# COMMAND ----------

# display(df_base_credit_init_curr.limit(10))
display(df_base_unit_base_curr
        .agg(f.countDistinct('fs_cust_id'))
)

print('-------')

# display(df_stg_creditscore.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## base credit score

# COMMAND ----------

df_base_credit_init_curr = (
    df_stg_creditscore
    .join(
        df_base_unit_base_curr
        .select(ls_param_credit_joining_keys)
        .distinct()
        , ls_param_credit_joining_keys
        , "inner"
    )
    .withColumn(
        'r_airtime', 
        f.when(
            f.col('sbl_x_reason_code').contains('R_AIRTIME'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'c_create', 
        f.when(
            f.col('sbl_x_reason_code').contains('C_CREATE'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'd_scrlow', 
        f.when(
            f.col('sbl_x_reason_code').contains('D_SCRLOW'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'd_matrix', 
        f.when(
            f.col('sbl_x_reason_code').contains('D_MATRIX'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'c_occup', 
        f.when(
            f.col('sbl_x_reason_code').contains('C_OCCUP'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'ageifpalert', 
        f.when(
            f.col('sbl_x_reason_code').contains('AgeIFPAlert'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'c_colctns', 
        f.when(
            f.col('sbl_x_reason_code').contains('C_COLCTNS'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'r_age', 
        f.when(
            f.col('sbl_x_reason_code').contains('R_AGE'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'r_matrix', 
        f.when(
            f.col('sbl_x_reason_code').contains('R_MATRIX'), f.lit('Y')
        ).otherwise('N')
    )
    .withColumn(
        'd_bnkrpcy', 
        f.when(
            f.col('sbl_x_reason_code').contains('D_BNKRPCY'), f.lit('Y')
        ).otherwise('N')
    )

)

# COMMAND ----------

ls_param_export_fields= [
    'fs_cust_id', 
    'reporting_date', 
    'reporting_cycle_type', 
    'credit_check_score', 
    'credit_check_dealer', 
    'sbl_x_reason_code', 
    'r_airtime', 
    'c_create', 
    'd_scrlow', 
    'd_matrix', 
    'c_occup', 
    'ageifpalert', 
    'c_colctns', 
    'r_age', 
    'r_matrix', 
    'd_bnkrpcy'
]

# COMMAND ----------

df_output_curr = (
    df_base_unit_base_curr
    .join(df_base_credit_init_curr, ls_param_credit_joining_keys, "inner")
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .select(*ls_param_export_fields)
)

# COMMAND ----------

display(df_output_curr)

# COMMAND ----------

display(df_base_credit_init_curr.limit(10))

# COMMAND ----------

# DBTITLE 1,check pct
display(df_base_credit_init_curr
        .agg(f.countDistinct('fs_cust_id'))
        
        ) # 389463 /537080 = 72% 

display(df_base_unit_base_curr
        .agg(f.countDistinct('fs_cust_id'))
        )

# COMMAND ----------


