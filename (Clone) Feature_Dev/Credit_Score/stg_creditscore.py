# Databricks notebook source
# MAGIC %md
# MAGIC ### Library

# COMMAND ----------

import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

# DBTITLE 1,directory
df_credit_score_raw = spark.read.format('delta').load("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/CREDIT_SCORE")

# COMMAND ----------

# DBTITLE 1,select requre fields
ls_param_field = [
    'combine_cust_id'
    , 'combine_credit_check_id'
    , 'combine_credit_check_score'
    , 'combine_credit_check_date'
    , 'combine_credit_dealer'
    , 'SBL_X_REASON_CODE'
    , 'sbl_x_reason_description'
] 

# COMMAND ----------

df_credit_score_raw = (
    df_credit_score_raw
    .select(ls_param_field)
)

# COMMAND ----------

 df_credit_score_raw = df_credit_score_raw.toDF(*[c.lower() for c in df_credit_score_raw.columns])

# COMMAND ----------

# DBTITLE 1,rename
std_params = {
    'col_map': {
        'combine_cust_id': 'fs_cust_id'
      , 'combine_credit_check_id' : 'credit_check_id'
      , 'combine_credit_check_score' : 'credit_check_score'
      , 'combine_credit_check_date': 'credit_check_date'
      , 'combine_credit_dealer': 'credit_check_dealer'
      , 'sbl_x_reason_code' : 'sbl_x_reason_code'
      , 'sbl_x_reason_description' : 'sbl_x_reason_desc'
    }
}

ls_param_col_map = std_params['col_map']

# COMMAND ----------

# DBTITLE 1,output
df_output_curr = (
    df_credit_score_raw
    .select(
        [f.col(c).alias(ls_param_col_map.get(c, c)) for c in  df_credit_score_raw.columns]
        )
)


display(df_output_curr.limit(100))

# COMMAND ----------

# DBTITLE 1,check null pct
display(df_output_curr
        .filter(f.col('sbl_x_reason_code').isNotNull())
        .groupBy('sbl_x_reason_code')
        .agg(
            f.count('fs_cust_id')
            , f.countDistinct('fs_cust_id')
        )
        ) 

        # 1,110,624 out of 1,708,660 missing... 

# COMMAND ----------

 # dbutils.fs.rm('dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/stg_credit_score', True)

# COMMAND ----------

# DBTITLE 1,export to prm layer
(
    df_output_curr
    .write
    .format("delta")
    .mode("overwrite")
   # .partitionBy("rec_created_month")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save('dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d200_intermediate/stg_credit_score')
)
