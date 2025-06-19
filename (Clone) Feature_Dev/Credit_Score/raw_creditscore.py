# Databricks notebook source
# MAGIC %md
# MAGIC ### S00001 Library

# COMMAND ----------

import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "prod_interflow",
  "sfSchema": "raw",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}


# COMMAND ----------

df_credit_score = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
       
        SELECT 
        CUSTREF as INT_CUSTREF
        ,ID AS INT_ID
        ,DECISION AS INT_DECISION
        ,PRE_BU_SCORE AS INT_PRE_BU_SCORE
        ,DNB_FINAL_SCORE AS INT_DNB_FINAL_SCORE
        , SUBMIT AS INT_SUBMIT
        , DEALER  AS INT_DEALER
        FROM PROD_INTEFLOW.RAW.CREDITREPORT
        WHERE 
        (DNB_FINAL_SCORE IS NOT NULL) 
        and (CUSTREF IS NOT NULL)
        and _is_latest = 1 
        and _is_deleted = 0
        QUALIFY row_number()over(partition by INT_CUSTREF order by INT_SUBMIT desc) =1
         """
    )
    .load()
)


# COMMAND ----------

df_sbl_fin_profile = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
       select  
        ACCNT_ID AS SBL_ACCNT_ID
        ,IDENTIFIER AS SBL_IDENTIFIER
        , X_CREDIT_SCORE AS SBL_X_CREDIT_SCORE
        , X_CREDIT_REQ_DT AS SBL_X_CREDIT_REQ_DT
        , X_REASON_CODE AS SBL_X_REASON_CODE
        , X_REASON_DESCRIPTION AS SBL_X_REASON_DESCRIPTION
        , X_DEALER_CODE AS SBL_X_DEALER_CODE
        , X_DEALER_DESC AS SBL_X_DEALER_DESC
        from PROD_AWS_PROD_MASKED.STAGE_PERM.DS_SIEBEL_S_FINAN_PROF_HIST 
        WHERE X_CREDIT_SCORE IS NOT NULL 
        QUALIFY row_number()over(partition by SBL_ACCNT_ID order by SBL_X_CREDIT_REQ_DT desc) = 1;
         """
    )
    .load()
)

# COMMAND ----------

df_combine = (
        df_credit_score
        .join(df_sbl_fin_profile, f.col('INT_CUSTREF') ==f.col('SBL_ACCNT_ID'), 'full')
        .withColumn(
            'combine_cust_id', 
                    f.coalesce(
                                f.col('INT_CUSTREF')
                                , f.col('SBL_ACCNT_ID')
                                )
                    
                    )
        .withColumn(
            'combine_credit_check_id', 
                    f.coalesce(
                            f.col('INT_ID')
                         , f.col('SBL_IDENTIFIER')
                 )   
                    )
        .withColumn('combine_credit_check_score'
                    , f.coalesce(
                            f.col('INT_DNB_FINAL_SCORE')
                            , f.col('SBL_X_CREDIT_SCORE')
                    )
                    )
        .withColumn('combine_credit_check_date'
                    , f.coalesce(
                            f.col('INT_SUBMIT')
                            , f.col('SBL_X_CREDIT_REQ_DT')
                    )
        )
        .withColumn('combine_credit_dealer'
                    , f.coalesce(
                            f.col('INT_DEALER')
                            , f.col('SBL_X_DEALER_DESC')
                    )
        )
)

# COMMAND ----------

# df_output_curr = df_combine

# COMMAND ----------

# DBTITLE 1,export to stage layer
(
    df_combine
    .write
    .format("delta")
    .mode("overwrite")
   # .partitionBy("rec_created_month")
    .option("overwriteSchema", "true")
    .option("partitionOverwriteMode", "dynamic")
    .save("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/CREDIT_SCORE")
)


# COMMAND ----------

# DBTITLE 1,check output
display(df_sbl_fin_profile
        .agg(f.count('SBL_ACCNT_ID')
             , f.countDistinct('SBL_ACCNT_ID')
             )
        )

# COMMAND ----------

# DBTITLE 1,check output
display(df_credit_score.count())

print('sbl_fin_file')

display(df_sbl_fin_profile.count())

# COMMAND ----------

# DBTITLE 1,check output
display(df_credit_score.limit(10))

# COMMAND ----------

# DBTITLE 1,check output
display(df_sbl_fin_profile.limit(10))

# COMMAND ----------

# DBTITLE 1,check output
display(df_combine.limit(10))
