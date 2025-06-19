# Databricks notebook source
# DBTITLE 1,import library
import pyspark
import os 
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql.functions import regexp_replace 
from pyspark.sql import Window
from pyspark.sql.functions import col 

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

# MAGIC %run "./stg_billing_dev"

# COMMAND ----------

df_account_t = spark.read.format('delta').load(dir_acct_t)

df_account_t = (
    df_account_t
    .filter(col('_is_latest') ==1) # pick latest record
    .filter(col('_is_deleted') ==0) 
    .filter(~col('account_no').startswith('S')) # remove subscription account 
    .withColumn('account_no', regexp_replace("account_no", "^0+", "")) # remove leading 0 in account_t 
)


# COMMAND ----------

# DBTITLE 1,checck account_t
display(
    df_account_t
    .select('poid_id0', 'account_no')
    .distinct()
    .agg(f.countDistinct('poid_id0'), 
         f.count('poid_id0'),
         f.count('account_no')
        ))

# COMMAND ----------

# DBTITLE 1,check post filter 2021
display(df_bill_t_01
        .select('end_t','poid_id0','bill_no','account_obj_id0')
        .join(df_account_t.alias('a'), col('a.poid_id0') == col('account_obj_id0'), 'inner') # inner join with account_t with treatment 
        .withColumn('end_time', f.from_utc_timestamp(from_unixtime('end_t'),"Pacific/Auckland"))
        .withColumn('end_month', date_format("end_time","yyyy-MM"))
        .groupBy('end_month')
        .agg(
            f.count("*").alias('total_rowcnt')
            ,f.countDistinct('bill_no').alias('total_unique_bill')
            ,f.count('bill_no').alias('total_bill')
            , f.countDistinct('account_obj_id0').alias('total_unique_account')
        )
        )

# COMMAND ----------

# DBTITLE 1,check post remove duplicate
display(df_bill_t_02
        .select('end_t','poid_id0','bill_no','account_obj_id0')
        .join(df_account_t.alias('a'), col('a.poid_id0') == col('account_obj_id0'), 'inner') # inner join with account_t with treatment 
        .withColumn('end_time', f.from_utc_timestamp(from_unixtime('end_t'),"Pacific/Auckland"))
        .withColumn('end_month', date_format("end_time","yyyy-MM"))
        .groupBy('end_month')
        .agg(
            f.count("*").alias('total_rowcnt')
            ,f.countDistinct('bill_no').alias('total_unique_bill')
            ,f.count('bill_no').alias('total_bill')
            , f.countDistinct('account_obj_id0').alias('total_unique_account')
        )
        )

# COMMAND ----------

# DBTITLE 1,Post Transformation Check
display(df_bill_base
        .select('end_t','poid_id0','bill_no','account_obj_id0', 'account_no', 'end_month')
        .groupBy('end_month')
        .agg(
            f.count("*").alias('total_rowcnt')
            ,f.countDistinct('poid_id0').alias('cnt_unique_key')
            ,f.countDistinct('bill_no').alias('total_unique_bill')
            ,f.count('bill_no').alias('total_bill')
            ,f.countDistinct('account_no').alias('cnt_accnt_no')
            , f.countDistinct('account_obj_id0').alias('total_unique_account')
        )
        )

# COMMAND ----------

display(df_output_curr.count())
