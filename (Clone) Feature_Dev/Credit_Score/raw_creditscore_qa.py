# Databricks notebook source
# MAGIC %md
# MAGIC ### Library

# COMMAND ----------

import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from pyspark.sql.functions import date_format

# COMMAND ----------

df_credit_score_raw = spark.read.format('delta').load("dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/CREDIT_SCORE")

# COMMAND ----------

display(df_credit_score_raw.limit(100))
display(df_credit_score_raw.count())

# COMMAND ----------

display(df_credit_score_raw
        .groupBy(
            date_format(f.col('combine_credit_check_date'), 'yyyy-MM')
        )
       .agg(f.count('combine_cust_id')
            , f.countDistinct('combine_cust_id')
            )
       )


# COMMAND ----------


