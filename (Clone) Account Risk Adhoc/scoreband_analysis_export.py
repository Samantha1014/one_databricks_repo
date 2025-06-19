# Databricks notebook source
# MAGIC %md
# MAGIC **scoreband analysis** 

# COMMAND ----------

# MAGIC %md
# MAGIC objective - for initial credit check score, how does the current credict score banding perform in terms of filtering out bad debt customers? \
# MAGIC -- current score criteria 375 to 400, does more bad debt incur more at higher score (i.e. scoring system deteroriate) or actually the opposite? \
# MAGIC -- if we were to raise score, which is our optimal point? 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library

# COMMAND ----------

import pyspark
import os
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from jinja2 import Template
from pyspark.sql import SparkSession 
from datetime import date
from datetime import timedelta
import numpy as np
import pandas as pd
import datetime as dt
from pyspark.sql.functions import sum, col
# ------------- Use snowflake utility
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils


# COMMAND ----------

# MAGIC %md
# MAGIC ##parameter

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## DB connectivity

# COMMAND ----------

# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "PROD_INTEFLOW",
  "sfSchema": "RAW",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}


# COMMAND ----------

# MAGIC %md
# MAGIC # Credit Score Source Data Check

# COMMAND ----------

# post Aug-22, credict score come from interflow 
df_inteflow_credit_check = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , "select * from PROD_INTEFLOW.RAW.CREDITREPORT"
    )
    .load()
)
display(df_inteflow_credit_check.limit(100))
na_count = df_inteflow_credit_check.where(f.col('DNB_FINAL_SCORE').isNull()).count()
row_count = df_inteflow_credit_check.count()
print('missing value percentage', na_count/row_count*100)

# COMMAND ----------

# pre Aug-2022, credict score can be obtained from siebel financial profile 
df_siebel_credit_check = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , "select * from PROD_AWS_PROD_MASKED.STAGE_PERM.DS_SIEBEL_S_FINAN_PROF"
    )
    .load()
)

display(df_siebel_credit_check.limit(100))
na_count = df_siebel_credit_check.where(f.col('x_credit_score').isNull()).count()
row_count = df_siebel_credit_check.count()
print('missing value percentage', na_count/row_count*100)

# COMMAND ----------

display(
    df_inteflow_credit_check
    .groupBy("existing_cust")
    .agg(
        f.count("*").alias("cnt")
        , f.countDistinct("ID").alias("cnt_Credit_check_ID")
        , f.avg("DNB_FINAL_SCORE")
        )
    .sort(f.desc("cnt_Credit_check_ID"))
    )
 

# COMMAND ----------

display(
    df_siebel_credit_check
    .groupBy("X_EXISTING_CUSTOMER")
    .agg(
        f.count("*").alias("cnt")
        , f.countDistinct("IDENTIFIER").alias("cnt_IDENTIFIER")
        , f.avg("x_credit_score")
        )
    .sort(f.desc("cnt_IDENTIFIER"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #credit score and payment 

# COMMAND ----------

df_credit_score = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
        with credit_score as (
        select 
        CUSTOMER_MARKET_SEGMENT
        ,current_credit_check_score
        , first_credit_check_score
        , first_credit_check_current_status
        , current_credit_check_current_status
        , first_credit_check_submit_date
        , current_credit_check_submit_date
        , customer_alt_source_id
        , account_source_id
        , account_no
        , account_name
        , account_write_off_amount
        , account_write_off_month
	   from prod_account_risk.modelled.f_account_risk_monthly_snapshot
        where first_credit_check_score is not null 
	   qualify row_number() over(partition by account_no order by d_snapshot_date_key desc) = 1
        ), 
        last_payment as (
        select 
        lastmodifieddatetime as last_payment_date
        , account_obj_id0
        , amount as last_payment_amount 
        from PROD_BRM.RAW.PINPAP_EVENT_BAL_IMPACTS_T
        where 
        item_obj_type in ('/item/payment')
        qualify row_number()over(partition by account_obj_id0 order by lastmodifieddatetime desc) = 1
        ), 
        credit_score_payment as (
        select * from credit_score c left join last_payment l on c.account_source_id = l.account_obj_id0
        )
        select * from credit_score_payment 
         """
    )
    .load()
)
display(df_credit_score.limit(100))
print(df_credit_score.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #IFP Primary Layer Load 

# COMMAND ----------

dir_prim_bill = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_bill/reporting_cycle_type=calendar cycle'
df_prim_ifp_bill = spark.read.format("delta").load(dir_prim_bill)
#display(df_bill_test.limit(100))

dir_prim_service = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_service/reporting_cycle_type=calendar cycle'
df_prim_ifp_service = spark.read.format("delta").load(dir_prim_service)
display(df_prim_ifp_service.limit(100))


# COMMAND ----------

df_prim_ifp_service = df_prim_ifp_service.filter((f.col("ifp_event_type" ) =="new") & (f.col("ifp_type") =="device") & (f.col("ifp_order_date")>="2022-01-01"))
df_prim_ifp_bill =  df_prim_ifp_bill.filter((f.col("ifp_event_type" ) =="new") & (f.col("ifp_type") =="device") & (f.col("ifp_order_date")>="2022-01-01"))
print(df_prim_ifp_service.count())
print(df_prim_ifp_bill.count())
display(df_prim_ifp_service.limit(100))
display(df_prim_ifp_bill.limit(100))
# calculate min max 
display(df_prim_ifp_bill.select(f.max("ifp_order_date")).collect()[0][0]) 
display(df_prim_ifp_service.select(f.min("ifp_order_date")).collect()[0][0]) 

# COMMAND ----------

display(
    df_prim_ifp_service.groupBy("ifp_event_type").agg(
        f.count("*").alias("cnt_row"), 
        f.countDistinct("fs_ifp_id").alias("cnt_ifp_id")
    )
)

# COMMAND ----------

display(
    df_prim_ifp_bill.groupBy("ifp_event_type").agg(
        f.count("*").alias("cnt_row"), 
        f.countDistinct("fs_ifp_id").alias("cnt_ifp_id")
    )
)

# COMMAND ----------

# select column to union 
ls_columns = ['reporting_date',
'reporting_cycle_type',
'fs_cust_id',
'fs_acct_id',
'fs_ifp_id',
'fs_ifp_order_id',
'ifp_order_date',
'ifp_level',
'ifp_type',
'ifp_order_num',
'ifp_merchandise_id',
'ifp_model',
'ifp_rrp',
'ifp_term',
'ifp_term_start_date',
'ifp_term_end_date',
'ifp_sales_channel_group',
'ifp_sales_channel',
'ifp_sales_channel_branch',
'ifp_event_type'
]

df_prim_ifp_bill = df_prim_ifp_bill.select(ls_columns)
df_prim_ifp_service = df_prim_ifp_service.select(ls_columns)
display(df_prim_ifp_bill.limit(100))
display(df_prim_ifp_service.limit(100))

# COMMAND ----------

# union ifp on bill and ifp on service 
df_prim_ifp_full = df_prim_ifp_service.union(df_prim_ifp_bill)
display(df_prim_ifp_full.limit(100))
print(df_prim_ifp_full.count()) # 187497

# COMMAND ----------

# MAGIC %md
# MAGIC # Inner Join with Credit Score

# COMMAND ----------

## only join with credit score is not null and consumer only 
df_prim_joined = df_prim_ifp_full.alias("a").join(
    df_credit_score.alias("b"),
    f.col("fs_acct_id") == f.col("account_no"),
    "inner"
).filter(f.col("customer_market_segment")=="Consumer")

# ifp provision to write off in months 
df_prim_joined = df_prim_joined.withColumn(
    "ifp_write_off_period",
    f.datediff(
        f.to_date(f.concat(f.col("account_write_off_month"), f.lit("-01")), "yyyy-MM-dd"),
        f.to_date(f.col("ifp_order_date"), "yyyy-MM-dd")
    )/30.25
)

display(df_prim_joined.limit(100))
print(df_prim_joined.count()) # 88032

# check write_off percentage in general 
int_write_off_count = df_prim_joined.where(f.col('ACCOUNT_WRITE_OFF_AMOUNT').isNotNull()).count()
int_row_count = df_prim_joined.count()
print('write off percentage', int_write_off_count/int_row_count*100)


# COMMAND ----------

df_result = df_prim_joined.withColumn(
    "quartile_grp"
    ,f.ntile(20).over(
        Window
       .orderBy(f.asc("FIRST_CREDIT_CHECK_SCORE"))
    )
)
display(df_result.limit(100))




# COMMAND ----------

df_quartile_result = df_result\
.groupBy("quartile_grp") \
.agg( 
    f.max("FIRST_CREDIT_CHECK_SCORE").alias("max_credit_score")
    , f.min("FIRST_CREDIT_CHECK_SCORE").alias("min_credit_score")
    , f.count("fs_acct_id").alias("cnt_account")
    , f.count(f.when(f.col("ACCOUNT_WRITE_OFF_AMOUNT").isNotNull(),True)).alias("write_off_count")
)\
.sort(f.asc("quartile_grp"))


# COMMAND ----------

# get cumsum of write off count 

display(df_quartile_result.withColumn(
    "write_off_percentage"
    ,(f.col("write_off_count")/f.col("cnt_account"))*100
                                    )
        .withColumn("cum_sum_write_off_count"
                     , f.sum("write_off_count").over( 
                                                     Window.orderBy("quartile_grp")
                                                    .rowsBetween(Window.unboundedPreceding,0))
                    )
        .withColumn(
             "baseline%",
            (f.col("cum_sum_write_off_count")/int_write_off_count)*100
                   )
        .withColumn("uplift", 
            f.col("baseline%")/(f.col("quartile_grp")*5)
                    )      
)
# check write_off percentage in general 
#write_off_count = df_result.where(f.col('ACCOUNT_WRITE_OFF_AMOUNT').isNotNull()).count()
#row_count = df_result.count()
#print('write off percentage', write_off_count/row_count*100)


# COMMAND ----------

# MAGIC %md
# MAGIC # Export to Snowflake

# COMMAND ----------

vt_param_table_curr = "lab_ml_store.sandbox.account_risk_score_band_tracker"
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "lab_ml_store",
  "sfSchema": "sandbox",
  "sfWarehouse": "LAB_DS_WH"
}

# write table to snowflake 
df_result.write \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", vt_param_table_curr) \
  .mode("overwrite") \
  .save()



