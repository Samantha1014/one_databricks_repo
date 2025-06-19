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
from pyspark.sql.functions import when
from pyspark.sql.functions import *
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
print(row_count)

# COMMAND ----------

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession 

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------


# Register the DataFrame as a temporary view
df_inteflow_credit_check.createOrReplaceTempView("my_table")

display(spark.sql("select count(1) from my_table limit 100"))

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
# MAGIC #IFP on service

# COMMAND ----------

df_ifp_on_service = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """  SELECT
            EVENT_DATE as join_date
            , DATE_TRUNC("MONTH", EVENT_DATE) AS EVENT_MONTH
            , EVENT
            , CUSTOMER_SIEBEL_ROW_ID AS cust_src_id
            , ACCT_NUM
            , PRIM_ACCS_NUM
            , SBSCNORDER as siebel_order_num
            , MERCHANDISE_ID
            , OA_IF_DEVICE_TYPE as ifp_type
            , OA_IF_DEVICE as ifp_model
            , DEVICE_RRP as ifp_rrp
            , HOB_IND
           , OA_IF_START_DT as ifp_term_start_date
           , OA_IF_END_DT as ifp_term_end_date
            , OA_IF_TERM_TTL as ifp_term
            , OA_IF_DEVICE_AMOUNT as ifp_value
        FROM PROD_AWS_PROD_MASKED.STAGE_PERM.DS_EDW_IFP_EVENT_SUMMARY_ML 
        where ifp_term_start_date >= '2022-01-01'
        and event in ('New Interest Free Contract')
        and hob_ind in ('Activation')
        AND OA_IF_DEVICE_TYPE IN ('Device') """
    )
    .load()
)
display(df_ifp_on_service.limit(100))


# COMMAND ----------

# MAGIC %md
# MAGIC #ifp on bill 

# COMMAND ----------

df_ifp_on_bill = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ with D_ACCT AS (
    SELECT
        BILLING_ACCOUNT_NUMBER
    , BILLING_ACCOUNT_SOURCE_ID
    FROM PROD_PDB_MASKED.MODELLED.D_BILLING_ACCOUNT
QUALIFY  ROW_NUMBER() OVER(PARTITION BY BILLING_ACCOUNT_SOURCE_ID 
ORDER BY RECORD_END_DATE_TIME DESC, RECORD_START_DATE_TIME DESC) = 1
)
, EXTRACT_TARGET AS (
    SELECT
        DISTINCT
        ---TOP 100
          TO_DATE(TAB_A.ORDER_DATE) AS join_date
        , TO_DATE(date_trunc('MONTH', TAB_A.ORDER_DATE)) AS event_month
        , TAB_A.ORDER_SUB_TYPE as event
        , TAB_A.BILLING_ACCOUNT_SOURCE_ID as cust_src_id
        , TAB_C.BILLING_ACCOUNT_NUMBER as acct_num
        , TAB_A.LINKED_SERVICE_ID as prim_accs_num
        , TAB_A.ORDER_NUMBER as siebel_order_num
        , TAB_A.MERCHANDISE_ID
        , TAB_A.IFP_TYPE
        , TAB_A.MODEL_NAME as ifp_model
        , TAB_A.IFP_RECOMMENDED_RETAIL_PRICE as ifp_rrp
        , TAB_A.TRANSACTION_TYPE as event_sub_type
        , TAB_A.IFP_START_DATE as ifp_term_start_date
        , TAB_A.IFP_END_DATE as ifp_term_end_date
        , TAB_A.IFP_TERM_MONTHS as ifp_term
        , TAB_A.IFP_TOTAL_VALUE as ifp_value
    FROM PROD_PDB_MASKED.MODELLED.F_IFP_SALES_TRANSACTION TAB_A
    LEFT JOIN D_ACCT TAB_C
        ON TAB_A.BILLING_ACCOUNT_SOURCE_ID = TAB_C.BILLING_ACCOUNT_SOURCE_ID
)
SELECT 
    *
FROM EXTRACT_TARGET
WHERE ifp_term_start_date >= '2022-01-01'
and EVENT_SUB_TYPE in ('Sale')
and ifp_type in ('Ifp-Device') """
    )
    .load()
)
display(df_ifp_on_bill.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC #ifp combine 

# COMMAND ----------

df_ifp_full = df_ifp_on_service.union(df_ifp_on_bill)
display(df_ifp_full.limit(100))
print(df_ifp_full.count())

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
# MAGIC ##inner join

# COMMAND ----------

## only join with credit score is not null and consumer only 
df_joined = df_ifp_full.alias("a").join(
    df_credit_score.alias("b"),
    f.col("acct_num") == f.col("account_no"),
    "inner"
).filter(f.col("customer_market_segment")=="Consumer")

display(df_joined.limit(100))
print(df_joined.count())
# df_filtered = df_joined.filter(f.col("customer_market_segment")=="Consumer")

# COMMAND ----------

# calculate ifp provision to write off 
df_result = df_joined.withColumn(
    "ifp_write_off_period",
    f.datediff(
        f.to_date(f.concat(f.col("account_write_off_month"), f.lit("-01")), "yyyy-MM-dd"),
        f.to_date(f.col("event_month"), "yyyy-MM")
    )
)
display(df_result.limit(100))
print(df_result.count()) 
#64133 rows 

# check write_off percentage in general 
write_off_count = df_result.where(f.col('ACCOUNT_WRITE_OFF_AMOUNT').isNotNull()).count()
row_count = df_result.count()
print('write off percentage', write_off_count/row_count*100)

# COMMAND ----------

# MAGIC %md
# MAGIC #get 5% population percentaile score 

# COMMAND ----------

# get credit score and account no 
df_percentile = df_result.select(col('ACCOUNT_NO'), col('FIRST_CREDIT_CHECK_SCORE' ))

# count of account no by first credit check score 
df_percentile = df_percentile \
.groupBy("FIRST_CREDIT_CHECK_SCORE") \
.agg(
f.count('ACCOUNT_NO').alias('cnt_account_no')
)
display(df_percentile)



# COMMAND ----------

# Calculate cumulative count and percentage
df_percentile = df_percentile.withColumn('cnt_cumsum', sum(col('cnt_account_no')).over(Window.orderBy(f.asc("FIRST_CREDIT_CHECK_SCORE"))))
total_accounts = df_percentile.select(sum('cnt_account_no')).collect()[0][0]
# print(total_accounts)
# print(df_percentile) 

df_percentile = df_percentile.withColumn('pct_cumsum', col('cnt_cumsum') / total_accounts)
# display(df_percentile)
percentiles = [i / 100.0 for i in range(5, 101, 5)]

percentile_scores = {}
for percentile in percentiles:
    closest_row = df_percentile.filter(df_percentile["pct_cumsum"] >= percentile).orderBy("pct_cumsum").first()
    score = closest_row["FIRST_CREDIT_CHECK_SCORE"]
    percentile_scores[percentile] = score
display(percentile_scores)
# df_percentile_scores = spark.createDataFrame(percentile_scores.items(), ["Percentile", "Credit Score"])
# display(df_percentile_scores)



# COMMAND ----------

print(percentile_scores[0.05])

# COMMAND ----------

 
 df= df_result.withColumn('group',
                    when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[1.0] , 1.0) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.95] , 0.95) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.90] , 0.90) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.85] , 0.85)      
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.8] , 0.8) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.75] , 0.75) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.7] , 0.7)      
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.65] , 0.65) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.6] , 0.6) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.55] , 0.55) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.5] , 0.5)      
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.45] , 0.45) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.4] , 0.4) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.35] , 0.35) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.3] , 0.3) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.25] , 0.25) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.2] , 0.2) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.15] , 0.15) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.1] , 0.1) 
                   .when(df_result['FIRST_CREDIT_CHECK_SCORE'] > percentile_scores[0.05] , 0.05) 
                   .otherwise(0))
 
 display(df.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC #feature store intermediate layer load 

# COMMAND ----------

# calendar cycle  int layer load 
dir_data_master = '/mnt/feature-store-prod-lab/d200_intermediate/d201_mobile_oa_consumer/'
dir_reporting_cycle = 'reporting_cycle_type=calendar cycle'
dir_data_ifp_on_bill = os.path.join(dir_data_master, "int_ssc_ifp_on_bill/", dir_reporting_cycle)
dir_data_ifp_on_service = os.path.join(dir_data_master, "int_ssc_ifp_on_service", dir_reporting_cycle )


df_ifp_on_bill_raw = spark.read.format("delta").load(dir_data_ifp_on_bill)
df_ifp_on_service_raw = spark.read.format("delta").load(dir_data_ifp_on_service)
display(df_ifp_on_bill_raw.limit(100))
display(df_ifp_on_service_raw.limit(100))

# COMMAND ----------

display(df_ifp_on_bill_raw.select(f.max("ifp_event_date")).collect()[0][0]) # 2024-02-04
display(df_ifp_on_bill_raw.select(f.min("ifp_event_date")).collect()[0][0]) # 2022-12-11 

display(df_ifp_on_service_raw.select(f.max("ifp_event_date")).collect()[0][0]) # 2024-02-04
display(df_ifp_on_service_raw.select(f.min("ifp_event_date")).collect()[0][0]) # 2021-08-15 

# COMMAND ----------

display(
    df_ifp_on_bill_raw.groupBy("ifp_event_type").agg(
        f.count("*").alias("cnt_row"), 
        f.countDistinct("fs_ifp_id").alias("cnt_ifp_id")
    )
)

# COMMAND ----------

df_ifp_on_bill_new = df_ifp_on_bill_raw.filter((f.col("ifp_event_type" ) =="new") & (f.col("ifp_type") =="device") & (f.col("ifp_event_date") >="2022-01-01"))
print(df_ifp_on_bill_new.count())
display(df_ifp_on_bill_new.limit(100))
display(df_ifp_on_bill_new.select(f.max("ifp_event_date")).collect()[0][0]) # datetime.date(2024, 2, 1)
display(df_ifp_on_bill_new.select(f.min("ifp_event_date")).collect()[0][0]) # datetime.date(2022, 12, 7)


# COMMAND ----------

df_ifp_on_service_new = df_ifp_on_service_raw.filter((f.col("ifp_event_type" ) =="new") & (f.col("ifp_type") =="device") & (f.col("ifp_event_date")>="2022-01-01"))
print(df_ifp_on_service_new.count())
display(df_ifp_on_service_new.limit(100))
display(df_ifp_on_service_new.select(f.max("ifp_event_date")).collect()[0][0]) 
display(df_ifp_on_service_new.select(f.min("ifp_event_date")).collect()[0][0]) 

# COMMAND ----------

# make sure record are distinct 
display(
    df_ifp_on_bill_new.groupBy("ifp_event_type").agg(
        f.count("*").alias("cnt_row"), 
        f.countDistinct("fs_ifp_id").alias("cnt_ifp_id")
    )
)

# make sure record are distinct 
display(
    df_ifp_on_service_new.groupBy("ifp_event_type").agg(
        f.count("*").alias("cnt_row"), 
        f.countDistinct("fs_ifp_id").alias("cnt_ifp_id")
    )
)


# COMMAND ----------

dir_bill_account = '/mnt/feature-store-prod-lab/d200_intermediate/d201_mobile_oa_consumer/int_ssc_billing_account'
df_bill_account = spark.read.format("delta").load(dir_bill_account)

display(df_bill_account.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC #Primary Layer Load 

# COMMAND ----------

dir_test_bill = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_bill/reporting_cycle_type=calendar cycle'
df_test_bill = spark.read.format("delta").load(dir_test_bill)
#display(df_bill_test.limit(100))

dir_test_service = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_service/reporting_cycle_type=calendar cycle'
df_test_service = spark.read.format("delta").load(dir_test_service)
display(df_test_service.limit(100))



# COMMAND ----------

df_test_service_new = df_test_service.filter((f.col("ifp_event_type" ) =="new") & (f.col("ifp_type") =="device") & (f.col("ifp_order_date")>="2022-01-01"))
df_test_bill_new =  df_test_bill.filter((f.col("ifp_event_type" ) =="new") & (f.col("ifp_type") =="device") & (f.col("ifp_order_date")>="2022-01-01"))
print(df_test_service_new.count())
print(df_test_bill_new.count())
display(df_test_service_new.limit(100))
display(df_test_bill_new.limit(100))
# calculate min max 
display(df_test_service_new.select(f.max("ifp_order_date")).collect()[0][0]) 
display(df_test_service_new.select(f.min("ifp_order_date")).collect()[0][0]) 

# COMMAND ----------

display(
    df_test_service_new.groupBy("ifp_event_type").agg(
        f.count("*").alias("cnt_row"), 
        f.countDistinct("fs_ifp_id").alias("cnt_ifp_id")
    )
)

# COMMAND ----------

display(
    df_test_bill_new.groupBy("ifp_event_type").agg(
        f.count("*").alias("cnt_row"), 
        f.countDistinct("fs_ifp_id").alias("cnt_ifp_id")
    )
)

# COMMAND ----------

display(df_test_bill_new.limit(100))
display(df_test_service_new.limit(100))

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

df_prm_bill_new = df_test_bill_new.select(ls_columns)
df_prm_service_new = df_test_service_new.select(ls_columns)
display(df_prm_bill_new.limit(100))
display(df_prm_service_new.limit(100))

# COMMAND ----------

df_prim_ifp_full = df_prm_service_new.union(df_prm_bill_new)
display(df_prim_ifp_full.limit(100))
print(df_prim_ifp_full.count()) # 187497

# COMMAND ----------

## only join with credit score is not null and consumer only 
df_prim_joined = df_prim_ifp_full.alias("a").join(
    df_credit_score.alias("b"),
    f.col("fs_acct_id") == f.col("account_no"),
    "inner"
).filter(f.col("customer_market_segment")=="Consumer")

display(df_prim_joined.limit(100))
print(df_prim_joined.count()) # 88032


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
int_write_off_count_total = df_result.where(f.col('ACCOUNT_WRITE_OFF_AMOUNT').isNotNull()).count()
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
            (f.col("cum_sum_write_off_count")/int_write_off_count_total)*100
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



# COMMAND ----------

# MAGIC %md
# MAGIC #Raw Layer Load Check

# COMMAND ----------

dir = "/mnt/feature-store-prod-lab/d100_raw/d101_dp_c360/f_ifp_sales_transaction" 
df_raw_layer_ifp_on_bill = spark.read.format("delta").load(dir)
display(df_raw_layer_ifp_on_bill.limit(100))


# COMMAND ----------

# display(df_raw_layer_ifp_on_service.filter(f.col('billing_account_num') == '490358793'))
display(df_ifp_on_service_raw.filter(f.col('fs_acct_id')=='490756900'))


# COMMAND ----------

dir_service_raw = '/mnt/feature-store-prod-lab/d100_raw/d102_dwh_edw/ifp_events_on_service'
df_raw_layer_ifp_on_service = spark.read.format("delta").load(dir_service_raw)
display(df_raw_layer_ifp_on_service.limit(100))
display(df_raw_layer_ifp_on_service.select(f.min("event_date")).collect()[0][0])
display(df_raw_layer_ifp_on_service.select(f.max("event_date")).collect()[0][0])

# COMMAND ----------


