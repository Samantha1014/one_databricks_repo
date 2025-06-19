# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Score on OA Consumer Base 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Finding
# MAGIC 1. Credit check is based on customer level 
# MAGIC 2. Total credit check with first credit score is not null, bill level 685K, cust level 669K 
# MAGIC 3. Total OA consumer base (with min bill open date >= 2022-01-01), bill level 222996, cust level 221117
# MAGIC 4. inner join oa base with credit data using billing acct as foreign key, got 197334 out of 222996 = 88% 
# MAGIC 5. if using oa base(bill level) anti-join credit data and the anti-join set join back to credit data using cust id, we got 14 rows more can be imputed back 
# MAGIC 6. inner join oa base with credit data using cust id as foreight key, got 195835/  221117 = 88%  also
# MAGIC 7. if using oa base(cust level) anti-join credit data and then the anti-join set join back to credit data using bill acct, we got 1 rows more can be imputed back 
# MAGIC 8. Tenure calc - 75t pct - 8 months, 85 pct - 11 months, 95 pct - 16 months 
# MAGIC 9. 75 pct - {5% : 2.9} {10% - 2.59}, {15% - 2.22},{20% - 1.95}
# MAGIC 10. 85 pct - {5% : 2.84} {10% - 2.59}, {15% - 2.23},{20% - 1.96}
# MAGIC 11. 85 pct - {5% : 2.79} {10% - 2.56}, {15% - 2.21},{20% - 1.96}
# MAGIC 12.  reasons why not 100% new oa activation can be joined to credit score \
# MAGIC     a. there are missing data from interflow to account risk dataset - 2 records appear to have score in interflow but not in account risk \
# MAGIC     b. credit check perform in cust level, with the same customer it has done a credit check ages ago (e.g 2014), and open another account recently, in this case it wont have the first credit check score 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##library

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
# MAGIC ##DB connect

# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "Prod_Account_Risk",
  "sfSchema": "Modelled",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ##feature store directory

# COMMAND ----------

dir_oa_unit_base = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
#  feature layer - dbfs:/mnt/feature-store/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle
# dbfs:/mnt/feature-store/d300_primary/d301_mobile_oa_consumer/prm_unit_base
# int layer - dbfs:/mnt/feature-store/d200_intermediate/d201_mobile_oa_consumer/int_ssc_billing_account
# serv layer - dbfs:/mnt/feature-store/d600_serving/serv_mobile_oa_consumer/reporting_cycle_type=calendar cycle
df_oa_unit_base = spark.read.format("delta").load(dir_oa_unit_base)
display(df_oa_unit_base.limit(100))
print(df_oa_unit_base.count())


# COMMAND ----------

# test with only d_billing dimension in intermediate layer 
# dir_int_d_billing = 'dbfs:/mnt/feature-store-prod-lab/d200_intermediate/d201_mobile_oa_consumer/int_ssc_billing_account'
# df_int_d_billing = spark.read.format("delta").load(dir_int_d_billing)
# display(df_int_d_billing.limit(100))
# print(df_int_d_billing.count())

# COMMAND ----------

# display (df_oa_unit_base.filter(f.col('billing_acct_open_date') >= '2022-01-01').limit(100)) 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Get OA Consumer Billing Account

# COMMAND ----------

display(
    df_oa_unit_base
    .filter(f.col('fs_acct_id') == '490856466')
    .select('reporting_date', 'fs_cust_id', 'fs_acct_id', 'fs_srvc_id', 'billing_acct_open_date')
    .distinct()
    .orderBy(f.desc('reporting_date'), f.desc('billing_acct_open_date'))
)

# COMMAND ----------

# calculate duplicate account id with different cust id 
display(
    df_oa_unit_base
    .select("reporting_date", 'fs_acct_id', 'billing_acct_open_date')
    .distinct()
    .withColumn(
        "cnt"
        , f.count("*").over(
            Window
            .partitionBy('reporting_date', 'fs_acct_id')
        )
    )
    .select("reporting_date", "fs_acct_id", "cnt")
    .distinct()
    .withColumn(
        "dup_flag"
        , f.when(f.col("cnt") > 1, f.lit(1)).otherwise(f.lit(0))
    )
    .groupBy("reporting_date")
    .agg(
        f.countDistinct("fs_acct_id")
        , f.sum("dup_flag")
    )
    .orderBy(f.desc("reporting_date"))
)

# COMMAND ----------


ls_columns = [#'fs_acct_id',
              'reporting_date',
              'fs_cust_id',
              'fs_acct_id',
#'market_segment_type',
#'cust_start_date',
#'cust_tenure',
#'num_of_active_acct_cnt',
'billing_acct_open_date'
#'billing_acct_tenure',
#'product_holding_desc'
]
# df_oa_unit_base_account = df_oa_unit_base.select(ls_columns).filter(f.col('billing_acct_open_date')>='2022-01-01')

# check duplicate 
display(
    df_oa_unit_base
    .select(ls_columns)
    .distinct()
    .withColumn(
        "cnt"
        , f.count("*").over(
            Window
            .partitionBy("reporting_date",'fs_acct_id')
        )
    )
    .filter(f.col("cnt")>1)  
)


# 490856466
# 490856466
# 493589641
# 493589641
# 500307820
# 500307820
# 365225344
# 365225344
# 416468370
# 416468370
# 444636031
# 444636031
# 469876627
# 469876627
# 470916514


# remove duplicate 
#df_oa_unit_base_account = df_oa_unit_base_account.drop_duplicates(['fs_acct_id',  'billing_acct_open_date'])

#display(df_oa_unit_base_account.limit(100).orderBy('fs_acct_id'))
#print(df_oa_unit_base_account.count())

# COMMAND ----------

df_oa_unit_account=\
    df_oa_unit_base\
    .select(ls_columns)\
    .withColumn("index", 
                row_number()
                .over(Window
                      .partitionBy("fs_acct_id")
                      .orderBy("billing_acct_open_date","reporting_date")
                      )
                ) \
    .filter(f.col("index")==1)\
    .filter(f.col("billing_acct_open_date")>="2022-01-01")
    # .filter(f.col("fs_acct_id")=="500307820")

# one customer have multiple billing acct     
display(
    df_oa_unit_account
    .agg(
        f.count("*").alias("cnt")
        ,f.countDistinct("fs_acct_id").alias("disintct_cnt_acct")
        ,f.countDistinct("fs_cust_id").alias("distinct_cnt_cust")
    )
) # 222996 distinct account with 221117 distinct cust since Jan 2022

# check one billing account only have one billing account start date
# should return 0 result 
display(
    df_oa_unit_account
    .groupBy("fs_acct_id")
    .agg(
        f.count("billing_acct_open_date").alias("cnt")
    )
    .filter(f.col("cnt")>1)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Get credit score data from account risk

# COMMAND ----------

df_credit_score = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
       select 
         current_credit_check_score
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
        , service_types
	   from prod_account_risk.modelled.f_account_risk_monthly_snapshot
        where first_credit_check_score is not null 
	   qualify row_number() over(partition by account_no order by d_snapshot_date_key desc) = 1
         """
    )
    .load()
)


print(df_credit_score.count()) # 685K inlcude mobile oa, bb and other unknown 
display(df_credit_score.limit(100))
display(df_credit_score
        .agg(
            f.countDistinct("ACCOUNT_NO").alias("cnt_distinct_bill")
            ,f.count("*")
            ,f.countDistinct("CUSTOMER_ALT_SOURCE_ID").alias("cnt_distinct_cust")
        )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inner join based on billing account

# COMMAND ----------

## only join with credit score is not null and consumer only 
df_joined = df_oa_unit_account.alias("a").join(
    df_credit_score.alias("b"),
    f.col("a.fs_acct_id") == f.col("b.account_no"),
    "inner"
)

display(df_joined.limit(100))
print(df_joined.count()) # 197334 post inner join vs. oa base acct 222996
print(df_oa_unit_account.count())  # capture 88% of oa consumer base  

# COMMAND ----------

# anti-join
df_anti = df_oa_unit_account.alias("a").join(
    df_credit_score.alias("b"),
    f.col("a.fs_acct_id") == f.col("b.account_no"),
    "anti"
)
print(df_anti.count()) # 25662 rows


# join back to credit score using fs_cust_id to see how much records can be imputed back 

display(df_anti.join(
    df_credit_score,
    f.col("fs_cust_id")== f.col("customer_alt_source_id")
    , "inner"
)
        .select("reporting_date", "fs_cust_id","fs_acct_id","billing_acct_open_date","FIRST_CREDIT_CHECK_SCORE")
        .distinct()
        .count()
)
# only 14 rows can be joined back... with cust level 

# COMMAND ----------

# check oa consumer that didnt match with any credit score 
display(df_anti.join(
    df_credit_score,
    f.col("fs_cust_id")== f.col("customer_alt_source_id")
    , "anti"
)
        .select("reporting_date", "fs_cust_id","fs_acct_id","billing_acct_open_date")
)
# they are missing at source data but have records in interflow 
# check 3 records 
# fs_cust_id    fs_acct_id   bill_open
# 1-156VXLCH	489989727	2022-11-26 - has credit score 739 in interflow, credit check conducted on 28-11-2022 06:52:32 PM
# 1-8JE0WH1	    490470448	2023-01-29  - it's cust_id has another bill acct - 401120174, which joined in 18-12-2014, and interflow conducted credit check on 15-12-2014(775 score)
# 1-116J58U2	482474412	2022-01-01 - has conducted credit check 02-01-2022 accd to interflow, credit score 642, not show up in account risk data 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inner join based on cust id

# COMMAND ----------

## only join with credit score is not null and consumer only based on cust id 
df_joined_cust = df_oa_unit_account.alias("a").join(
    df_credit_score.alias("b"),
    f.col("a.fs_cust_id") == f.col("b.customer_alt_source_id"),
    "inner"
)

# display(df_joined.limit(100))
display(
    df_joined_cust
    .agg(
     f.count("*").alias("cnt")
        ,f.countDistinct("fs_acct_id").alias("disintct_cnt_acct")
        ,f.countDistinct("fs_cust_id").alias("distinct_cnt_cust")  # 195835 out of 221117
    )
)
print(df_oa_unit_account.agg(countDistinct("fs_cust_id")).collect()[0][0]) #195835/  221117 = 88% 

# COMMAND ----------

# anti-join
df_anti = df_oa_unit_account.alias("a").join(
    df_credit_score.alias("b"),
    f.col("a.fs_cust_id") == f.col("b.customer_alt_source_id"),
    "anti"
)
print(df_anti.count()) # 25649 rows


# join back to credit score using fs_acct_id to see how much records can be imputed back 

display(df_anti.join(
    df_credit_score,
    f.col("fs_acct_id")== f.col("account_no")
    , "inner"
)
        .select("reporting_date", "fs_cust_id","fs_acct_id","billing_acct_open_date","FIRST_CREDIT_CHECK_SCORE")
        .distinct()
        .count()
)
# only 1 billing acct can be join back using billing acct  ....

# COMMAND ----------

# MAGIC %md
# MAGIC ##tenure distribution

# COMMAND ----------

# tenure distribution 
print(
    df_joined
    .withColumn(
    "tenure_to_wo",
    round(f.datediff(
        f.to_date(f.concat(f.col("account_write_off_month"), f.lit("-01")), "yyyy-MM-dd"),
        f.to_date(f.col("billing_acct_open_date"), "yyyy-MM-dd")
    )/30.25)
    )
    .approxQuantile(
        "tenure_to_wo"
        ,[0.75,0.85,0.95]
        ,0.01)
    )

# 75 pct = 8 months 
# 85 pct = 11 months 
# 95 pct = 16 months 

# COMMAND ----------

# MAGIC %md
# MAGIC #Tenure Calculation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 75pct Tenure = 8 months

# COMMAND ----------

# calculate write off % by 8 months period (75pct)
df_result= df_joined \
.withColumn("write_off_date",f.to_date(f.concat(f.col("account_write_off_month"),f.lit("-01")),"yyyy-MM-dd"))\
.withColumn("end_date", f.add_months("billing_acct_open_date",8))\
.withColumn(
    "write_off_flag",
    f.when(
        (f.col("write_off_date") >= f.col("billing_acct_open_date")) & 
        (f.col("write_off_date") <= f.col("end_date")),1
    ).otherwise(0)
    )\
.withColumn(
    "quartile_grp"
    ,f.ntile(20).over(
        Window
       .orderBy(f.asc("FIRST_CREDIT_CHECK_SCORE"))
    )
)


# COMMAND ----------


df_quartile_result = df_result\
.groupBy("quartile_grp") \
.agg( 
    f.max("FIRST_CREDIT_CHECK_SCORE").alias("max_credit_score")
    , f.min("FIRST_CREDIT_CHECK_SCORE").alias("min_credit_score")
    , f.count("fs_acct_id").alias("cnt_account")
    , f.sum(f.when(f.col("write_off_flag")== 0,1).otherwise(0)).alias("cnt_non_write_off")
    , f.sum(f.when(f.col("write_off_flag")==1,1).otherwise(0)).alias("cnt_write_off")  
)\
.sort(f.asc("quartile_grp"))

# display(df_quartile_result)

# COMMAND ----------

# get cumsum of write off count 
# int_write_off_count_total = df_result.where(f.col('ACCOUNT_WRITE_OFF_AMOUNT').isNotNull()).count()
int_write_off_count_total = df_quartile_result.agg(sum("cnt_write_off").alias("total_write_off_cnt")).collect()[0][0] # 18073

display(df_quartile_result
        .withColumn('quartile value',
                    f.col("quartile_grp")*5)
        .withColumn(
    "write_off_percentage"
    ,(f.col("cnt_write_off")/f.col("cnt_account"))*100
                                    )
        .withColumn("cum_sum_write_off_count"
                     , f.sum("cnt_write_off").over( 
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


# COMMAND ----------

print('write off percentage', int_write_off_count_total/df_result.count()*100)  # 9% 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 85pct Tenure = 11 months

# COMMAND ----------

# calculate write off % by 11 months period (85pct)
df_result= df_joined \
.withColumn("write_off_date",f.to_date(f.concat(f.col("account_write_off_month"),f.lit("-01")),"yyyy-MM-dd"))\
.withColumn("end_date", f.add_months("billing_acct_open_date",11))\
.withColumn(
    "write_off_flag",
    f.when(
        (f.col("write_off_date") >= f.col("billing_acct_open_date")) & 
        (f.col("write_off_date") <= f.col("end_date")),1
    ).otherwise(0)
    )\
.withColumn(
    "quartile_grp"
    ,f.ntile(20).over(
        Window
       .orderBy(f.asc("FIRST_CREDIT_CHECK_SCORE"))
    )
)


# COMMAND ----------


df_quartile_result = df_result\
.groupBy("quartile_grp") \
.agg( 
    f.max("FIRST_CREDIT_CHECK_SCORE").alias("max_credit_score")
    , f.min("FIRST_CREDIT_CHECK_SCORE").alias("min_credit_score")
    , f.count("fs_acct_id").alias("cnt_account")
    , f.sum(f.when(f.col("write_off_flag")== 0,1).otherwise(0)).alias("cnt_non_write_off")
    , f.sum(f.when(f.col("write_off_flag")==1,1).otherwise(0)).alias("cnt_write_off")  
)\
.sort(f.asc("quartile_grp"))

# display(df_quartile_result)

# COMMAND ----------

# get cumsum of write off count 
int_write_off_count_total = df_quartile_result.agg(sum("cnt_write_off").alias("total_write_off_cnt")).collect()[0][0] # 20757
print(int_write_off_count_total)
display(df_quartile_result
        .withColumn('quartile value',
                    f.col("quartile_grp")*5)
        .withColumn(
    "write_off_percentage"
    ,(f.col("cnt_write_off")/f.col("cnt_account"))*100
                                    )
        .withColumn("cum_sum_write_off_count"
                     , f.sum("cnt_write_off").over( 
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


# COMMAND ----------

# MAGIC %md
# MAGIC ##95 pct Tenure = 16 months 

# COMMAND ----------

# calculate write off % by 11 months period (85pct)
df_result= df_joined \
.withColumn("write_off_date",f.to_date(f.concat(f.col("account_write_off_month"),f.lit("-01")),"yyyy-MM-dd"))\
.withColumn("end_date", f.add_months("billing_acct_open_date",16))\
.withColumn(
    "write_off_flag",
    f.when(
        (f.col("write_off_date") >= f.col("billing_acct_open_date")) & 
        (f.col("write_off_date") <= f.col("end_date")),1
    ).otherwise(0)
    )\
.withColumn(
    "quartile_grp"
    ,f.ntile(20).over(
        Window
       .orderBy(f.asc("FIRST_CREDIT_CHECK_SCORE"))
    )
)


# COMMAND ----------

df_quartile_result = df_result\
.groupBy("quartile_grp") \
.agg( 
    f.max("FIRST_CREDIT_CHECK_SCORE").alias("max_credit_score")
    , f.min("FIRST_CREDIT_CHECK_SCORE").alias("min_credit_score")
    , f.count("fs_acct_id").alias("cnt_account")
    , f.sum(f.when(f.col("write_off_flag")== 0,1).otherwise(0)).alias("cnt_non_write_off")
    , f.sum(f.when(f.col("write_off_flag")==1,1).otherwise(0)).alias("cnt_write_off")  
)\
.sort(f.asc("quartile_grp"))

# display(df_quartile_result)

# COMMAND ----------


int_write_off_count_total = df_quartile_result.agg(sum("cnt_write_off").alias("total_write_off_cnt")).collect()[0][0] #23387
print(int_write_off_count_total)

# get cumsum of write off count 
display(df_quartile_result
        .withColumn('quartile value',
                    f.col("quartile_grp")*5)
        .withColumn(
    "write_off_percentage"
    ,(f.col("cnt_write_off")/f.col("cnt_account"))*100
                                    )
        .withColumn("cum_sum_write_off_count"
                     , f.sum("cnt_write_off").over( 
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

