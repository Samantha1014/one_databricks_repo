# Databricks notebook source
# MAGIC %md ##Payment Request
# MAGIC
# MAGIC Created: 17/5/2024
# MAGIC
# MAGIC Last updated: 17/5/2024
# MAGIC
# MAGIC
# MAGIC Input:
# MAGIC
# MAGIC * 'dbfs:/mnt/feature-store-prod-lab/d100_raw/d101_dp_c360/d_billing_account/reporting_cycle_type=rolling cycle' (payment data)
# MAGIC * 'dbfs:/mnt/ml-lab/dev_users/dev_ks/account_risk/customer_curr_20240517' (Current customer)
# MAGIC
# MAGIC
# MAGIC **Objective**
# MAGIC
# MAGIC Insight analysis to support decision making on removing paper bills, or increasing charge of paperbills. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Request from Renee Thrower** (6/5/2024 via email)
# MAGIC
# MAGIC We are working through some options in the payment space to either reduce costs, or recover costs. For these we need a bit of data to help us with the impacts and decision making.
# MAGIC
# MAGIC 1.	Removing paper bills, or increasing charge of paperbills.
# MAGIC
# MAGIC •	We want a bit more understanding of the types of customers getting paper bills.
# MAGIC
# MAGIC •	This includes: Segment, age, account tenure, account monthly value, Avg days to pay if possible (are they on time or late payers), how they pay, do we have an email address noted on their account for something other than billing.
# MAGIC
# MAGIC •	Also if there is any way to see those getting paperbills but not being charged the current $2.50. There is a flag in Siebel that if an agent selects the customer will still get a paperbill but wont be charged.
# MAGIC
# MAGIC 2.	Adding a Refund link to the bill or automating refunds
# MAGIC
# MAGIC •	We have two options here, but both we can assess with same data
# MAGIC
# MAGIC •	Over a 4-6 month period can we see by month how many customers churn (full churn with no more services on the account) and are left with either a credit balance on the account, $0, or owing us some month. We would exclude involuntary churn here.
# MAGIC
# MAGIC •	We also need to see the values likely grouped so we can see what vols and values this could impact (@Dolores BoisvertHerrmann any thoughts on grouping here?)
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Features required**
# MAGIC
# MAGIC * Segment, age, account tenure, account monthly value, Avg days to pay if possible (are they on time or late payers), how they pay, do we have an email address noted on their account for something other than billing.
# MAGIC
# MAGIC * `data_update_date` - `billing_account_open_date` = account tenure 
# MAGIC * `bill_paper_charge_exemption_flag` (getting paperbills but not being charged the current $2.50)
# MAGIC * `bill_payment_method_desc` (e.g. Other, Credit Card, Direct Debit, Cheque)
# MAGIC * `bill_delivery_type_desc` (e.g. Paper, Email+CSV and post, Email+CSV, EBill)
# MAGIC * `billing_notification_email` (email address) -- if 'Unknown' then had_email_flag = 0 otherwise 1.
# MAGIC * `current_balance_bill_amount` (account monthly value)
# MAGIC * `bill_day_of_month_number`
# MAGIC * `billing_account_status_desc` (e.g. Active, Closed, null)
# MAGIC * PROD_MAR_TECH.SERVING.CUSTOMER_CURR[CUSTOMER_MKT_SEGMENT, CUSTOMER_BIRTH_DATE] (Segment, Age) -- joining PROD_MAR_TECH.SERVING.CUSTOMER_CURR[customer_id] = df_raw_bill[customer_source_id]
# MAGIC

# COMMAND ----------

# MAGIC %md ##1. Initial set up

# COMMAND ----------

import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format

# COMMAND ----------

# MAGIC %md ##2. Input data

# COMMAND ----------

##--- Input data

# payment
df_raw_bill = spark.read.format('delta').load('dbfs:/mnt/feature-store-prod-lab/d100_raw/d101_dp_c360/d_billing_account/reporting_cycle_type=rolling cycle')

# bill_delivery_type_desc  to define paper or ebill 
# current recode, set "current_record_ind == 1"

# customer info
df_raw_cust = spark.read.format('delta').load('dbfs:/mnt/ml-lab/dev_users/dev_ks/account_risk/customer_curr_20240517')

# COMMAND ----------

# sample data
display(df_raw_bill
        .limit(1000))

# COMMAND ----------

display(df_raw_bill
        .select('bill_delivery_type_desc', 'billing_account_number', 'bill_payment_method_desc')
        #.distinct()
        .groupBy('bill_delivery_type_desc','bill_payment_method_desc')
        .agg(f.countDistinct('billing_account_number'))
        )

# COMMAND ----------

display(df_raw_bill
        .filter((df_raw_bill["current_record_ind"] == 1))   
        .withColumn(
                   "active_service",
                   f.when(df_raw_bill["number_of_active_service_count"] == "null", None).otherwise(df_raw_bill["number_of_active_service_count"].cast("integer"))
        )        
        .groupBy("billing_account_status_desc")
        .agg(f.countDistinct('billing_account_number').alias("acct_cnt"),
             f.min("active_service").alias("min_active_service"),
             f.max("active_service").alias("max_active_service"),
             f.avg("active_service").alias("avg_active_service")
        )
)
        

# COMMAND ----------

# number_of_active_service_count
display(df_raw_bill
        .filter(df_raw_bill["current_record_ind"] == 1)
       # .withColumn(
       #            "active_service",
       #            f.when(df_raw_bill["number_of_active_service_count"] == "null", None).otherwise(df_raw_bill["number_of_active_service_count"].cast("integer"))
       # )
        #.groupBy('number_of_active_service_count')
       # .agg(f.countDistinct('billing_account_number').alias("acct_cnt")
       #      f.min("active_service").alias("min_active_service"),
       #      f.max("active_service").alias("max_active_service"),
       #      f.avg("active_service").alias("avg_active_service")
       # )
       .limit(100) 
)



# COMMAND ----------

##--- the latest record at billing account level, record_end_date = '9999-12-31' 

display(df_raw_bill
        .filter(df_raw_bill["customer_source_id"] == "1-7YXAFNX")
        .orderBy("customer_source_id", "record_end_date")
        )

# COMMAND ----------

# sample data
display(df_raw_cust
        .withColumn("customer_age", f.round(f.months_between(f.col("RECORD_UPDATE_DATE_TIME"), f.col("CUSTOMER_BIRTH_DATE"))/12)) 
        .withColumn("age",
                   f.when(f.year("CUSTOMER_BIRTH_DATE") == 1900, None)  # Set to NULL if year is 1900
                   .otherwise(f.round(f.months_between(f.col("RECORD_UPDATE_DATE_TIME"), f.col("CUSTOMER_BIRTH_DATE"))/12))
        )
        .limit(100))

# COMMAND ----------

# MAGIC %md ##3. Process data

# COMMAND ----------

##--- create new flags and groups

df_proc = (df_raw_bill
        .filter((df_raw_bill["current_record_ind"] == 1) &
                (df_raw_bill["billing_account_status_desc"] == 'Active'))
        # had_email_flag
        .withColumn(
                  "had_email_flag",
                  f.when(f.col("billing_notification_email").isNull(), 'N')
                   .when(f.col("billing_notification_email") != 'Unknown', 'Y')
                  .otherwise('N')
              )           
        #  acct_tenure_mth, acct_tenure_group
        .withColumn("acct_tenure_mth",
                    f.months_between(f.col("data_update_date"), f.col("billing_account_open_date"))
              )
        .withColumn(
                  "acct_tenure_group",
                  f.when(f.col("acct_tenure_mth").isNull(), "z) Unknown")
                   .when(f.col("acct_tenure_mth") <=3, "a) 0-3 mth")
                   .when(f.col("acct_tenure_mth") <= 6, "b) 4-6 mth")
                   .when(f.col("acct_tenure_mth") <= 12, "c) 7-12 mth")
                   .when(f.col("acct_tenure_mth") <= 24, "d) 1-2 yr")
                   .when(f.col("acct_tenure_mth") <= 36, "e) 2-3 yr")
                   .when(f.col("acct_tenure_mth") <= 60, "f) 3-5 yr")
                   .when(f.col("acct_tenure_mth") <= 120, "g) 5-10 yr")
                   .when(f.col("acct_tenure_mth") > 120, "h) 11+ yr")
                  .otherwise("z) Unknown")
              )   
        # active_srvc_cnt, active_srvc_group
        .withColumn(
                   "active_srvc_cnt",
                   f.when(df_raw_bill["number_of_active_service_count"] == "null", None).otherwise(df_raw_bill["number_of_active_service_count"].cast("integer"))
        )      
        .withColumn(
                  "active_srvc_group",
                  f.when(f.col("active_srvc_cnt").isNull(), "Unknown")
                   .when(f.col("active_srvc_cnt") <= 4, f.col("active_srvc_cnt").cast("string"))
                   .when(f.col("active_srvc_cnt") >= 5, "5+")
                  .otherwise("Unknown")
              ) 
        # get segment, cust_age_group
        .join(df_raw_cust
              .withColumn("customer_age",
                   f.when(f.year("CUSTOMER_BIRTH_DATE") == 1900, None)  # Set to NULL if year is 1900
                   .otherwise(f.round(f.months_between(f.col("RECORD_UPDATE_DATE_TIME"), f.col("CUSTOMER_BIRTH_DATE"))/12))
              )
              .withColumn(
                  "cust_age_group",
                  f.when(f.col("customer_age").isNull(), "z) Unknown")
                   .when(f.col("customer_age") < 21, "a) <21")
                   .when(f.col("customer_age") < 31, "b) 21-30")
                   .when(f.col("customer_age") < 41, "c) 31-40")
                   .when(f.col("customer_age") < 51, "d) 41-50")
                   .when(f.col("customer_age") < 51, "e) 51-60")
                   .when(f.col("customer_age") > 60, "f) 61+")
                  .otherwise("z) Unknown")
              ) 
              .select("customer_id", "customer_mkt_segment", "cust_age_group"),              
              
              df_raw_bill["customer_source_id"] == df_raw_cust["customer_id"], how="left")        
)
       
        

# COMMAND ----------

display(df_proc.limit(100))

# COMMAND ----------

# summary data
df_smr = (df_proc
          .groupBy("customer_mkt_segment", "collection_status_code", "cust_age_group", "acct_tenure_group", 
                   "bill_payment_method_desc", "bill_delivery_type_desc", "bill_paper_charge_exemption_flag",  
                   "active_srvc_group", "had_email_flag")
          .agg(
              f.countDistinct('billing_account_number').alias("acct_cnt"),
              f.round(f.avg(f.col("current_balance_bill_amount")),2).alias("bill_balance_avg"),
              f.min(f.col("current_balance_bill_amount")).alias("bill_balance_min"),
              f.max(f.col("current_balance_bill_amount")).alias("bill_balance_max"),

              f.round(f.avg(f.col("active_srvc_cnt"))).alias("active_srvc_avg"),
              f.min(f.col("active_srvc_cnt")).alias("active_srvc_min"),
              f.max(f.col("active_srvc_cnt")).alias("active_srvc_max")              
          )
)

# COMMAND ----------

display(df_smr)

# COMMAND ----------

# create flags and grouping variables

display(df_raw_bill
        .filter((df_raw_bill["current_record_ind"] == 1) &
                (df_raw_bill["billing_account_status_desc"] == 'Active'))
        # had_email_flag
        .withColumn(
                  "had_email_flag",
                  f.when(f.col("billing_notification_email").isNull(), 0)
                   .when(f.col("billing_notification_email") != 'Unknown', 1)
                  .otherwise(0)
              )           
        #  acct_tenure_mth, acct_tenure_group
        .withColumn("acct_tenure_mth",
                    f.months_between(f.col("data_update_date"), f.col("billing_account_open_date"))
              )
        .withColumn(
                  "acct_tenure_group",
                  f.when(f.col("acct_tenure_mth").isNull(), "z) Unknown")
                   .when(f.col("acct_tenure_mth") <=3, "a) 0-3 mth")
                   .when(f.col("acct_tenure_mth") <= 6, "b) 4-6 mth")
                   .when(f.col("acct_tenure_mth") <= 12, "c) 7-12 mth")
                   .when(f.col("acct_tenure_mth") <= 24, "d) 1-2 yr")
                   .when(f.col("acct_tenure_mth") <= 36, "e) 2-3 yr")
                   .when(f.col("acct_tenure_mth") <= 60, "f) 3-5 yr")
                   .when(f.col("acct_tenure_mth") <= 120, "g) 5-10 yr")
                   .when(f.col("acct_tenure_mth") > 120, "h) 11+ yr")
                  .otherwise("z) Unknown")
              )   
        # active_srvc_cnt, active_srvc_group
        .withColumn(
                   "active_srvc_cnt",
                   f.when(df_raw_bill["number_of_active_service_count"] == "null", None).otherwise(df_raw_bill["number_of_active_service_count"].cast("integer"))
        )      
        .withColumn(
                  "active_srvc_group",
                  f.when(f.col("active_srvc_cnt").isNull(), "Unknown")
                   .when(f.col("active_srvc_cnt") <= 4, f.col("active_srvc_cnt").cast("string"))
                   .when(f.col("acct_tenure_mth") >= 5, "5+")
                  .otherwise("Unknown")
              )                       
        # get segment, cust_age_group
        .join(df_raw_cust
              .withColumn("customer_age",
                   f.when(f.year("CUSTOMER_BIRTH_DATE") == 1900, None)  # Set to NULL if year is 1900
                   .otherwise(f.round(f.months_between(f.col("RECORD_UPDATE_DATE_TIME"), f.col("CUSTOMER_BIRTH_DATE"))/12))
              )
              .withColumn(
                  "cust_age_group",
                  f.when(f.col("customer_age").isNull(), "z) Unknown")
                   .when(f.col("customer_age") < 21, "a) <21")
                   .when(f.col("customer_age") < 31, "b) 21-30")
                   .when(f.col("customer_age") < 41, "c) 31-40")
                   .when(f.col("customer_age") < 51, "d) 41-50")
                   .when(f.col("customer_age") < 51, "e) 51-60")
                   .when(f.col("customer_age") > 60, "f) 61+")
                  .otherwise("z) Unknown")
              ) 
              .select("customer_id", "customer_mkt_segment", "cust_age_group"),              
              
              df_raw_bill["customer_source_id"] == df_raw_cust["customer_id"], how="left")
              
        .limit(100)
)

# COMMAND ----------

# MAGIC %md ##4. Save output

# COMMAND ----------

from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("CSV Export Example") \
    .getOrCreate()

# Assuming df_result is your DataFrame

# Specify the local path where you want to save the CSV file
output_path = "dbfs:/mnt/ml-lab/dev_users/dev_ks/account_risk/payment_summery_20240517.csv"

# Write the DataFrame to CSV
df_smr.coalesce(1).write \
    .format("csv") \
    .option("header", "true") \
    .mode("overwrite") \
    .save(output_path)

# COMMAND ----------

# MAGIC %md ### Average days to pay

# COMMAND ----------

# average days to pay 

dbutils.fs.ls('/mnt/feature-store-prod-lab/d200_staging/d299_src/')

# COMMAND ----------


