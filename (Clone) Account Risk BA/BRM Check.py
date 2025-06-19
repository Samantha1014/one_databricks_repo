# Databricks notebook source
# MAGIC %md
# MAGIC ### FINDING & SUMMARY
# MAGIC 1. Using Bill_T join to account_t,  316,053/ 24,799,681 = 1.2% missing rate, account not exist to bill_t appears to be prepay only product holding 
# MAGIC 2. oa consumer unit base join with bill_t, there are 5 records exsiting in oa consumer base but not in bill_t data lake, but these 5 has records in account risk bill history - **suggesting there might be missing data from bill t data lake(this has been fixed)** 
# MAGIC 3. item_t can be used to reterieve write off and payment records. compare item_t write off vs. account risk write off, we have 27 out of 470,254 not match (less than 0.1%) and for these 27 are more close to source after spot checking 
# MAGIC 4. item_t payment compare to collection report(ATB) for its last payment date and payment amount, we have 2735/384396 ~ 0.7%  unmatch rate, and the unmatch reason is that collection atb grab the first record of the date if multiple payments made for the same date while we took the last.
# MAGIC 5. compare item_t payment vs. oa consumer, we have around 2.7% never payer (exclude those have account set up in jan, feb 2024 )

# COMMAND ----------

# MAGIC %md
# MAGIC ### LIBRARY

# COMMAND ----------

import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession 
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import row_number 
from pyspark.sql.functions import date_format
from pyspark.sql.functions import lag, col
from pyspark.sql.functions import datediff
from pyspark.sql.functions import regexp_replace

# COMMAND ----------

# MAGIC %md
# MAGIC ### DIRECTORY

# COMMAND ----------

dir_bill_t = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T'
dir_item_t = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_ITEM_T'
dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
dir_acct = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T'



# COMMAND ----------

# MAGIC %md
# MAGIC ### DB CONNECT TO SF 

# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "SANDBOX",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### ACCOUNT RISK WRITE OFF 

# COMMAND ----------

df_ar_wo = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option("query"
        , """select 
account_no
,account_write_off_amount
, invoice_write_off_amount
, account_alt_source_id
, account_source_id
,to_date(account_write_off_month || '-01') as account_write_off_month
from prod_account_risk.modelled.f_account_risk_monthly_snapshot
where account_write_off_month is not null
qualify (row_number()over(partition by account_alt_source_id order by d_snapshot_date_key desc) )=1
""")
    
).load()


# COMMAND ----------

display(df_ar_wo.count())

# COMMAND ----------

df_bill_t = spark.read.format("delta").load(dir_bill_t).filter((f.col('_IS_LATEST') ==1) & (f.col('_IS_DELETED') ==0))
df_item_t = spark.read.format("delta").load(dir_item_t).filter((f.col('_IS_LATEST') ==1) & (f.col('_IS_DELETED') ==0))
# display(df_bill_t.count()) # 462,288,252 
display(df_item_t.count()) # 38,794,942
df_accnt_t = spark.read.format("delta").load(dir_acct).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### COMPARE BILLT HISTORY 

# COMMAND ----------

display(
    df_bill_t
    .select( f.from_utc_timestamp(from_unixtime("DUE_T"),"Pacific/Auckland").alias('DUE_T')
            ,'invoice_obj_id0'
            , 'account_obj_id0')
    .filter(f.col('DUE_T') >='2023-01-01')
    .filter(f.col('invoice_obj_id0')!=0)
    .withColumn('DUE_MONTH',date_format("DUE_T","yyyy-MM")  )
    .groupBy('DUE_MONTH')
    .agg(f.countDistinct('account_obj_id0')
         , f.countDistinct('invoice_obj_id0'))
)

# COMMAND ----------

display(df_item_t.limit(100))
display(df_item_t.count()) # 36,595,489

# COMMAND ----------

# MAGIC %md
# MAGIC ###BILL T TRANSFORMATION

# COMMAND ----------

display(
df_accnt_t
#.select('poid_id0', 'crea')
.filter(f.col('account_no')=='458779374')
)

# COMMAND ----------

display(df_bill_t
.select( "ACCOUNT_OBJ_ID0"
        , "BILL_NO"
        , "INVOICE_OBJ_ID0"
        ,f.from_utc_timestamp(from_unixtime("MOD_T"),"Pacific/Auckland").alias('MOD_T')
        ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_T')
        ,f.from_utc_timestamp(from_unixtime("START_T"),"Pacific/Auckland").alias('START_T')
        ,f.from_utc_timestamp(from_unixtime("END_T"),"Pacific/Auckland").alias('END_T')
        ,f.from_utc_timestamp(from_unixtime("DUE_T"),"Pacific/Auckland").alias('DUE_T')
        ,f.from_utc_timestamp(from_unixtime("CLOSED_T"),"Pacific/Auckland").alias('CLOSED_T')
        , "CURRENCY"
        , "CURRENT_TOTAL"
        , "PREVIOUS_TOTAL"
        , "SUBORDS_TOTAL"
        , "TOTAL_DUE"
        , "ADJUSTED"
        , "DISPUTED"
        , "DUE"
        , "RECVD"
        , "WRITEOFF"
        , "TRANSFERRED"
        , "BILLINFO_OBJ_ID0"
        )
    .distinct()
    .filter(f.col("_IS_LATEST") ==1)
    .filter(f.col("INVOICE_OBJ_ID0") != 0 )
    .filter(f.col("ACCOUNT_OBJ_ID0") == '1720163424038')
    .withColumn("True_Receive_At_Current_Bill",  lag("previous_total",-1,0).over(
        Window.partitionBy("account_obj_id0").orderBy(f.col("due_t").asc()) 
        ) - f.col("TOTAL_DUE")
                )
    .withColumn("late_payment_days", datediff('CLOSED_T','DUE_T')) # if -ve means earlier 
)

# .filter(f.col("ACCOUNT_OBJ_ID0") == "1694152720824")
# current utc, need to convet to nzt 

# COMMAND ----------

df_oa_consumer = spark.read.format("delta").load(dir_oa_consumer)
df_acct_t = spark.read.format('delta').load(dir_acct).filter(f.col('_is_latest') ==1).filter(f.col('_is_deleted')==0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### BILL_T VS OA CONSUMER

# COMMAND ----------

display(df_oa_consumer
        .select('fs_cust_id', 'fs_acct_id')
        .filter(f.col('reporting_date')>='2021-01-01')
        .distinct()
        .count()
)
#   746K ACCOUNT SINCE 2021 

# COMMAND ----------

display(df_acct_t
.withColumn('account_no', regexp_replace("account_no", "^0+", ""))
.select('poid_id0', 'account_no'
        ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_T')
         )
        .filter(f.col('account_no') =='144435')
        .limit(100))
# account_no start with 'S%' is subscription account 

# COMMAND ----------

df_bill_sub_t = df_bill_t.select('account_obj_id0').filter(f.col('_IS_LATEST') ==1).distinct()

df_acct_sub_t = (df_acct_t
.withColumn('account_no', regexp_replace("account_no", "^0+", "")) # remove leading 0 
.select('poid_id0', 'account_no'
        ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_T')
         )
        )

df_oa_consumer_sub = (df_oa_consumer
        .select('fs_cust_id', 'fs_acct_id')
        .filter(f.col('reporting_date')>='2021-01-01')
        .distinct())

# check account t join with bill t - missing records seems to be prepay 
display(df_acct_sub_t
    .alias('a')
    .join(df_bill_sub_t.alias('b'), f.col('a.poid_id0') == f.col('b.account_obj_id0'), 'left')
    .filter(f.col('account_obj_id0').isNull())
    .withColumn("CREATED_MONTH", date_format("CREATED_T","yyyy-MM") )
    .groupBy("CREATED_MONTH")
    .agg(f.count("account_no"))
 ) 
 
 # anti join rate 
 # 316,053/ 24,799,681 = 1.2% 

# check account_no for anti-join set = 405944180, 405912990,405864505, all account product belong to prepay 

# display(df_acct_sub_t.count()) # 24799681


# COMMAND ----------

df_bill_acct = (df_bill_sub_t.alias('a')
.join(df_acct_sub_t.alias('b'), f.col('a.account_obj_id0') == f.col('b.poid_id0'),'inner' )
)

display(df_bill_acct.count()) # 24,486,453

# COMMAND ----------

# check oa consumer matching rate with brm 
display(
    df_oa_consumer_sub.alias('a')
    .join(df_bill_acct.alias('b'), f.col('a.fs_acct_id') == f.col('b.account_no'),'left')
    .filter(f.col('account_no').isNull())
    .limit(100)
    # .count()
)

 # need to strip leading zero for account_t.account_no 
 # only 5 records missing from bill_t joining oa consumer unit base as of run date,  not in bill_t but its account_no existing in account_t 
 # missing records has been fixed 

# COMMAND ----------

# MAGIC %md
# MAGIC ### BILL_T VOLUME CHECK ESTIMATE

# COMMAND ----------

display(df_bill_t
.select( "ACCOUNT_OBJ_ID0"
        , "BILL_NO"
        , "INVOICE_OBJ_ID0"
        ,f.from_utc_timestamp(from_unixtime("DUE_T"),"Pacific/Auckland").alias('DUE_T'))
        .withColumn("DUE_MONTH", date_format("DUE_T","yyyy-MM") )
        .distinct()
    .filter(f.col("INVOICE_OBJ_ID0") != 0 )
    .filter(f.col('DUE_MONTH')>='2021-01')
    .groupBy('DUE_MONTH')
    .agg(f.count('BILL_NO'))
)


# COMMAND ----------

# MAGIC %md
# MAGIC ###ITEM_T FOR PAYMENT

# COMMAND ----------

# check history for item_t created t 

display(
    df_item_t
    .filter(f.col('poid_type')=='/item/payment')
    .select( f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_T')
            ,'poid_id0'
            , 'account_obj_id0')
    .filter(f.col('CREATED_T') >='2023-01-01')
    .withColumn('CREATED_MONTH',date_format("CREATED_T","yyyy-MM")  )
    .groupBy('CREATED_MONTH')
    .agg(f.countDistinct('account_obj_id0')
         , f.countDistinct('poid_id0'))
)

# COMMAND ----------

display(df_item_t.limit(100))

# COMMAND ----------

display(df_item_t
.select( 'ACCOUNT_OBJ_ID0'
        ,'POID_TYPE'
        ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('PAYMENT_DATE')
        , 'ITEM_TOTAL'
)
.filter(f.col('poid_type').rlike("(?i)/item/payment") )
.filter(f.col('account_obj_id0') =='1694152720824') 
)

# does payment reversal means fail transaction?  --1904457475429

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

display(df_item_t
.select('POID_ID0'
        , 'POID_TYPE'
        , f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE')
        , f.from_utc_timestamp(from_unixtime("MOD_T"),"Pacific/Auckland").alias('MOD_T')
        , 'ACCOUNT_OBJ_ID0'
        , 'ITEM_NO'
        , 'ITEM_TOTAL'
        , 'TRANSFERED'
        , 'BILLINFO_OBJ_ID0' 
    )
    .filter((f.col('POID_TYPE') =='/item/writeoff') | (f.col('POID_TYPE') =='/item/writeoff_reversal') )
    .filter(f.col('account_obj_id0') =='466233488120')
    .limit(100)
    )

# 1749983116739

# 1863371348702


# COMMAND ----------

# MAGIC %md
# MAGIC ### VENN   

# COMMAND ----------

display(df_item_t
        .select('account_obj_id0'
                , 'poid_type'
                ,'item_total'
                ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE'))
        .filter(f.col('poid_type') =='/item/writeoff')
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.col('created_date').asc()) ))
        .filter(f.col('rnk') ==1)
        .count()
        ) # 101241  for item_t wo 

#account risk account  
print('-------------------next display-----------------')

display(df_ar_wo
        .distinct()
        .count()
        ) # 95312

print('---------------------next display----------------')

display(
df_item_t
        .select('account_obj_id0'
                , 'poid_type'
                ,'item_total'
                ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE'))
        .filter(f.col('poid_type') =='/item/writeoff')
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.col('created_date').asc()) ))
        .filter(f.col('rnk') ==1)
        .join(df_ar_wo, f.col('account_obj_id0') == f.col('account_source_id'), 'anti')
        .limit(100)
)


# COMMAND ----------

# load full base of item_t filter on write off type
dir_item_t_full = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T'
df_item_wo = (spark.read.format("delta").load(dir_item_t_full)
.filter((f.col('_IS_LATEST') ==1) & (f.col('_IS_DELETED') ==0))
.filter(f.col('poid_type') =='/item/writeoff')
)

display(df_item_wo.count())
# 1,838,177

# COMMAND ----------

# in Item_T (full) but not in AR 
display(
df_item_wo
        .select('account_obj_id0'
                , 'poid_type'
                ,'item_total'
                ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE'))
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.asc('CREATED_DATE'))))
        .filter(f.col('rnk')==1)
        # .count()
       .join(df_ar_wo,f.col('account_obj_id0') == f.col('account_source_id'), 'anti')
)

# 581 in item_t but not in AR - majority are created in late 2023 and early 2024
# item_t is more accurate - all these account have write off according to source BRM collection 
# item_t write off count = 470,835

# COMMAND ----------

# in AR but not in item_t - only 3 records and these 3 records ARE NOT IN SOURCE BRM 
display(
df_item_wo
        .select('account_obj_id0'
                , 'poid_type'
                ,'item_total'
                ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE'))
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.asc('CREATED_DATE'))))
        .filter(f.col('rnk')==1)
        .join(df_ar_wo,f.col('account_obj_id0') == f.col('account_source_id'), 'right')
        .filter(f.col('account_obj_id0').isNull())
)



# COMMAND ----------

# MAGIC %md
# MAGIC ### CHECK ITEM WO VS. ACCOUNT RISK WO

# COMMAND ----------

## common pool 
display(
df_item_wo
        .select('account_obj_id0'
                , 'poid_type'
                ,'item_total'
                ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE'))
        .filter(f.col('poid_type') =='/item/writeoff')
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.col('created_date').asc()) ))
        .filter(f.col('rnk') ==1)
        .join(df_ar_wo, f.col('account_obj_id0') == f.col('account_source_id'), 'inner')
        .withColumn('amount_dff', f.col('item_total')-f.col('account_write_off_amount'))
        .count()
       #  .filter(f.col('amount_dff') != 0 )
)

# cnt for common pool = 470,254 
# 27 out of 470,254 not match = less than 0.1% 

# COMMAND ----------

# MAGIC %md
# MAGIC ### CHECK PAYMENT DATE/AMOUNT VS. WRITE OFF REPORT 

# COMMAND ----------

dir_coll_rpt = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Coll_kpi_03032024221357.csv'
df_coll_rpt = spark.read.csv(dir_coll_rpt,header = True)

# COMMAND ----------


df_coll_rpt = (df_coll_rpt.
        select('Account Ref No','Last Payment Date'
               , 'Last Payment Amount' )
        .join(
            (df_acct_t
             .select('poid_id0', 'account_no'
        ,f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_T')
         ))  
            , f.col('Account Ref No') == f.col('account_no'),'inner')
        )
    #401,531
display(df_coll_rpt.select('Account Ref No').distinct().count()) # 395423

# COMMAND ----------

# load full base of payment date 
dir_item_t_full = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T'
df_item_pay = (spark.read.format("delta").load(dir_item_t_full)
.filter((f.col('_IS_LATEST') ==1) & (f.col('_IS_DELETED') ==0))
.filter(f.col('poid_type') =='/item/payment')
)

display(df_item_pay.count()) # 110,133,820


# COMMAND ----------

# get last payment date and amount 

display(df_item_pay
        .select(
         f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE')
        , f.from_utc_timestamp(from_unixtime("MOD_T"),"Pacific/Auckland").alias('MOD_T')
        , 'ACCOUNT_OBJ_ID0'
        , 'ITEM_TOTAL')
        .withColumn('created_time', date_format('CREATED_DATE', "HH:mm:ss")) # get first entry of the last record's day if multiple entry happen on same date
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.desc('CREATED_DATE'),f.asc('created_time'))))
        .filter(f.col('rnk')==1)
        .join(df_coll_rpt, f.col('account_obj_id0')==f.col('poid_id0'),'inner')
        .select('account_obj_id0')
        .distinct()
        .count()
        #.withColumn('amount_dff', f.col('item_total') - f.col('Last Payment Amount'))
        #.filter(f.col('amount_dff') != 0)
        )
        # 2735  out of 390416 ? -- check 10 cases, all due to timing differnece, somehow some times the report capture the second to the latest payment date for this 2.7K scenario 
        # but collection report has 395,423 distinct account no?  
        # inner join distinct account 384396  ~ 2735/384396 ~ 0.7% 


# COMMAND ----------

display(df_item_pay
        .select(
         f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE')
        , f.from_utc_timestamp(from_unixtime("MOD_T"),"Pacific/Auckland").alias('MOD_T')
        , 'ACCOUNT_OBJ_ID0'
        , 'ITEM_TOTAL')
        .withColumn('created_time', date_format('CREATED_DATE', "HH:mm:ss")) # get first entry of the last record's day if multiple entry happen on same date
        .withColumn('rnk', f.row_number().over(Window.partitionBy('account_obj_id0').orderBy(f.desc('CREATED_DATE'),f.asc('created_time'))))
        .filter(f.col('rnk')==1)
        .join(df_coll_rpt, f.col('account_obj_id0')==f.col('poid_id0'),'right')
        .filter(f.col('account_obj_id0').isNull())
        # .filter(f.col('Last Payment Date').isNotNull()) # get rid of never payer  --> return 0 
        .select('Account Ref No')
        .distinct()
        .count()
        )

        # 11115 rows with distinct acct 11027,  11,027 out of 395,423 account in collection report ~ 2.7% for never payer 

# COMMAND ----------

# MAGIC %md
# MAGIC ### CHECK PAYMENT vs. OA CONSUMER 

# COMMAND ----------


display(df_oa_consumer_sub # 746521
        .join(df_acct_sub_t,f.col('fs_acct_id') ==f.col('account_no'),'inner' )
        .join((df_item_pay.select('ACCOUNT_OBJ_ID0').distinct()), 
              f.col('poid_id0') == f.col('ACCOUNT_OBJ_ID0'), 'anti' )
        )  # 26394  out of 746521 ~ 3.5% ? 

# this also include new account that has not paid yet 
# jan 24 acct open - 1490
# feb 24 acct open - 4747 
# remove new acct  (26394 -1490-4747 )/ 746521 ~ 2.7% 


# COMMAND ----------

# MAGIC %md
# MAGIC ### BRM EVENT BILLING PAYMENT T CHECK

# COMMAND ----------

display(
    dbutils.fs.ls("/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T/"))
  
  

# COMMAND ----------

dir_payment_t = "/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T"
dir_event_bal_impact_t = '/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T'
df_payment_t = spark.read.format('delta').load(dir_payment_t).filter(f.col('_is_latest') ==1).filter(f.col('_is_deleted') ==0)
df_bal_impact = spark.read.format('delta').load(dir_event_bal_impact_t).filter(f.col('_is_latest') ==1).filter(f.col('_is_deleted') ==0)

# COMMAND ----------

display(df_payment_t.count()) # 110146559 

print('oooooooooooooooooNEXT LINE oooooooooooooooooooo')
display(df_bal_impact.count()) # 3,919,766,185

# COMMAND ----------

# MAGIC %md
# MAGIC ### PAYMENT TRANSFOMATION

# COMMAND ----------

display(df_item_t.alias('a')
        .select('poid_id0'
                ,'poid_type'
                , f.from_utc_timestamp(from_unixtime("CREATED_T"),"Pacific/Auckland").alias('CREATED_DATE')
                , 'account_obj_id0'
                , 'ITEM_TOTAL'
                )
        .filter(f.col('poid_type') =='/item/payment')
        .join(df_bal_impact.alias('b'), f.col('a.poid_id0') ==f.col('b.item_obj_id0'),'inner')
        .join(df_payment_t.alias('c'), f.col('b.obj_id0') == f.col('c.obj_id0'),'inner')
        .select('poid_id0', 'a.account_obj_id0'
                ,'CREATED_DATE'
                , 'a.ITEM_TOTAL'
                , 'trans_id'
                , 'pay_type'
                , 'channel_id'
                , 'sub_trans_id'
                )
        .filter(f.col('account_no')=='492187222')
        )

# can be done us




