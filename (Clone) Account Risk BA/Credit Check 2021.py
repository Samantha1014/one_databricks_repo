# Databricks notebook source
# MAGIC %md
# MAGIC ### Library

# COMMAND ----------

display(
    dbutils.fs.ls("/mnt/prod_brm/raw/cdc/"))

# COMMAND ----------

import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession  
from pyspark.sql.functions import when

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connectivity 

# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "PROD_INTERFLOW",
  "sfSchema": "RAW",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

# DBTITLE 1,Load Interflow
df_credit_score = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option("query"
        , """
            select * from LAB_ML_STORE.SANDBOX.INTERFLOW_CREDIT_REPORT_2018_2021
            union 
            select * from lab_ml_store.sandbox.interflow_credit_report_202101_202402
            union 
            select * from lab_ml_store.sandbox.interflow_credit_report_2015_2018
             """)
    
).load()



# COMMAND ----------

# DBTITLE 1,Load Siebel Profile
df_siebel_profile = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option("query"
        , "select * from PROD_SIEBEL.RAW.SBL_FINAN_PROF where _is_latest = 1 and _is_deleted = 0;")  
).load()



# COMMAND ----------

# DBTITLE 1,Load Profile Hist
df_aws_siebel_profile_hist = (
    spark.read.format('snowflake').options(**options)
    .option("query", "select * from PROD_AWS_PROD_MASKED.STAGE_PERM.DS_SIEBEL_S_FINAN_PROF_HIST").load()
)



# COMMAND ----------

# DBTITLE 1,Load Consumer Base
dir_oa_base = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_base = spark.read.format('delta').load(dir_oa_base)

# COMMAND ----------

# DBTITLE 1,Convert to Temp view
df_credit_score.createOrReplaceTempView("df_credit_score")
df_aws_siebel_profile_hist.createOrReplaceTempView('df_aws_siebel_profile_hist')
df_siebel_profile.createOrReplaceTempView('df_siebel_profile')
df_oa_base.createOrReplaceTempView('df_oa_base')

# COMMAND ----------

display(df_credit_score.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ###Compare Sources

# COMMAND ----------

# DBTITLE 1,Map just for interflow
# MAGIC %sql
# MAGIC with oa_base as (
# MAGIC select distinct fs_cust_id, fs_acct_id from df_oa_base 
# MAGIC where reporting_date >= '2021-01-01' 
# MAGIC --and billing_acct_open_date >='2021-01-01'
# MAGIC ) 
# MAGIC select count(distinct a.fs_cust_id) 
# MAGIC ,sum(count(distinct fs_cust_id)) over() as sum 
# MAGIC , count(distinct a.fs_cust_id)  / sum(count(distinct fs_cust_id)) over() as pct
# MAGIC , case when b.dnb_final_score is null then 1 else 0 end as miss_flg  
# MAGIC from oa_base a inner join df_credit_score b on a.fs_cust_id = b.custref 
# MAGIC group by 4;
# MAGIC
# MAGIC
# MAGIC --- missing rate 33% for new acct open since 2021 

# COMMAND ----------

# DBTITLE 1,Compare Siebel Vs. Inteflow
# MAGIC %sql
# MAGIC With interflow as (
# MAGIC SELECT DISTINCT CUSTREF,PRE_BU_SCORE,DNB_FINAL_SCORE, submit 
# MAGIC FROM df_credit_score
# MAGIC WHERE (DNB_FINAL_SCORE IS NOT NULL) and custref is not null 
# MAGIC qualify row_number()over(partition by custref order by submit desc) =1 
# MAGIC ),
# MAGIC siebel as(
# MAGIC select distinct accnt_id, x_credit_score, x_credit_req_dt 
# MAGIC --from df_aws_siebel_profile_hist 
# MAGIC from df_siebel_profile  
# MAGIC where x_credit_score is not null 
# MAGIC qualify row_number() over(partition by accnt_id order by x_credit_req_dt desc) = 1
# MAGIC ), mapping as (
# MAGIC select * , case when accnt_id is null and custref is not null then 'in inteflow not in siebel'
# MAGIC when accnt_id is not null and custref is null then 'in siebel not in interflow'
# MAGIC when accnt_id is not null and custref is not null then 'in both' end as flg 
# MAGIC from interflow full join siebel on custref= accnt_id
# MAGIC ) 
# MAGIC select count(1), flg from mapping
# MAGIC group by flg
# MAGIC
# MAGIC ;
# MAGIC
# MAGIC ---case why it only in interflow not in siebel?
# MAGIC --- 1. siebel only keep last score of the date if multiple check occurs on the same date 
# MAGIC --- case when it in siebel not in interflow? 
# MAGIC ---- 1. old history before 2021 in sieble 
# MAGIC ---- 2. from interflow source file we miss records 
# MAGIC ---- 3. duplicate profile in interflow 
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,compare credit score in common pool
# MAGIC %sql
# MAGIC With interflow as (
# MAGIC SELECT DISTINCT CUSTREF,PRE_BU_SCORE,DNB_FINAL_SCORE, submit FROM df_credit_score
# MAGIC WHERE (DNB_FINAL_SCORE IS NOT NULL) and custref is not null 
# MAGIC qualify row_number()over(partition by custref order by submit desc) =1 
# MAGIC ),
# MAGIC siebel as(
# MAGIC select distinct accnt_id, x_credit_score, x_credit_req_dt  
# MAGIC --from PROD_AWS_PROD_MASKED.STAGE_PERM.DS_SIEBEL_S_FINAN_PROF_HIST 
# MAGIC from  df_siebel_profile
# MAGIC where x_credit_score is not null 
# MAGIC qualify row_number() over(partition by accnt_id order by x_credit_req_dt desc) = 1
# MAGIC ), mapping as (
# MAGIC select * , case when accnt_id is null and custref is not null then 'in inteflow not in siebel'
# MAGIC when accnt_id is not null and custref is null then 'in siebel not in interflow'
# MAGIC when accnt_id is not null and custref is not null then 'in both' end as flg 
# MAGIC from interflow full join siebel on custref= accnt_id
# MAGIC )
# MAGIC select *, (dnb_final_score - x_credit_score) as diff  from mapping where flg = 'in both'
# MAGIC and (dnb_final_score - x_credit_score) != 0;
# MAGIC
# MAGIC --- only 36 records difference -- acceptable 

# COMMAND ----------

# DBTITLE 1,combine credit score
# MAGIC %sql
# MAGIC With interflow as (
# MAGIC SELECT DISTINCT CUSTREF,PRE_BU_SCORE,DNB_FINAL_SCORE, submit 
# MAGIC FROM df_credit_score
# MAGIC WHERE (DNB_FINAL_SCORE IS NOT NULL) and custref is not null 
# MAGIC qualify row_number()over(partition by custref order by submit asc) =1 
# MAGIC ), siebel as(
# MAGIC select distinct accnt_id, x_credit_score, x_credit_req_dt 
# MAGIC from df_aws_siebel_profile_hist 
# MAGIC --from df_siebel_profile ---(this table doest have hist?)
# MAGIC where x_credit_score is not null 
# MAGIC qualify row_number() over(partition by accnt_id order by x_credit_req_dt asc) = 1
# MAGIC )
# MAGIC select a.CUSTREF, coalesce(a.DNB_FINAL_SCORE,b.x_credit_score ) as combine_credit_score from interflow a left join siebel b on a.CUSTREF = b.accnt_id 
# MAGIC

# COMMAND ----------

# store the output result into a dataframe 
df_combine_credit_score = _sqldf
df_combine_credit_score.createOrReplaceTempView('df_combine_credit_score')

# COMMAND ----------

# DBTITLE 1,credit score base
# MAGIC %sql
# MAGIC select count(1) from df_combine_credit_score ; 
# MAGIC --1376187

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_oa_base limit 100;

# COMMAND ----------

# DBTITLE 1,combine unit base
# MAGIC %sql
# MAGIC with oa_base as (
# MAGIC select distinct fs_cust_id, fs_acct_id 
# MAGIC from df_oa_base where reporting_date >='2021-01-01'
# MAGIC --and billing_acct_open_date >= '2021-01-01'
# MAGIC )
# MAGIC select 
# MAGIC count( distinct fs_cust_id ) as cnt
# MAGIC , sum(count(distinct fs_cust_id)) over() as total_cust
# MAGIC , count( distinct fs_cust_id ) / sum(count(distinct fs_cust_id)) over() as pct
# MAGIC , case when b.custref is null then 1 else 0 end as miss_flg
# MAGIC  from oa_base a
# MAGIC left join df_combine_credit_score b 
# MAGIC on a.fs_cust_id = b.custref
# MAGIC group by 4;
# MAGIC
# MAGIC
# MAGIC --746521 for total active oa consumer since 2021 
# MAGIC --343165 for customer bill start on jan 2021 
# MAGIC ---26885 / 343165 = 7.8% missing rate for new customers if they bill account started after 2021-01-01 
# MAGIC
# MAGIC ---0.37 missing rate for 2018 till 2024

# COMMAND ----------

# DBTITLE 1,distribution of missing credit score
# MAGIC %sql
# MAGIC with oa_base as (
# MAGIC select distinct fs_cust_id, fs_acct_id, date_trunc('YEAR', cust_start_date) as open_year
# MAGIC from df_oa_base where reporting_date >='2021-01-01'
# MAGIC --and billing_acct_open_date >= '2021-01-01'
# MAGIC qualify row_number() over(partition by fs_cust_id, fs_acct_id order by billing_acct_open_date asc) =1 
# MAGIC )
# MAGIC select 
# MAGIC count(distinct fs_cust_id), open_year from oa_base a
# MAGIC left anti join df_combine_credit_score b 
# MAGIC on a.fs_cust_id = b.custref
# MAGIC group by 2;
# MAGIC
# MAGIC -- spot check a couple, there are customer who joined before 2021 
# MAGIC -- also when it mapped to a access change file -- according to sam, it is a migration? 
# MAGIC -- e.g. fs_cust_id = 1-100ALN5C or fs_cust_id = 1-100BXGJP 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Siebel Financial Profile Missing Rate

# COMMAND ----------

# DBTITLE 1,analyse sieble financial profile missing rate
# MAGIC %sql
# MAGIC select 
# MAGIC avg(case when CRDT_CRD_EXPT_DT is null then 1 else 0 end ) as missing_CRDT_CRD_EXPT_DT
# MAGIC , avg(case when CC_NUMBER is null then 1 else 0 end ) as missing_CC_NUMBER
# MAGIC ,  avg(case when HOME_OWN_TYP_CD is null then 1 else 0 end ) as HOME_OWN_TYP_CD
# MAGIC ,  avg(case when X_ADDITIONAL_INFO is null then 1 else 0 end ) as X_ADDITIONAL_INFO
# MAGIC ,  avg(case when X_CRDT_APP_TYPE is null then 1 else 0 end ) as X_CRDT_APP_TYPE
# MAGIC ,  avg(case when X_CRDT_STAT_CD is null then 1 else 0 end ) as X_CRDT_STAT_CD
# MAGIC ,  avg(case when X_CREDIT_STATUS_INFO is null then 1 else 0 end ) as X_CREDIT_STATUS_INFO
# MAGIC ,  avg(case when X_CUR_CITY is null then 1 else 0 end ) as X_CUR_CITY
# MAGIC ,  avg(case when X_CUR_POSTCODE is null then 1 else 0 end ) as X_CUR_POSTCODE
# MAGIC ,  avg(case when X_CUR_TIMEATADDRRESS is null then 1 else 0 end ) as X_CUR_TIMEATADDRRESS
# MAGIC ,  avg(case when X_EXISTING_CUSTOMER is null then 1 else 0 end ) as X_EXISTING_CUSTOMER
# MAGIC ,  avg(case when X_FIN_POSITION is null then 1 else 0 end ) as X_FIN_POSITION
# MAGIC ,  avg(case when X_FIN_STATUS is null then 1 else 0 end ) as X_FIN_STATUS
# MAGIC ,  avg(case when X_NUM_ARREARS is null then 1 else 0 end ) as X_NUM_ARREARS
# MAGIC ,  avg(case when X_NUM_BARS is null then 1 else 0 end ) as X_NUM_BARS
# MAGIC ,  avg(case when X_NUM_CREDIT_CARDS is null then 1 else 0 end ) as X_NUM_CREDIT_CARDS
# MAGIC ,  avg(case when X_NUM_LOCATIONS is null then 1 else 0 end ) as X_NUM_LOCATIONS
# MAGIC ,  avg(case when X_NUM_PARTNERS is null then 1 else 0 end ) as X_NUM_PARTNERS
# MAGIC ,  avg(case when X_PENDING_CONNECTIONS is null then 1 else 0 end ) as X_PENDING_CONNECTIONS
# MAGIC ,  avg(case when X_REASON_CODE is null then 1 else 0 end ) as X_REASON_CODE
# MAGIC ,  avg(case when X_REASON_DESCRIPTION is null then 1 else 0 end ) as X_REASON_DESCRIPTION
# MAGIC --,  avg(case when X_TIME_AS_CUSTOMER is null then 1 else 0 end ) as X_TIME_AS_CUSTOMER --- 100% missing
# MAGIC ,  avg(case when X_DEALER_CODE is null then 1 else 0 end ) as X_DEALER_CODE
# MAGIC ,  avg(case when X_DEALER_DESC is null then 1 else 0 end ) as X_DEALER_DESC
# MAGIC from df_siebel_profile ;
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(distinct concat(last_upd, accnt_id)), count(1) from df_siebel_profile
# MAGIC where accnt_id is not null 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Rate for Credit Score Fields

# COMMAND ----------

# DBTITLE 1,Missing Rate for Credit Report
# MAGIC %sql
# MAGIC select 
# MAGIC avg(case when DOB is null then 1 else 0 end ) as DOB
# MAGIC , avg(case when RES_STATUS is null then 1 else 0 end ) as RES_STATUS
# MAGIC ,  avg(case when EMP_STATUS is null then 1 else 0 end ) as EMP_STATUS
# MAGIC ,  avg(case when HOMEPHN is null then 1 else 0 end ) as HOMEPHN
# MAGIC ,  avg(case when CUSTTIME is null then 1 else 0 end ) as CUSTTIME
# MAGIC ,  avg(case when ARREARS is null then 1 else 0 end ) as ARREARS
# MAGIC ,  avg(case when decision is null then 1 else 0 end) as decision 
# MAGIC from df_credit_score 
# MAGIC where custref is not null;

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(1), count(distinct id) from df_credit_score 
# MAGIC where id is not null 
# MAGIC and _is_latest = 1 and _is_deleted =0 ;
