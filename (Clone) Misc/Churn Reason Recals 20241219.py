# Databricks notebook source
import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window
from typing import Optional, List, Dict, Tuple

# COMMAND ----------

# snowflake connector
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "RAW",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

query_sfmc_email = """ 
    select * from PROD_MAR_TECH.SERVING.SFMC_EMAIL_PERFORMANCE
    where campaignname in (
       '240827-RM-FIX-Converged-Discount-Removal-Email-Scale 2210'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale 2210-Queued'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort B-0210'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort B'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Oreo Customers'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 8-11'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale 3110'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 13-16-18'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 26-28-29'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort A-0210'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Cohort A'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale Ineligible DOM 20-22-24 (v2)'
        ,'240827-RM-FIX-Converged-Discount-Removal-Email-Scale 1710'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Three Day Two'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Four'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Three_Email_Washup'
        , '240501-DH-MOBPM-Project_Oreo_Batch_Five V2'
        ,  '240501-DH-MOBPM-Project_Oreo_Batch_Six'
    --  ,'240501-DH-MOBPM-Project_Oreo-WashUp'
    --  ,'240501-DH-MOBPM-Project_Oreo_Batch_Six' 
    --    ,'240501-DH-MOBPM-Project_Oreo_Batch_Five'
    --    ,'240501-DH-MOBPM-Project_Oreo_Batch_Three_Email_Washup'
    --    , '240501-DH-MOBPM-Project_Oreo_Batch_Four'
    --    ,'240501-DH-MOBPM-Project_Oreo_Batch_Three Day Two'
    --    ,'240501-DH-MOBPM-Project_Oreo_Batch_Three'
    --    ,'240501-DH-MOBPM-Project_Oreo_Send_Version_Three'
    --    ,'240501-DH-MOBPM-Project_Oreo_Send'
    )   
""" 

# COMMAND ----------

# query_converged = """
#     select * from LAB_ML_STORE.SANDBOX.SC_ONE_OFF_2D_FS_MOBILE_OA_CONSUMER_EXT_CONVERGED
# """

query_bb_base = """
select 
d_snapshot_date_key 
, TO_DATE(TO_VARCHAR(d_snapshot_date_key), 'YYYYMMDD') as bb_reporting_date
,s.service_id as bb_fs_serv_id
, billing_account_number as bb_fs_acct_id
, c.customer_source_id as bb_fs_cust_id
, service_access_type_name
, s.proposition_product_name
, s.plan_name
, s.broadband_discount_oa_msisdn as converged_oa_msisdn
from prod_pdb_masked.modelled.f_service_activity_daily_snapshot f 
inner join prod_pdb_masked.modelled.d_service_curr s on f.service_source_id = s.service_source_id 
and s.current_record_ind = 1 
inner join prod_pdb_masked.modelled.d_billing_account_curr b on b.billing_account_source_id = s.billing_account_source_id 
and b.current_record_ind = 1
inner join prod_pdb_masked.modelled.d_customer_curr c on c.customer_source_id = b.customer_source_id
and c.current_record_ind = 1
where  
s.service_type_name in ('Broadband')
and 
( 
 (service_access_type_name is not null)
   
or 
 ( s.service_type_name in ('Broadband')
   and service_access_type_name is null   
   and s.proposition_product_name in ('home phone wireless broadband discount proposition'
   , 'home phone plus broadband discount proposition')
   )
 )
and c.market_segment_name in ('Consumer')
and f.d_snapshot_date_key in ('20230423','20240505','20240303','20241201','20230326','20240519','20240204','20240602','20240721','20240107','20240121',
'20231203','20240331','20231126','20240630','20241006','20241013','20231105','20231210','20231217','20240128','20241110','20240714','20240324','20240414','20231231','20240616','20230618','20231008','20240512','20231224','20240804','20240211','20240623','20240929','20240310','20241103','20240901','20240407',
'20240114','20241117','20240317','20240818','20240609','20230813','20240922','20231119','20241208','20240811','20240908','20230226','20231112','20240225',
'20230521','20230101','20241124','20240526','20240218','20241027','20240825','20240707','20240728','20241020','20230716','20240421','20230129','20240428',
'20230910','20240915' , '20241215', '20241222', '20241229', '20250105', '20250112'); 

"""
# since 2023-01

# COMMAND ----------

query_sfmc_sms = """
    select * from PROD_MAR_TECH.SERVING.SFMC_ON_NET_SMS_MESSAGE 
    where sms_name ilike '%converged%'
"""

# COMMAND ----------

query_port_base = """
    SELECT 
        * 
        , case when gaining_service_provider_name in ('2degrees Mobile', '2degrees NextGen Mobile', '2degrees NextGen Broadband') then '2degrees' 
            when gaining_service_provider_name in ('Skinny Mobile') then 'Skinny'
            when gaining_service_provider_name in ('Spark Mobile') then 'Spark'
            when gaining_service_provider_name in ('Mercury Mobile') then 'Mercury'
            else 'Other' end as gaining_service_provider_name_grp
    FROM LAB_ML_STORE.SANDBOX.SC_ONE_OFF_2D_PORT_OUT_BASE;
"""
# from edw to snowflake script 

# COMMAND ----------

# directly from EDW, no transformation 
query_port_out = """
    select * from LAB_ML_STORE.SANDBOX.SC_ONE_OFF_2D_PORT_OUT;
"""

# COMMAND ----------

query_wallet_offer = """
    Select *
    from PROD_WALLET.SERVING.DS_YT_WALLET_DAILY_SS
    where
        1 = 1
        --and (
           -- BALANCE_EARNED > 0
            ---OR BALANCE_EXPIRED > 0
      --  )
    --and d_date_key = 20241126
    and campaign_type <> 'Trade-in';

"""

# COMMAND ----------

# DBTITLE 1,load data
df_campaign_list_e = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_sfmc_email
    )
    .load()
)

df_campaign_list_s = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_sfmc_sms
    )
    .load()
)

df_port_base = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_port_base
    )
    .load()
)


# df_port_output_v2 = (
#     spark.read.format("snowflake")
#     .options(**options)
#     .option(
#         "query"
#         , query_port_out
#     )
#     .load()
# )


# df_convged_base = (
#     spark.read.format("snowflake")
#     .options(**options)
#     .option(
#         "query"
#         , query_converged
#     )
#     .load()
# )

df_bb_base = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_bb_base
    )
    .load()
)

df_wallet_offer = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_wallet_offer
    )
    .load()
)



# unit base
df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=rolling cycle')

# mvnt base 

df_srvc_mvnt = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation/reporting_cycle_type=rolling cycle')



# COMMAND ----------

display(df_wallet_offer.limit(10))

# COMMAND ----------

display(
    df_port_base
    .limit(100)
)

# COMMAND ----------

# from discount removal - around 50% of them transfer to port out reasons, another 50% transfer to other reasons 
display(
        df_srvc_mvnt
        .select('fs_acct_id', 'fs_srvc_id', 'fs_cust_id', 'movement_date', 'deactivate_reason', 'reporting_date')
        .distinct()
        #.filter(f.col('movement_date').between( '2024-02-01', '2024-01-31'))
        .groupBy('reporting_date', 'deactivate_reason')
        .agg(
                f.countDistinct('fs_srvc_id')
        )
)

# COMMAND ----------

display(df_port_base.limit(10))

# COMMAND ----------

display(
    df_port_base
    .groupBy('date_key')
    .agg(f.count('FS_SRVC_ID'))
)    


# COMMAND ----------

df_comm_base_agg = (
    df_campaign_list_e
    .select('customer_id', 'EVENTDATE', 'EMAILNAME')
    .distinct()
    .union(
        df_campaign_list_s
        .select('customer_id', 'SEND_DATE', 'SMS_NAME')
    )
    .distinct()
    .withColumnRenamed('customer_id', 'fs_cust_id')
    .groupBy('fs_cust_id')
    .agg(f.countDistinct('EMAILNAME').alias('comm_cnt')
         , f.min('eventdate').alias('min_event_date')
         , f.max('eventdate').alias('max_event_date')
    )
)

# COMMAND ----------

display(
    df_comm_base_agg
    .groupBy(f.date_format('min_event_date', 'yyyy-MM-dd'))
    .agg(
         f.count('*')
        , f.countDistinct('fs_cust_id')
    )
)

# COMMAND ----------

display(df_fea_unitbase
        .filter(f.col('reporting_date') >='2023-01-01')
        .select('reporting_date', 'reporting_cycle_type')
        .withColumn('reporting_date_format', f.date_format('reporting_date', 'yyyyMMdd'))
        .distinct()
)

# COMMAND ----------

display(
    df_port_base
    .groupBy('reporting_date')
    .agg(f.countDistinct('fs_srvc_id'))    
)

# COMMAND ----------

display(
    df_fea_unitbase
    .filter(f.col('reporting_date') >='2022-01-01')
    .groupBy('reporting_date')
    .agg(f.countDistinct('fs_srvc_id'))
)

# COMMAND ----------

display(
    df_comm_base_agg
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 base dev

# COMMAND ----------

# DBTITLE 1,converged base
df_convged_base = (
    df_fea_unitbase
    .select('fs_cust_id', 'fs_acct_id', 'fs_srvc_id', 'reporting_date', 'reporting_cycle_type')
    .filter(f.col('reporting_date') >='2024-01-01')
    .distinct()
    .join(df_bb_base
          .select('bb_reporting_date', 'bb_fs_acct_id')
          .distinct()
          , 
          (f.col('fs_acct_id') == f.col('bb_fs_acct_id') ) &
          (f.col('reporting_date') == f.col('bb_reporting_date'))
          , 'left')
    .withColumn(
        'product_holding'
        , f.when(
            f.col('bb_fs_acct_id').isNotNull()
            , 'OA+BB'
        )
        .otherwise('OA only')
    )
    #.limit(10)
    # .groupBy('reporting_date', 'product_holding')
    # .agg(f.countDistinct('fs_acct_id'))    
)
        

# COMMAND ----------

df_wallet_offer_daily_agg = (
    df_wallet_offer
    .select('d_date_key', 'WALLET_ID', 'BALANCE_EARNED', 'CAMPAIGN_ID', 'CAMPAIGN_START_DATE_TIME', 'BILLING_ACCOUNT_NUMBER', 'CAMPAIGN_NAME')
    .groupBy('d_date_key', 'WALLET_ID', 'BILLING_ACCOUNT_NUMBER')
    .agg(
        f.sum('BALANCE_EARNED').alias('SUM_WALLENT_BALANCE')
        , f.countDistinct('CAMPAIGN_ID').alias('CNT_WALLET_CAMPAIGN')
        , f.min('CAMPAIGN_START_DATE_TIME').alias('min_campaign_date')
        , f.max('CAMPAIGN_START_DATE_TIME').alias('max_campaign_date')
        #, f.collect_list('CAMPAIGN_NAME').alias('CAMPAIGN_NAME_LIST')
    )
    .withColumn('wallet_date_key', f.to_date(f.col("d_date_key").cast("string"), "yyyyMMdd") )
    .filter(f.col('wallet_date_key') >= '2024-01-01')
    .drop('d_date_key')
)

# COMMAND ----------

# DBTITLE 1,active base for comms not comms
df_active_base = (
    df_fea_unitbase
    .filter(f.col('reporting_date') >='2024-01-01')
    .join(
        df_port_base
        #.drop('reporting_date', 'reporting_cycle_type')
        .withColumn('target_reporting_date', f.next_day('target_date', 'Sunday'))
        , ['fs_acct_id', 'fs_srvc_id', 'reporting_date']
        , 'anti'
    ) # last snapshot before churn  # remvoe churn 
    .join(
        df_comm_base_agg
        , ['fs_cust_id']
        , 'left'
    )
    .withColumn(
        'discount_removal'
        , f.when(
            f.col('min_event_date').isNull()
            , f.lit(0)
        )
        .otherwise(f.lit(1))
    ) # flag for discount removal 
    .join(
        df_convged_base
        , ['reporting_date', 'fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
        , 'left'
    )
    .join(
        df_wallet_offer_daily_agg
        , (f.col('fs_srvc_id') == f.col('WALLET_ID'))
        & (f.col('reporting_date') == f.col('wallet_date_key'))
        , 'left'
    )
    .groupBy('reporting_date', 'discount_removal', 'product_holding')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*') 
        , f.countDistinct('wallet_id')
        , f.sum('SUM_WALLENT_BALANCE').alias('SUM_WALLENT_BALANCE')
    )
)

#display(df_active_base.limit(10))

# COMMAND ----------

display(df_active_base)

# COMMAND ----------

df_port_output =(
    df_port_base
    .withColumn('target_reporting_date', f.next_day('target_date', 'Sunday'))
    .join(df_comm_base_agg, ['fs_cust_id'], 'left')
    .withColumn(
        'discount_removal'
        , f.when(
            f.col('min_event_date').isNull()
            , f.lit(0)
        )
        .otherwise(f.lit(1))
    )
    .join(
        df_convged_base
        , ['reporting_date', 'fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
        , 'left'
    )
)

# COMMAND ----------

display(
    df_port_output
    .filter(f.col('target_date') >='2024-01-01')
    .drop('reporting_cycle_type')
    .groupBy('target_reporting_date', 'product_holding', 'discount_removal')
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.countDistinct('FS_ACCT_ID')
    )
    #.limit(100)
)

# COMMAND ----------

display(df_port_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### export to snowflake 

# COMMAND ----------

(
    df_port_output
    .write
    .format("snowflake")
    .options(**options)
    .option("dbtable", "")
    .mode("append")
    .save()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 

# COMMAND ----------

df_output_curr = (
    df_port_output
    .filter(f.col('target_date') >='2024-01-01')
    .join(
        df_wallet_offer_daily_agg
        .filter(f.col('wallet_date_key') >= '2024-01-01')
        , (f.col('fs_srvc_id') == f.col('WALLET_ID'))
            & (f.col('target_date') == f.col('wallet_date_key')) 
        , 'left' 
    )
    .drop('reporting_cycle_type')
    #.count()
)

# COMMAND ----------

display(
    df_output_curr
    .agg(
        f.countDistinct('FS_CUST_ID')
        , f.count('*')
    )
)

# COMMAND ----------

(
    df_output_curr
    .write
    .mode('append')
    .format('delta')
    .option('header', True)
    .save('/mnt/ml-lab/dev_users/dev_sc/adhoc/2d_churn_20250114')
)

# COMMAND ----------

#df_output_curr = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/adhoc/2d_churn_20241219')
df_output_curr = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/adhoc/2d_churn_20250114')

# COMMAND ----------

display(
    df_output_curr
    .count()
)

# COMMAND ----------

display(df_output_curr)

# COMMAND ----------

display(df_active_base)
