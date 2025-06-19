# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 set up

# COMMAND ----------

# DBTITLE 1,Essential PySpark and Other Libraries Import
import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window
from typing import Optional, List, Dict, Tuple

# COMMAND ----------

# DBTITLE 1,Connecting to Snowflake (Credentials & Parameters)
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

# DBTITLE 1,SFMC Email Performance Query
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

# DBTITLE 1,Retrieve Converged SMS Messages from SFMC Table
query_sfmc_sms = """
    select * from PROD_MAR_TECH.SERVING.SFMC_ON_NET_SMS_MESSAGE 
    where sms_name ilike '%converged%'
"""

# COMMAND ----------

# DBTITLE 1,Port Base Query
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


# COMMAND ----------

# DBTITLE 1,get BB service details
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
'20230910','20240915' , '20241215'); 

"""
# since 2023-01

# COMMAND ----------

# DBTITLE 1,bb churned query
query_bb_churned = """
 SELECT
	SERVICE_ACCESS_TYPE_NAME ,
	SERVICE_TYPE_NAME ,
	SERVICE_ID ,
	SERVICE_LINKED_ID ,
	SERVICE_STATUS_NAME ,
	Plan_name,
	TO_char(SERVICE_DEACTIVATION_DATE_TIME, 'yyyy-mm-dd') AS SERVICE_DEACTIVATION_DATE ,
	SERVICE_DEACTIVATION_DATE_TIME,
	SERVICE_DEACTIVATION_REASON_NAME ,
	service_deactivation_reason_desc,
	c.billing_account_number,
	customer_service_full_address,
	product_holding_desc,
	a.service_start_date_time,
	c.current_balance_bill_amount,
	a.service_sales_channel_name
FROM prod_pdb_masked.MODELLED.D_SERVICE_CURR a 
LEFT JOIN prod_pdb_masked.MODELLED.D_CUSTOMER_CURR b ON
	a.customer_source_id_sha2 = b.customer_source_id_sha2
LEFT JOIN prod_pdb_masked.modelled.d_billing_account c ON
	c.billing_account_source_id = a.billing_account_source_id
WHERE
	market_segment_name = 'Consumer'
	AND 
 SERVICE_DEACTIVATION_DATE_TIME BETWEEN '2022-04-01' AND '2024-12-30'
	AND service_type_name IN ('Broadband')
	AND b.current_record_ind = 1
	AND a.CURRENT_RECORD_IND = 1
	AND c.current_record_ind = 1
    AND SERVICE_STATUS_NAME NOT IN ('Active', 'Transferred')
	AND SERVICE_ACCESS_TYPE_NAME IS NOT NULL 
	And SERVICE_DEACTIVATION_DATE_TIME IS NOT NULL ; 
"""

# COMMAND ----------

# DBTITLE 1,wallet offer query
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

df_bb_churned = (
    spark.read.format("snowflake")
    .options(**options)
    .option(
        "query"
        , query_bb_churned
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

display(
    df_srvc_mvnt
    .groupBy(f.date_format('movement_date','yyyy-MM'), 'deactivate_type')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.countDistinct('fs_srvc_id')
    )
)

# COMMAND ----------

display(
    df_bb_base
    .groupBy('BB_REPORTING_DATE')
    .agg(
        f.countDistinct('bb_fs_serv_id')
        , f.countDistinct('bb_fs_acct_id')
    )    
        
)

# COMMAND ----------

display(df_bb_churned.limit(10))

# COMMAND ----------

display(df_wallet_offer.limit(10))

# COMMAND ----------

display(
    df_campaign_list_e
    .groupBy('eventdate', 'CAMPAIGNNAME')     
    .agg(f.countDistinct('customer_id'))
)

# COMMAND ----------

# DBTITLE 1,check example
# print('converged')
# display(df_convged_base.limit(10))
print('bb_base')
display(df_bb_base.limit(10))

print('bb_churned')
display(df_bb_churned.limit(10))

print('port_base')
display(df_port_base.limit(10))

print('campaign_list_email')
display(df_campaign_list_e.limit(10))


print('campaign_list_sms')
display(df_campaign_list_s.limit(10))

# COMMAND ----------

# DBTITLE 1,combine sms and email
df_comm_base = (
    df_campaign_list_e
    .select('customer_id', 'EVENTDATE', 'EMAILNAME')
    .distinct()
    .union(
        df_campaign_list_s
        .select('customer_id', 'SEND_DATE', 'SMS_NAME')
    )
    .distinct()
    .withColumnRenamed('customer_id', 'fs_cust_id')
)

# COMMAND ----------

# DBTITLE 1,agg comm base
df_comm_base_agg = (
    df_comm_base
    .groupBy('fs_cust_id')
    .agg(f.countDistinct('EMAILNAME').alias('comm_cnt')
         , f.min('eventdate').alias('min_event_date')
         , f.max('eventdate').alias('max_event_date')
    )
)

# COMMAND ----------

# DBTITLE 1,check comm base count
display(df_comm_base_agg
        .groupBy(f.date_format('min_event_date', 'yyyy-MM-dd'))
        .agg(
            f.count('*')
            , f.countDistinct('fs_cust_id')
        )
)

# COMMAND ----------

display(df_srvc_mvnt.limit(10))

# COMMAND ----------

# DBTITLE 1,check discount removal comms base vs. overall churn
# from discount removal - around 50% of them transfer to port out reasons, another 50% transfer to other reasons 
display(
        df_srvc_mvnt
        .select('fs_acct_id', 'fs_srvc_id', 'fs_cust_id', 'movement_date', 'deactivate_reason', 'reporting_date')
        .distinct()
        #.filter(f.col('movement_date').between( '2024-02-01', '2024-01-31'))
        .groupBy('reporting_date', 'deactivate_reason', 'reporting_cycle_type')
        .agg(
                f.countDistinct('fs_srvc_id')
        )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### s02 broadband

# COMMAND ----------

# DBTITLE 1,unit base check
display(df_fea_unitbase
        .filter(f.col('reporting_date') >='2023-01-01')
        .select('reporting_date', 'reporting_cycle_type')
        .withColumn('reporting_date_format', f.date_format('reporting_date', 'yyyyMMdd'))
        .distinct()
)

# COMMAND ----------

# DBTITLE 1,bb base check
display(
        df_bb_base
        .groupBy('bb_reporting_date')
        .agg(
                f.countDistinct('bb_fs_acct_id')
                , f.countDistinct('BB_FS_SERV_ID')
        )
)

# COMMAND ----------

# DBTITLE 1,converged base = oa + bb
df_convged_base = (
    df_fea_unitbase
    .select('fs_cust_id', 'fs_acct_id', 'fs_srvc_id', 'reporting_date', 'reporting_cycle_type')
    .filter(f.col('reporting_date') >='2023-01-01')
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

display(
    df_fea_unitbase
    .filter(f.col('reporting_date') >= '2024-01-01')
    .groupBy('reporting_date')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.countDistinct('fs_srvc_id')
    )     
)

# COMMAND ----------

# DBTITLE 1,check converged base
display(
    df_convged_base
    .filter(f.col('reporting_date') >= '2024-01-01')
    .groupBy(
        'reporting_date'
        , 'product_holding'
    ) 
    .agg(
        f.countDistinct('fs_acct_id')
        , f.countDistinct('fs_srvc_id')
        , f.count('*')
    )
)

# COMMAND ----------

# DBTITLE 1,check service level bb churn
display(
    df_bb_churned
    .filter(f.col('SERVICE_DEACTIVATION_DATE_TIME') >= '2024-08-01')    
) # there are customers that transfer from wireless to fiber or vice versa 

# COMMAND ----------

# DBTITLE 1,get max report date bb
vt_max_bb_reporting_date = df_bb_base.agg(f.max("bb_reporting_date")).collect()[0][0]

# COMMAND ----------

# DBTITLE 1,check max bb date
vt_max_bb_reporting_date

# COMMAND ----------

# DBTITLE 1,derived bb acct level churn base
df_bb_churn_base = (
    df_bb_base
    .withColumn(
        'index'
        , f.row_number().over(
            Window
            .partitionBy('BB_FS_ACCT_ID')
            .orderBy(f.desc('BB_REPORTING_DATE'))
        )
    )
    .filter(f.col('index') == 1)
    .withColumn(
        'bb_churn_flag'
        , f.when(
            f.col('BB_REPORTING_DATE') < vt_max_bb_reporting_date
            , 1
        )
        .when( 
            f.col('BB_REPORTING_DATE') == vt_max_bb_reporting_date
            , 0
        )       
    )
    .filter(f.col('bb_churn_flag') == 1)
    .select('BB_REPORTING_DATE', 'BB_FS_ACCT_ID','BB_FS_CUST_ID', 'bb_churn_flag')
    .distinct()
)    

# COMMAND ----------

display(df_port_base.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### s03 overview

# COMMAND ----------

# DBTITLE 1,port base
display(
        df_port_base
        .groupBy(f.date_format('target_date', 'yyyy-MM')
                 , f.next_day('target_date', 'Sunday')
                  , 'GAINING_SERVICE_PROVIDER_NAME_GRP')
        .agg(
                f.countDistinct('fs_acct_id')
                , f.countDistinct('fs_srvc_id')
        )    
)
# overall trend since 2022 - increased proportion in sep, oct period (periodic increased porportion to 2d)
# this year we see Jun, july increase (project oreo)

# COMMAND ----------

# DBTITLE 1,port and converged
display(
    df_port_base
    .filter(f.col('reporting_date') >= '2023-01-01')
    .join(
        df_convged_base
        , ['reporting_date', 'fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
        , 'left'
    )
    .groupBy(
        'reporting_date'
        , 'GAINING_SERVICE_PROVIDER_NAME_GRP'
        , 'product_holding'
    )
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.countDistinct('fs_acct_id')
    )
)    

# COMMAND ----------

# DBTITLE 1,check converged vs. comms
display(
      df_convged_base
      .filter(f.col('product_holding') == 'OA+BB')
      .filter(f.col('reporting_date') >= '2024-06-01')
      .select('fs_cust_id', 'fs_acct_id')
      .distinct()
      .join(
            df_comm_base_agg
            , ['fs_cust_id']
            , 'anti'
      ) 
      .agg(
            f.countDistinct('fs_cust_id')
            , f.count('*')
      )
)
    # not all converged customers will be comms - some expection - the intial convreged customers does not have bundle discount 
    # wireless broadband - rural no converged discount apply initially  
    # also some errors - that not comms but have bundle disocunt and still not removal  - e.g. 399237873
    

# COMMAND ----------

# DBTITLE 1,check base and comms
display(
    df_convged_base
    .filter(f.col('reporting_date') >= '2024-09-01')
    .join(df_comm_base_agg, ['fs_cust_id'], 'inner')
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.countDistinct('fs_acct_id')
        , f.count('*')
    )    
)

# COMMAND ----------

# DBTITLE 1,Comms base join with converged to create base for churn rate calcs
display(
    df_convged_base
    .filter(f.col('reporting_date') >= '2024-01-01')
    .join(df_comm_base_agg, ['fs_cust_id'], 'left')
    .withColumn(
        'comms_flag'
        , f.when(
             # (f.date_format('min_event_date', 'yyyy-MM-dd' ) <= f.col('reporting_date'))
             f.col('min_event_date').isNotNull()
            , f.lit('comms')
        )
        .otherwise(f.lit('no_comms'))
    )
    .groupBy('reporting_date', 'comms_flag', 'product_holding')
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.countDistinct('fs_acct_id')
        , f.count('*') 
    )
)


# COMMAND ----------

# DBTITLE 1,port vs. comms
display(
    df_port_base
    .filter(f.col('target_date') >= '2024-01-01')
    .join(df_comm_base_agg, ['fs_cust_id'], 'left')
    .withColumn(
        'Comm_Status'
        , f.when(
            f.col('target_date') >= f.col('min_event_date')
            , 'ChurnPostComm'
        )
        .when(
            (f.col('target_date') <= f.col('min_event_date'))
            | (f.col('min_event_date').isNull())
            , 'ChurnNoComm'
        )
    )
    .withColumn(
        'discount_removal'
        , f.when(
            f.col('min_event_date').isNull()
            , f.lit(0)
        )
        .otherwise(f.lit(1))
    )
    .groupBy('Comm_Status'
             , 'GAINING_SERVICE_PROVIDER_NAME_GRP'
             , f.next_day('target_date', 'Sunday')
             , f.date_format('target_date', 'yyyy-MM')
    )
    .agg(
        f.countDistinct('fs_cust_id')
         , f.countDistinct('fs_srvc_id')
        )
)

# COMMAND ----------

# DBTITLE 1,port months after comms
display(
    df_port_base
    .filter(f.col('target_date') >= '2024-01-01')
    #.select('reporting_date', 'date_key', 'fs_acct_id', 'fs_srvc_id', 'fs_cust_id', '')
    .join(df_comm_base_agg, ['fs_cust_id'], 'left')
    .withColumn(
        'Comm_Status'
        , f.when(
            f.col('target_date') >= f.col('min_event_date')
            , 'ChurnPostComm'
        )
        .when(
            (f.col('target_date') <= f.col('min_event_date'))
            | (f.col('min_event_date').isNull())
            , 'ChurnNoComm'
        )
        # .when(
        #     f.col('eventdate').isNull()
        #     , 'ChurnNoComm'
        # )
    )
    .withColumn(
        'port_tenure_from_comm'
        , f.months_between(
            'target_date', f.col('min_event_date')
        )
    )
    .filter(f.col('Comm_Status') == 'ChurnPostComm')
    .groupBy('GAINING_SERVICE_PROVIDER_NAME_GRP', 'comm_cnt', 'RATEPLAN_DESC')
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.avg('port_tenure_from_comm')
        ,f.median('port_tenure_from_comm')
        , f.min('port_tenure_from_comm')
        , f.max('port_tenure_from_comm')
        , f.percentile_approx('port_tenure_from_comm', 0.75)
    )
)

# COMMAND ----------

display(
    df_comm_base_agg
    .groupBy(f.date_format('min_event_date', 'yyyy-MM-dd'))
    .agg(f.countDistinct('fs_cust_id'))
)

# COMMAND ----------

# DBTITLE 1,port + comms + converged + bb churned
df_output_base = (
    df_port_base
    .filter(f.col('target_date') >= '2024-01-01')
    .withColumn(
        '2d_promo_period'
        , f.when(
            f.col('target_date').between('2024-11-22', '2024-12-08')
            , 'Y'
        )
        .otherwise('N')
    )
    .withColumn(
        'discount_removal_period'
        , f.when(
            f.col('target_date').between('2024-09-05', '2024-11-29')
            , 'Y'
        )
        .otherwise('N')
    )
    .withColumn('target_reporting_date', f.next_day('target_date', 'Sunday'))
    #.select('reporting_date', 'date_key', 'fs_acct_id', 'fs_srvc_id', 'fs_cust_id', '')
    .join(df_comm_base_agg, ['fs_cust_id'], 'left')
    .withColumn(
        'Comm_Status'
        , f.when(
            f.col('target_date') >= f.col('min_event_date')
            , 'ChurnPostComm'
        )
        .when(
            (f.col('target_date') <= f.col('min_event_date'))
            | (f.col('min_event_date').isNull())
            , 'ChurnNoComm'
        )
    )
    .withColumn(
        'discount_removal'
        , f.when(
            f.col('min_event_date').isNull()
            , f.lit(0)
        )
        .otherwise(f.lit(1))
    )
    .withColumn(
        'port_tenure_post_comm'
        , f.months_between(
            'target_date', f.col('min_event_date')
        )
    )
   .join(
       df_convged_base
       .drop('bb_reporting_date', 'bb_fs_acct_id')
       ,['reporting_date', 'fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
       , 'left'
    )
   .join(
       df_bb_churn_base
       .filter(f.col('bb_reporting_date') >= '2024-01-01')
        ,(f.col('fs_acct_id') == f.col('bb_fs_acct_id')) 
        & (f.col('fs_cust_id') == f.col('bb_fs_cust_id'))
        # & (f.col('reporting_date') == f.col('bb_reporting_date'))
        , 'left'
    )
   .withColumnRenamed('BB_REPORTING_DATE', 'bb_last_reporting_date')
   .withColumn(
        'bb_churn_status', 
        f.when(
            f.col('bb_last_reporting_date') >= f.col('min_event_date')
            , 'BBChurnPostComm'
        )
        .when(
            f.col('bb_last_reporting_date') <= f.col('min_event_date')
            , 'BBChurnNoComm'
        )
        .when(
            f.col('bb_last_reporting_date').isNull()
            , 'BBNoChurn'
        )
    )
)

# COMMAND ----------

# DBTITLE 1,wallet daily aggregate
df_wallet_offer_daily_agg = (
    df_wallet_offer
    .select('d_date_key', 'WALLET_ID', 'BALANCE_EARNED', 'CAMPAIGN_ID', 'CAMPAIGN_START_DATE_TIME', 'BILLING_ACCOUNT_NUMBER', 'CAMPAIGN_NAME')
    .groupBy('d_date_key', 'WALLET_ID', 'BILLING_ACCOUNT_NUMBER')
    .agg(
        f.sum('BALANCE_EARNED').alias('SUM_WALLENT_BALANCE')
        , f.countDistinct('CAMPAIGN_ID').alias('CNT_WALLET_CAMPAIGN')
        , f.min('CAMPAIGN_START_DATE_TIME').alias('min_campaign_date')
        , f.max('CAMPAIGN_START_DATE_TIME').alias('max_campaign_date')
        , f.collect_list('CAMPAIGN_NAME').alias('CAMPAIGN_NAME_LIST')
    )
    .withColumn('wallet_date_key', f.to_date(f.col("d_date_key").cast("string"), "yyyyMMdd") )
    .drop('d_date_key')
)

# COMMAND ----------

# DBTITLE 1,wallet balance for
display(
    df_convged_base
    .filter(f.col('reporting_date') >= '2024-01-01')
    .join(df_comm_base_agg, ['fs_cust_id'], 'left')
    .join(
        df_wallet_offer_daily_agg
        , (f.col('fs_srvc_id') == f.col('WALLET_ID'))
        & (f.col('reporting_date') == f.col('wallet_date_key'))
        , 'left'
    )
    .withColumn(
        'comms_flag'
        , f.when(
             # (f.date_format('min_event_date', 'yyyy-MM-dd' ) <= f.col('reporting_date'))
             f.col('min_event_date').isNotNull()
            , f.lit('comms')
        )
        .otherwise(f.lit('no_comms'))
    )
    .groupBy('reporting_date', 'comms_flag', 'product_holding')
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.countDistinct('fs_acct_id')
        , f.count('*') 
        , f.countDistinct('wallet_id')
        , f.sum('SUM_WALLENT_BALANCE').alias('SUM_WALLENT_BALANCE')
    )
)

# COMMAND ----------

df_output_curr = (
    df_output_base
    .join(
        df_wallet_offer_daily_agg
        , (f.col('fs_srvc_id') == f.col('WALLET_ID'))
            & (f.col('target_date') == f.col('wallet_date_key')) 
        , 'left' 
    )
    #.count()
)

# COMMAND ----------

display(df_output_curr.limit(10))

display(
    df_output_curr
    .agg(
        f.countDistinct('fs_srvc_id')
        , f.countDistinct('fs_acct_id')
        , f.count('*')
    )   
)

# COMMAND ----------

# DBTITLE 1,look at discount period
df_wallet_offer_disc=(
    df_wallet_offer
    .filter(f.col('CAMPAIGN_START_DATE_TIME').between('2024-09-05', '2024-11-21'))
    .select('BILLING_ACCOUNT_NUMBER', 'WALLET_ID' , 'CAMPAIGN_ID', 'BALANCE_EARNED', 'CAMPAIGN_START_DATE_TIME' )
    .distinct()
    .groupBy('BILLING_ACCOUNT_NUMBER', 'WALLET_ID'
             #, 'CAMPAIGN_ID', 'CAMPAIGN_START_DATE_TIME'
    )
    .agg(
        f.sum('BALANCE_EARNED').alias('SUM_BALANCE_EARNED')
        , f.countDistinct('CAMPAIGN_ID').alias('CNT_WALLET_CAMPAIGN') 
        , f.min('CAMPAIGN_START_DATE_TIME').alias('min_campaign_date')
        , f.max('CAMPAIGN_START_DATE_TIME').alias('max_campaign_date')
        )
    #.filter(f.col('BILLING_ACCOUNT_NUMBER') == '267128')
)

# COMMAND ----------

# DBTITLE 1,check wallent balance
display(
    df_output_base
    .filter(f.col('discount_removal') == 0)
    .filter(f.col('GAINING_SERVICE_PROVIDER_NAME_GRP') == '2degrees')
    .filter(f.col('discount_removal_period') =='Y')
    .filter(f.col('2d_promo_period') == 'N')
    .join(
        df_wallet_offer_disc
        #.filter(f.col('d_date_key') >= 20240901)
        , f.col('fs_srvc_id') == f.col('WALLET_ID')
        , 'inner'
    )
   .groupBy('BILLING_ACCOUNT_NUMBER', 'WALLET_ID' , 'CNT_WALLET_CAMPAIGN', 'min_campaign_date')
   .agg(f.max('SUM_BALANCE_EARNED'))
) 

# service level 348 vs. service level 1113

# COMMAND ----------

# DBTITLE 1,test discount period join with wallet snapshot
display(
    df_output_base
    .filter(f.col('discount_removal') == 1)
    .filter(f.col('GAINING_SERVICE_PROVIDER_NAME_GRP') == '2degrees')
    .filter(f.col('discount_removal_period') =='Y')
    .filter(f.col('2d_promo_period') == 'N')
    .join(
        df_wallet_offer_daily_agg
        , (f.col('fs_srvc_id') == f.col('WALLET_ID')) & 
            (f.col('target_date') == f.col('wallet_date_key'))
        , 'left'
    ) 
) 

# 348 VS. 393 

# COMMAND ----------

display(df_output_curr)
