# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 set up

# COMMAND ----------

# DBTITLE 1,library
import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window
from typing import Optional, List, Dict, Tuple
from pyspark.sql.functions import broadcast

# COMMAND ----------

# MAGIC %run "./utils_stratified_sampling"

# COMMAND ----------

# DBTITLE 1,directory
dir_data_dl_brm = '/mnt/prod_brm/raw/cdc'

# COMMAND ----------

# DBTITLE 1,parameter
ls_joining_key = ['fs_acct_id', 'fs_cust_id', 'fs_srvc_id', 'reporting_date', 'reporting_cycle_type']
vt_reporting_date = '2024-12-01'

# COMMAND ----------

# DBTITLE 1,connector
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "prod_pdb_masked",
  "sfSchema": "modelled",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

# DBTITLE 1,load data
#cohort segmentaiton 
df_cohort_all = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all' )

# raw billing account for DOM date 
df_raw_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d100_raw/d101_dp_c360/d_billing_account')


# raw brm bill_t 
df_raw_billt = spark.read.format('delta').load('/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T')

# feature unit base 
df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')
# df_fea_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6')

# payment and billing 
df_payment_latest = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_latest')
df_fea_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6')

# stag payment and bill 
df_stag_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_bill_t')
df_stag_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist') 

# COMMAND ----------

# DBTITLE 1,bill t raw daily update
df_bill_base = spark.sql(
    f"""
     with extract_target as (
    select
        regexp_replace(acct.account_no, '^0+', '') as fs_acct_id
        , bill.bill_no
        , acct.account_no
        , bill.mod_t as bill_mod_t
        , to_date(from_utc_timestamp(from_unixtime(bill.mod_t), 'Pacific/Auckland')) as bill_mod_date
        , to_date(from_utc_timestamp(from_unixtime(bill.start_t), 'Pacific/Auckland')) as bill_start_date
        , to_date(from_utc_timestamp(from_unixtime(bill.end_t), 'Pacific/Auckland')) as bill_end_date
        , to_date(from_utc_timestamp(from_unixtime(bill.due_t), 'Pacific/Auckland')) as bill_due_date   
        , to_date(from_utc_timestamp(from_unixtime(bill.closed_t), 'Pacific/Auckland')) as bill_close_date   
        , bill.closed_t as bill_close_t
        , bill.total_due
        , bill.due
        , bill.recvd
    from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T` as bill
    inner join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T` as acct
        on bill.account_obj_id0 = acct.poid_id0
        and acct._is_latest = 1
        and acct._is_deleted = 0
        and acct.account_no not like 'S%'
    where
        1 = 1
        and bill._is_latest = 1
        and bill._is_deleted = 0
        and bill.bill_no is not null 
    qualify row_number()over(partition by bill.bill_no order by bill_mod_date desc ) =1
)
select * from extract_target
where 1 = 1
    and bill_due_date between '2024-11-20' and '2024-12-10'
    """
)

# COMMAND ----------

display(df_bill_base
        .agg(f.max('bill_mod_date'))
        ) # last modified date 

# COMMAND ----------

# DBTITLE 1,export bill base
(
 df_bill_base
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/bill_base_20241205')
)

# COMMAND ----------

# DBTITLE 1,load bill base
df_bill_base_dl = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/bill_base_20241205')

# COMMAND ----------

display(df_bill_base_dl)

# COMMAND ----------

# DBTITLE 1,check
display(df_bill_base_dl
        .filter(f.col('bill_close_t')!=0)
        #.filter(f.col('bill_due_date') == '2024-11-18')
        .groupBy('bill_due_date', 'bill_end_date')
        .agg(f.countDistinct('fs_acct_id')
             )
        )

# COMMAND ----------

# DBTITLE 1,payment raw daily update
df_payment_base = spark.sql(
    f"""
     with extract_target as (
    select
        regexp_replace(acct.account_no, '^0+', '') as fs_acct_id
        , item.poid_id0 as item_poid_id0
        , to_date(from_utc_timestamp(from_unixtime(item.created_t), 'Pacific/Auckland')) as payment_create_date
        , to_date(from_utc_timestamp(from_unixtime(item.mod_t), 'Pacific/Auckland')) as payment_mod_date
        , to_date(from_utc_timestamp(from_unixtime(item.effective_t), 'Pacific/Auckland')) as payment_effective_date
        , item.item_total
    from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T` as item
    inner join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T` as acct
        on item.account_obj_id0 = acct.poid_id0
        and acct._is_latest = 1
        and acct._is_deleted = 0
        and acct.account_no not like 'S%'
    where
        1 = 1
        and item._is_latest = 1
        and item._is_deleted = 0
        and item.poid_type in ('/item/payment')
    qualify row_number()over(partition by item.poid_id0 order by payment_mod_date desc ) =1
)
select * from extract_target
where 1 = 1
    and  payment_effective_date between '2024-11-15' and '2024-12-06'
    """
)

# COMMAND ----------

display(df_payment_base.count()) # 606173


# COMMAND ----------

# DBTITLE 1,export payment
(
 df_payment_base
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/payment_base_20241205')
)

# COMMAND ----------

# DBTITLE 1,load payment base
df_payment_dl = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/payment_base_20241205')

# COMMAND ----------

display(df_payment_dl
        .groupBy('payment_effective_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.sum('item_total')
             , f.count('*')
             , f.countDistinct('item_poid_id0')
             )
        )

# COMMAND ----------

# DBTITLE 1,payment method curr
df_payment_method = spark.sql(
    f"""
        select bi.pay_type
            -- , bi.payinfo_obj_id0 
             , bi.payinfo_obj_type
            -- , acct.account_no 
             , regexp_replace(acct.account_no, '^0+', '') as fs_acct_id
            -- , bi.bill_obj_id0
             , bi.poid_id0 as bill_poid_id0
             , bt.bill_no
             , pay_t.payinfo_type
             , pay_t.payment_event_type
            from delta.`{dir_data_dl_brm}/RAW_PINPAP_BILLINFO_T` bi
            inner join delta.`{dir_data_dl_brm}/RAW_PINPAP_ACCOUNT_T` acct on bi.account_obj_id0 = acct.poid_id0
            inner join delta.`{dir_data_dl_brm}/RAW_PINPAP_BILL_T` bt on bt.ar_billinfo_obj_id0 = bi.ar_billinfo_obj_id0
            inner join delta.`{dir_data_dl_brm}/RAW_PINPAP_CONFIG_PAYMENT_PAY_TYPES_T` pay_t on pay_t.rec_id = bi.pay_type 
        where bi._is_deleted = 0 and bi._is_latest = 1
        and acct._is_deleted = 0 and acct._is_latest = 1 
        and bt._is_deleted = 0 and bt._is_latest = 1
        and pay_t._is_deleted = 0 and pay_t._is_latest = 1
        and account_no not like 'S%'
        and bt.parent_billinfo_obj_type in ('/billinfo')
        and bt.bill_no is not null 
--and account_no in ('492187222')
    """
)

# COMMAND ----------

# DBTITLE 1,bb landline voice
df_voice_base = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
         select 
            TO_DATE(TO_VARCHAR(d_snapshot_date_key), 'YYYYMMDD') as reporting_date
            ,s.service_id as bb_fs_srvc_id
            , s.service_linked_id as bb_fs_srvc_linked_id
            , billing_account_number as fs_acct_id
            , c.customer_source_id as bb_fs_cust_id
            , service_access_type_name 
            , s.service_type_name
            , s.proposition_product_name
            , s.plan_name as bb_plan_name
            , s.service_start_date_time as bb_service_start_date_time
            , s.service_first_activation_date as bb_service_first_activation_date
    from prod_pdb_masked.modelled.f_service_activity_daily_snapshot f 
    inner join prod_pdb_masked.modelled.d_service s on f.d_service_key = s.d_service_key 
    inner join prod_pdb_masked.modelled.d_billing_account b on b.d_billing_account_key = s.d_billing_account_key 
    inner join prod_pdb_masked.modelled.d_customer c on c.d_customer_key = b.d_customer_key
    where  
            s.service_type_name in ( 'Voice')
            and c.market_segment_name in ('Consumer')
            and f.d_snapshot_date_key in ( '20241204');
    """
    ).load()
)

# COMMAND ----------

# DBTITLE 1,auto pay
df_auto_pay = (
    df_payment_method
    .filter(f.col('PAYMENT_EVENT_TYPE')
            .isin(  '/event/billing/payment/dd'
                  , '/event/billing/payment/cc')
    )
)

# COMMAND ----------

# DBTITLE 1,landline acct
df_landline_acct = (
    df_voice_base
    .select('fs_acct_id')
    .distinct()
)

# COMMAND ----------

display(df_landline_acct
        .count()
        )
        # 82K 

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 transform & filter

# COMMAND ----------

# DBTITLE 1,reload
#df_bill_base = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/bill_base_20241113')

# COMMAND ----------

# DBTITLE 1,single connect
df_single_connect = (df_fea_unitbase
        .filter(f.col('reporting_date') == vt_reporting_date)
        .select('fs_acct_id', 'fs_srvc_id' , 'num_of_active_srvc_cnt')
        .distinct()
        .withColumn('cnt',
                     f.count('fs_srvc_id').over(Window.partitionBy('fs_acct_id')))
        .filter(f.col('cnt') ==1)
       # .filter(f.col('num_of_active_srvc_cnt')==1)
        .select('fs_acct_id')
        .distinct()
        )

# COMMAND ----------

# DBTITLE 1,check cnt
display(df_bill_base_dl
        .filter(f.col('bill_close_t') ==0)
        .filter(f.col('total_due') >0)
        .groupBy('bill_due_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

print('Check 29th Nov Due Date')

display(df_bill_base_dl
        .filter(f.col('bill_due_date') == '2024-12-06')
        .filter(f.col('bill_close_t') ==0)
        .filter(f.col('total_due') >0)
        )

# COMMAND ----------

display(df_auto_pay
        .filter(f.col('fs_acct_id') == '483205134')
        )

# COMMAND ----------

vt_reporting_date

# COMMAND ----------

display(df_cohort_all
        .filter(f.col('reporting_date') == vt_reporting_date)
        .groupBy('L2_combine')
        .agg(f.countDistinct('fs_acct_id'))
)

# COMMAND ----------

# DBTITLE 1,filter on criteria
# chronic late and sporadic late 
# due in 29th Nov 
# bill has not closed yet 
# total owning amount > 0 
# not in auto pay currently 
# 

df_sms_base = (df_cohort_all
        .filter(f.col('reporting_date') == vt_reporting_date)
        .filter(f.col('L2_combine').isin('Chronic Late', 'Sporadic Late', 'Struggling Payer', 'Overloaded', 'Intentional Offender'))
        .select('fs_acct_id'
                , 'fs_srvc_id'
                , 'L2_combine'
                , 'reporting_date'
                )
        .distinct()
        .join(
            df_bill_base_dl
                .filter(f.col('bill_due_date') == '2024-12-06')
                .filter(f.col('bill_close_t') ==0)
                .filter(f.col('total_due') >0)
            , ['fs_acct_id']
            , 'inner'
        )
        .join(df_auto_pay, ['fs_acct_id'], 'anti')
        .join(df_landline_acct, ['fs_acct_id'], 'anti')
        .join(df_single_connect, ['fs_acct_id'], 'inner')
        .distinct()
        # .groupBy('bill_due_date', 'L2_combine')
        # .agg(f.countDistinct('fs_acct_id')
        #      , f.countDistinct('bill_no')
        #     , f.count('*')
        #      )
        )

# COMMAND ----------

display(df_sms_base
        .groupBy('L2_combine')
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

# DBTITLE 1,check auto pay pct
display(df_cohort_all
        .filter(f.col('reporting_date') == vt_reporting_date)
        .filter(f.col('L2_combine').isin('Chronic Late', 'Sporadic Late'))
        .select('fs_acct_id'
                , 'fs_srvc_id'
                , 'L2_combine'
                , 'reporting_date'
                )
        .distinct()
        .join(
            df_bill_base_dl
                .filter(f.col('bill_due_date') == '2024-12-06')
                .filter(f.col('bill_close_t') ==0)
                .filter(f.col('total_due') >0)
            , ['fs_acct_id'], 'inner'
            )
        .join(df_auto_pay, ['fs_acct_id'], 'left')
        .withColumn('auto_pay_flag', f.when( f.col('bill_poid_id0').isNull(), 0).otherwise(1))
        .groupBy('L2_combine', 'reporting_date', 'bill_due_date', 'auto_pay_flag')
        .agg(f.countDistinct('fs_acct_id'))
)

# COMMAND ----------

# DBTITLE 1,check cnt
display(df_sms_base
        .groupBy('bill_due_date', 'reporting_date' , 'L2_combine')
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('bill_no')
             , f.count('*')
             )
        )

# COMMAND ----------

#dbutils.fs.rm('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_campaign_base_20241112')

# COMMAND ----------

# DBTITLE 1,export data
(
    df_sms_base
    .write
    .mode('overwrite')
    .format('delta')
    .option('header', True)
    .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_campaign_base_20241205')
)

# COMMAND ----------

# DBTITLE 1,import data
df_sms_base_v0 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_campaign_base_20241205')
# df_sms_base_v1 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_campaign_base_20241127')
# df_sms_base_v1 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_campaign_base_20241113')
# df_sms_base_v2 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_campaign_base_20241112')

# COMMAND ----------

display(df_sms_base_v0.count())
#display(df_sms_base_v1.count())

# COMMAND ----------

display(df_sms_base_v1
        .join(df_sms_base_v0, ['fs_acct_id'], 'anti')
        )

# COMMAND ----------

# display(df_sms_base_v1
#         .groupBy('bill_due_date', 'reporting_date' , 'L2_combine')
#         .agg(f.countDistinct('fs_acct_id')
#              , f.countDistinct('bill_no')
#              , f.count('*')
#              )
#         )

display(df_sms_base_v0
        .groupBy('bill_due_date', 'reporting_date' , 'L2_combine')
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('bill_no')
             , f.count('*')
             )
        )

# COMMAND ----------

# DBTITLE 1,check bill due date
display(df_sms_base_v0
        .groupBy('bill_due_date', 'bill_end_date')
        .agg(f.countDistinct('fs_acct_id'))
        )

display(df_sms_base_v0
        # .filter(f.col('bill_start_date') == '2024-10-02')
        .filter(f.col('bill_due_date') == '2024-12-06')
        )

# COMMAND ----------

# DBTITLE 1,check percentile
display(df_sms_base_v0
        .agg(
            f.min('total_due')
            , f.approx_percentile('total_due', 0.25)
             , f.approx_percentile('total_due', 0.5)
             ,f.approx_percentile('total_due', 0.75)
             , f.approx_percentile('total_due', 0.95)
             , f.max('total_due')
             )
        )


display(df_sms_base_v0
        .filter(f.col('total_due') >10)
        )        # <10 =  74 counts 

# COMMAND ----------

# MAGIC %md
# MAGIC #### s03 payment activity

# COMMAND ----------

# DBTITLE 1,pay before due
display(df_cohort_all
        .filter(f.col('reporting_date') == '2024-09-30')
        .filter(f.col('L2_combine').isin('Chronic Late', 'Sporadic Late'))
        .join(df_fea_bill
              .filter(f.col('reporting_date') == '2024-09-30')
              .filter(f.col('bill_due_amt') >0)
              .select('fs_acct_id', 'bill_cycle_end_date', 'bill_due_date', 'bill_no', 'bill_close_date', 'bill_due_amt')
              .distinct()
              , ['fs_acct_id']
              , 'left'
              )
        .join(
             df_payment_latest
            .filter(f.col('reporting_date') == '2024-09-30')
            .select('fs_acct_id', 'latest_payment_date', 'latest_payment_amt')
            .distinct()
            , ['fs_acct_id']
            , 'left'
        )
        .select('fs_acct_id', 'bill_cycle_end_date', 'bill_due_date', 'bill_no', 'bill_close_date'
                ,'bill_due_amt' , 'latest_payment_date', 'latest_payment_amt', 'reporting_date', 'L2_combine'
                )
        .distinct()
        .withColumn('pay_before_due'
                    , f.when(
                         f.col('latest_payment_date').between( f.col('bill_cycle_end_date') , f.col('bill_due_date') ) 
                        , 'Y' )
                        .otherwise('N')
                    )
        .withColumn('pct_pay_to_due', 
                    f.when(f.col('pay_before_due') == 'Y'
                           , f.col('latest_payment_amt')/ f.col('bill_due_amt')
                           )
                    .when(f.col('pay_before_due') == 'N'
                          , 0
                          )
                     )
        .withColumn('pay_full', f.when(f.col('pct_pay_to_due') >=1, 'full')
                                 .otherwise('not_full') 
                    )
        #.filter(f.col('L2_combine') == 'Sporadic Late')
        .groupBy('pay_before_due', 'L2_combine', 'pay_full', 'reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             , f.avg('pct_pay_to_due')
             )
        )

# COMMAND ----------

# DBTITLE 1,% of payment activity
display(df_sms_base
        .join(
            df_payment_latest
            .filter(f.col('reporting_date') == vt_reporting_date)
            .select('fs_acct_id', 'latest_payment_date', 'latest_payment_amt')
            .distinct()
            , ['fs_acct_id']
            , 'left'
        )
        .select('fs_acct_id', 'L2_combine', 'latest_payment_date', 'latest_payment_amt')
        .groupBy('L2_combine', 'latest_payment_date')
        .agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

# DBTITLE 1,check
display(df_fea_bill
        .filter(f.col('reporting_date') == vt_reporting_date)
        .select('reporting_date', 'fs_acct_id', 'bill_overdue_days_late_avg_6bmnth'
                , 'bill_overdue_days'
                ,'bill_payment_timeliness_status'
                , 'bill_no'
                , 'bill_due_amt')
        .distinct()
        # .agg(f.count('*')
        #      , f.countDistinct('fs_acct_id')
        #      )

        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### s06 output 

# COMMAND ----------

# DBTITLE 1,bill_overdue_days ntile
df_sms_output = (df_sms_base_v0
        .join(
            df_fea_bill
            .filter(f.col('reporting_date') == vt_reporting_date)
            .select('fs_acct_id', 'fs_cust_id', 'bill_overdue_days_late_avg_6bmnth', 'bill_overdue_days')
            .distinct()
            , ['fs_acct_id']
            , 'left'
        )
        .withColumn('rank', f.col('bill_overdue_days_late_avg_6bmnth'))
        .withColumn(
                    "bill_overdays_late_6bmnth_ntile"
                    , f.ntile(20).over(
                        Window
                        .orderBy(f.asc("rank"))
                    )
                )
        .withColumn(
                    "bill_overdue_days_ntile"
                    , f.ntile(20).over(
                        Window
                        .orderBy(f.asc("bill_overdue_days"))
                    )
                )
        # .groupBy('bill_overdays_late_6bmnth_ntile')
        # .agg(f.count('fs_acct_id'))
        )


# COMMAND ----------

df_sms_output_chronic = (
    df_sms_output
    .filter(f.col('L2_combine') == 'Chronic Late')
    .distinct()
)

df_sms_output_sporadic = (
    df_sms_output
    .filter(f.col('L2_combine') == 'Sporadic Late')
    .distinct()
)

# COMMAND ----------

display(df_sms_output_chronic.count())
display(df_sms_output_sporadic.count())

# COMMAND ----------

display(df_sms_output
        .groupBy( 'bill_overdays_late_6bmnth_ntile', 'bill_overdue_days_ntile')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### s04 generate sample

# COMMAND ----------

# DBTITLE 1,sample creation 1
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 700
#vt_param_proportion = 0.25
ls_param_strata_fields = [
    "bill_overdays_late_6bmnth_ntile"
   , 'bill_overdue_days_ntile'
  #  , "mmc_top_ntile"
  #  , "dr_top_ntile"
    #, "plan_family"
    #, "wallet_top_ntile"
    #, "tenure_top_ntile", "usg_top_ntile", "wallet_top_ntile"
    #, "plan_family", "ifp_acct_dvc_flag", "plan_discount_flag", "wallet_flag"
    #, "age_group", "network_dvc_brand", "srvc_privacy_flag"
]
#vt_param_priority_field = "wallet_control_flag"
#dict_param_priority_groups = {'Y':2}


df_base_control, df_base_target = generate_sample(
    df= df_sms_output_chronic
    , size=vt_param_sample_req
    #, proportion = vt_param_proportion
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=15
)

(df_base_control
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_chronic_control_20241205')
 )

(df_base_target
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_chronic_target_20241205')
 )

df_base_control  = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_chronic_control_20241205')
df_base_target = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_chronic_target_20241205')

# COMMAND ----------

# DBTITLE 1,check dup
display(df_base_control
        .join(df_base_target, ['fs_acct_id'], 'inner')
        )

# COMMAND ----------

# Get a list or set of IDs to be filtered out from df_base_control
ids_to_filter_out = df_base_control.select("fs_acct_id").distinct().rdd.flatMap(lambda x: x).collect()

# Filter df_sms_output_chronic to exclude records that are in df_base_control
df_base_target1 = df_sms_output_chronic.filter(~df_sms_output_chronic["fs_acct_id"].isin(ids_to_filter_out))

# COMMAND ----------

display(df_base_control
        .join(df_base_target1, ['fs_acct_id'], 'inner')
        )

# COMMAND ----------

df_chronic_late_target = df_base_target1
df_chronic_late_control = df_base_control

# COMMAND ----------

# DBTITLE 1,save
display(df_chronic_late_target.count())

display(df_chronic_late_control.count())

# COMMAND ----------

evaluate_sample(
    df_chronic_late_target
    , df_chronic_late_control
    , [ "bill_overdays_late_6bmnth_ntile", 'bill_overdue_days_ntile']
)

# COMMAND ----------

# DBTITLE 1,sample creation 2
# Control group creation
vt_param_seed = 20
vt_param_sample_req = 250
#vt_param_proportion = 0.25
ls_param_strata_fields = [
    "bill_overdays_late_6bmnth_ntile"
    , 'bill_overdue_days_ntile'
  #  , "mmc_top_ntile"
  #  , "dr_top_ntile"
    #, "plan_family"
    #, "wallet_top_ntile"
    #, "tenure_top_ntile", "usg_top_ntile", "wallet_top_ntile"
    #, "plan_family", "ifp_acct_dvc_flag", "plan_discount_flag", "wallet_flag"
    #, "age_group", "network_dvc_brand", "srvc_privacy_flag"
]
#vt_param_priority_field = "wallet_control_flag"
#dict_param_priority_groups = {'Y':2}


df_base_control, df_base_target = generate_sample(
    df= df_sms_output_sporadic
    , size=vt_param_sample_req
    #, proportion = vt_param_proportion
    , strata=ls_param_strata_fields
    #, priority_field=vt_param_priority_field
    #, priority_groups=dict_param_priority_groups
    , seed=15
)

(df_base_control
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_sporadic_control_20241205')
 )

(df_base_target
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_sporadic_target_20241205')
 )

df_base_control  = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_sporadic_control_20241205')
df_base_target = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_sporadic_target_20241205')

# COMMAND ----------

# Get a list or set of IDs to be filtered out from df_base_control
ids_to_filter_out = df_base_control.select("fs_acct_id").distinct().rdd.flatMap(lambda x: x).collect()

# Filter df_sms_output_sporadic to exclude records that are in df_base_control
df_base_target1 = df_sms_output_sporadic.filter(~df_sms_output_sporadic["fs_acct_id"].isin(ids_to_filter_out))

# COMMAND ----------

display(df_base_control
        .join(df_base_target1, ['fs_acct_id'], 'inner')
        )


# COMMAND ----------

df_sporadic_late_target = df_base_target1
df_sporadic_late_control = df_base_control

# COMMAND ----------

evaluate_sample(
    df_sporadic_late_target
    , df_sporadic_late_control
    , [ "bill_overdays_late_6bmnth_ntile", 'bill_overdue_days_ntile']
)

# COMMAND ----------

display(df_sporadic_late_target.count())

display(df_sporadic_late_control.count())

# COMMAND ----------

df_output_all_target = (df_chronic_late_target
        .select('fs_acct_id', 'fs_cust_id', 'fs_srvc_id', 'L2_combine')
        .union(
            df_sporadic_late_target
            .select('fs_acct_id', 'fs_cust_id','fs_srvc_id', 'L2_combine')
        )
        # .groupBy('L2_combine')
        # .agg(f.countDistinct('fs_acct_id')
        #      , f.count('fs_srvc_id')
        #      , f.count('*')
        #      )
        )

# COMMAND ----------

df_output_all_control = (df_chronic_late_control
       # .select('fs_acct_id', 'fs_cust_id', 'fs_srvc_id', 'L2_combine')
        .union(
            df_sporadic_late_control
            #.select('fs_acct_id', 'fs_cust_id','fs_srvc_id', 'L2_combine')
        )
        # .groupBy('L2_combine')
        # .agg(f.countDistinct('fs_acct_id')
        #      , f.count('fs_srvc_id')
        #      , f.count('*')
        #      )
        )

# COMMAND ----------

display(df_output_all_target
        .join(df_output_all_control, ['fs_acct_id'], 'inner')
        )

# COMMAND ----------

display(df_output_all_control
        .groupBy('L2_combine')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('fs_srvc_id')
             , f.count('*')
             )
        )

# COMMAND ----------

display(df_output_all_target
        .groupBy('L2_combine')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('fs_srvc_id')
             , f.count('*')
             )
        )



# COMMAND ----------

# DBTITLE 1,save version
(df_output_all_target
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_treatment_20241205')
 )

# COMMAND ----------

(df_output_all_control
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_control_20241205')
 )

# COMMAND ----------

df_output_treatment = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_treatment_20241205')
df_output_control = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_control_20241205')

# COMMAND ----------

display(df_output_control.count())

display(df_output_treatment.count())

# COMMAND ----------

display(df_output_treatment
        .select('fs_acct_id', 'fs_srvc_id', 'fs_cust_id')
        .distinct()
        )
