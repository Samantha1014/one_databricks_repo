# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 environment

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

# COMMAND ----------

# DBTITLE 1,directory
dir_data_dl_brm = '/mnt/prod_brm/raw/cdc'

# COMMAND ----------

# DBTITLE 1,load data
# feature unit base
df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base') 

# fea plan 
df_fea_plan = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_plan')

#cohort segmentaiton 
df_cohort_all = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all')

# stage bill base 
df_stag_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_bill_t')

# fea bill 
df_fea_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6')

# payment 
df_stag_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist')
df_prm_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_payment_cycle_rolling_6')

# COMMAND ----------

# DBTITLE 1,parameters 00
ls_param_unit_base_fields_key = [
    'fs_acct_id', 'fs_srvc_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type'
]

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 load data 

# COMMAND ----------

# DBTITLE 1,bill acct base
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
    and bill_due_date between '2024-11-23' and '2024-12-31'
    """
)

# COMMAND ----------

# DBTITLE 1,auto pay query
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
        qualify row_number()over(partition by fs_acct_id order by bi.mod_t desc, bi.last_bill_t  desc) =1
--and account_no in ('492187222')
    """
)

# COMMAND ----------

display(df_bill_base.count()) # 1210492 

# COMMAND ----------

display(
        df_bill_base
        .agg(
                f.max('bill_mod_date')
        )
)

# COMMAND ----------

# DBTITLE 1,cohort due date
display(df_cohort_all
        .filter(f.col('reporting_date') == '2025-02-09')
        .filter(f.col('L2_combine').isin(
                'Chronic Late', 'Sporadic Late'
                )
        )
        .join(
                df_bill_base
              #.filter(f.col('reporting_date') == '2024-11-17')
                .select('fs_acct_id', 'bill_start_date', 'bill_end_date', 'bill_due_date' , 'bill_close_date')
                .distinct()
                , ['fs_acct_id']
                , 'left'
        )
        .groupBy('bill_due_date')
        .agg(
                f.countDistinct('fs_acct_id').alias('distinct_acct')
        )
)

# COMMAND ----------

# DBTITLE 1,check example
display(
        df_cohort_all
        .filter(f.col('reporting_date') == '2024-12-01')
        .filter(f.col('L2_combine').isin('Chronic Late', 'Sporadic Late'))
        .join(
          df_bill_base
        #.filter(f.col('reporting_date') == '2024-11-17')
          .select('fs_acct_id', 'bill_start_date', 'bill_end_date', 'bill_due_date' , 'bill_close_date')
          .distinct()
          , ['fs_acct_id']
          , 'left'
        )
        .filter(f.col('bill_due_date') =='2024-12-06')
        .filter(f.col('bill_close_date') <= '1970-01-31')
        .limit(10)
        # .groupBy('bill_due_date')
        # .agg(f.countDistinct('fs_acct_id').alias('distinct_acct')
)

# COMMAND ----------

# DBTITLE 1,add payment method
df_test_base = (
      df_cohort_all
      .filter(f.col('reporting_date') == '2024-12-01') # latest reporting date 
      .filter(f.col('L2_combine').isin('Chronic Late', 'Sporadic Late')) # 2 cohorts 
      .join(
            df_bill_base
            .select('fs_acct_id', 'bill_start_date', 'bill_end_date', 'bill_due_date' , 'bill_close_date', 'bill_no')
            .distinct()
            , ['fs_acct_id']
            , 'left'
      )
      .filter(f.col('bill_due_date') =='2024-12-06')
      .filter(f.col('bill_close_date') <= '1970-01-31')
      .join(
            df_payment_method
            .filter(
                  f.col('PAYMENT_EVENT_TYPE')
                  .isin('/event/billing/payment/dd'
                        , '/event/billing/payment/cc')
            )
            , ['fs_acct_id', 'bill_no']
            , 'left'
      )
)

# COMMAND ----------

# DBTITLE 1,materialized table
df_test_base.write.mode('overwrite').saveAsTable('df_test_base_tbl')
df_test_base = spark.table('df_test_base_tbl')

# COMMAND ----------

# DBTITLE 1,check example
display(df_test_base.limit(10))

# COMMAND ----------

# DBTITLE 1,check autopay count in one dom
display(df_test_base
        .join(df_fea_plan
              .select(ls_param_unit_base_fields_key, '')
              , ['fs_acct_id', 'fs_srvc_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type'])
        .groupBy('payment_event_type', 'L2_combine')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

display(df_test_base
        .filter(f.col('fs_acct_id') == '461633546')
)

# COMMAND ----------

display(df_test_base
        .groupBy('bill_due_date', 'reporting_date')
        .agg(f.count('*').alias('cnt'),
              f.countDistinct('fs_acct_id')
              , f.countDistinct('fs_srvc_id')
             )
        #.filter(f.col('cnt') >1)
        )

# COMMAND ----------

display(df_prm_payment
        .filter(f.col('reporting_date') >= '2023-01-01')
        .groupBy('payment_method', 'auto_pay_flag', 'reporting_date'
                 , 'reporting_cycle_type'
                 )
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             , f.count('*')
            )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### s03 autopay per cohort  

# COMMAND ----------

display(df_payment_method.limit(10))

# COMMAND ----------

display(
    df_cohort_all
    .filter(f.col('reporting_date') == '2024-12-01')
    .join(df_payment_method
          , ['fs_acct_id']
          , 'left'   
    )
)


