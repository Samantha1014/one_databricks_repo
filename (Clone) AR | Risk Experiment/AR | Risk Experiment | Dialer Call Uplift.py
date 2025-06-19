# Databricks notebook source
# MAGIC %md
# MAGIC ### s01 set up

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

# DBTITLE 1,directories
dir_edw_data_parent = "/mnt/prod_edw/raw/cdc"
dir_brm_data_parent = "/mnt/prod_brm/raw/cdc"
dir_fs_data_parent = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer"

# COMMAND ----------

# MAGIC %md
# MAGIC ### s02 load data

# COMMAND ----------

#cohort segmentaiton 
df_cohort_all = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/df_combine_all')

# payment and billing 
df_payment_latest = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_latest')
df_fea_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_bill_cycle_billing_6')

# unit base 
df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base')

# stag payment and bill 
df_stag_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_bill_t')
df_stag_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist') 

## control group dialer calls 
df_output_control_20241201 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_control_20241201')
df_output_treatment_20241201 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_target_20241201')
df_output_control_20241124 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_control_20241124')
df_output_treatment_20241124 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_target_20241124')
df_output_control_20241117 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_control_20241117')
df_output_treatment_20241117 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/dailer_list/call_audience_target_20241117')


# COMMAND ----------

display(df_stag_bill
        .filter(f.col('bill_due_date') >= '2023-01-01')
        .select(
            f.col('fs_acct_id')
          #  , f.col('fs_cust_id')
            , f.col('bill_due_date')
                )
        .distinct()
        .groupBy('bill_due_date')
        .agg(f.countDistinct('fs_acct_id'))
)

# COMMAND ----------

# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "SANDBOX",
  "sfWarehouse": "LAB_DS_WH"
}


# COMMAND ----------

df_call_list = (spark
                .read
                .format("snowflake")
                .options(**options)
                .option(
                    "query"
                    , f"""select 
                       distinct 
                        ACCOUNT_REF_NO as fs_acct_id
                        , PRIMARY_CONTACT_FIRST_NAME
                        , PRIMARY_CONTACT_LAST_NAME
                        , HOME_PH_NUM
                        , FILE_DATE
                        from LAB_ML_STORE.SANDBOX.TEMP_AR_DAILER_CALLS_ACCOUNT_DW 
                        where FILE_DATE >= '2024-05-01' 
                        and fs_acct_id is not null 
                        """
                )
                .load()
             )

# COMMAND ----------

df_payment_base = spark.sql(
    f"""
     with extract_target as (
    select
        regexp_replace(acct.account_no, '^0+', '') as fs_acct_id
        , item.poid_id0 as item_poid_id0
        , to_date(from_utc_timestamp(from_unixtime(item.created_t), 'Pacific/Auckland')) as payment_create_date
        , to_date(from_utc_timestamp(from_unixtime(item.mod_t), 'Pacific/Auckland')) as payment_mod_date
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
    qualify row_number()over(partition by item.poid_id0 order by payment_mod_date desc ) = 1 
)
select * from extract_target
where 1 = 1
    and  payment_create_date between '2024-11-01' and '2024-12-30'
    """
)

# COMMAND ----------

display(df_output_control_20241117.limit(10))

# COMMAND ----------

display(df_output_control_20241201
        .drop('Group')
)

# COMMAND ----------

# DBTITLE 1,control groups
df_dialer_control = (
    df_output_control_20241201
    .drop('Group')
    .union(
        df_output_control_20241124
        .drop('Group')
    )
    .union(df_output_control_20241117)
)

df_dialer_treatment = (
    df_output_treatment_20241117
    .drop('Group')
    .union(
        df_output_treatment_20241124
        .drop('bill_overdue_days_ntile', 'Group')
    )
    .union(df_output_treatment_20241201
           .drop('bill_overdue_days_ntile', 'Group')
    )
)


# COMMAND ----------

# DBTITLE 1,check cnt
display(
    df_dialer_control
    .groupBy('reporting_date')
    .agg(
        f.countDistinct('fs_acct_id')
    )
)

display(
    df_dialer_treatment
    .groupBy('reporting_date')
    .agg(
        f.countDistinct('fs_acct_id')
    )
)

# COMMAND ----------

display(df_call_list
        .limit(10)   
)

# COMMAND ----------

# call list also in treatement lsit 
df_call_list_t = (
    df_call_list
    .filter(f.col('FILE_DATE') >= '2024-11-17')
    .join(df_dialer_treatment, ['fs_acct_id'], 'inner')
    .filter(f.col('FILE_DATE') >= f.col('reporting_date'))
    .filter(f.col('FILE_DATE') <= f.date_add('reporting_date', 7)) # target first 7 days 
    .withColumn(
        'rnk'
        , f.row_number().over(
            Window
            .partitionBy('fs_acct_id')
            .orderBy(f.asc('FILE_DATE'))
        )
    )
)

# COMMAND ----------

display(
    df_call_list_t
    .groupBy('FILE_DATE', 'fs_acct_id')
    .agg(f.countDistinct('reporting_date').alias('cnt'))
    .filter(f.col('cnt') >1)
)

# COMMAND ----------

display(df_call_list_t.limit(10))

# COMMAND ----------

display(df_call_list_t
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct(f.concat('fs_acct_id', 'rnk'))     
        )
)

# COMMAND ----------

display(df_call_list_t
        .groupBy('FILE_DATE', 'reporting_date')
        .agg(
            f.count('*')
            , f.countDistinct('fs_acct_id')
        )
)

# COMMAND ----------

df_payment_summary_t = (
    df_call_list_t
    .join(df_payment_base, ['fs_acct_id'], 'left')
    .groupBy('FILE_DATE', 'fs_acct_id', 'reporting_date', 'L2_combine')
    .agg(
        # Payment count and sum for payments made within 3 days
        f.sum(
            f.when(
                f.col('payment_create_date').between(
                    f.col('FILE_DATE')
                    , f.date_add(f.col('reporting_date'), 3)
                )
                , f.lit(1)
            )
              .otherwise(f.lit(0))
        ).alias('cnt_payment_3days')
        , f.sum(
            f.when(
                f.col('payment_create_date').between(
                    f.col('FILE_DATE')
                    , f.date_add(f.col('FILE_DATE'), 3)
                )
                , f.col('item_total')
            )
            .otherwise(0)
        ).alias('sum_payment_amt_3days')
    )
)

# Handling the case where no payment was made
# Since we are doing a left join, if there is no corresponding record in df_payment_base,
# the aggregated counts and sums will naturally be zero.

# COMMAND ----------

display(
    df_payment_summary_t
    .limit(10)
)

# COMMAND ----------

display(
    df_payment_summary_t
    .withColumn(
        'payment_flag'
        , f.when(f.col('sum_payment_amt_3days') < 0, f.lit(1))
            .otherwise(f.lit(0))
    )
    .groupBy('reporting_date', 'L2_combine', 'payment_flag')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*')
    )
    #.withColumn('pct', )           
)  


# COMMAND ----------

# MAGIC %md
# MAGIC #### control

# COMMAND ----------

display(df_dialer_control)

# COMMAND ----------

df_payment_summary_c = (
    df_dialer_control
    .join(df_payment_base, ['fs_acct_id'], 'left')
    .groupBy('FILE_DATE', 'fs_acct_id', 'reporting_date', 'L2_combine')
    .agg(
        # Payment count and sum for payments made within 3 days
        f.sum(
            f.when(
                f.col('payment_create_date').between(
                    f.col('FILE_DATE')
                    , f.date_add(f.col('FILE_DATE'), 3)
                )
                , f.lit(1)
            )
              .otherwise(f.lit(0))
        ).alias('cnt_payment_3days')
        , f.sum(
            f.when(
                f.col('payment_create_date').between(
                    f.col('FILE_DATE')
                    , f.date_add(f.col('FILE_DATE'), 3)
                )
                , f.col('item_total')
            )
            .otherwise(0)
        ).alias('sum_payment_amt_3days')
    )
)

# Handling the case where no payment was made
# Since we are doing a left join, if there is no corresponding record in df_payment_base,
# the aggregated counts and sums will naturally be zero.
