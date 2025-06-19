# Databricks notebook source
# MAGIC %md
# MAGIC ##### library

# COMMAND ----------

import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
import pandas as pd

# COMMAND ----------

# MAGIC %run "../utility_functions"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### directory

# COMMAND ----------

dir_edw_data_parent = "/mnt/prod_edw/raw/cdc"
dir_brm_data_parent = "/mnt/prod_brm/raw/cdc"
dir_fs_data_parent = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer"
dir_fs_data_stg = '/mnt/feature-store-prod-lab/d200_staging/d299_src'
dir_fs_data_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/'

# COMMAND ----------

# DBTITLE 1,directory
# collection action 
dir_brm_coll_action = os.path.join(dir_brm_data_parent, 'RAW_PINPAP_COLLECTIONS_ACTION_T')
dir_brm_coll_config_action = os.path.join(dir_brm_data_parent, 'RAW_PINPAP_CONFIG_COLLECTIONS_ACTION_T')
dir_brm_coll_config_scenario =  os.path.join(dir_brm_data_parent, 'RAW_PINPAP_CONFIG_COLLECTIONS_SCENARIO_T')
dir_brm_coll_scenario = os.path.join(dir_brm_data_parent, 'RAW_PINPAP_COLLECTIONS_SCENARIO_T')
dir_brm_coll_scenario_milstone = os.path.join(dir_brm_data_parent, 'RAW_PINPAP_VF_COLLS_SCENARIO_MILESTONES')

# account table 
dir_brm_acct = os.path.join(dir_brm_data_parent, 'RAW_PINPAP_ACCOUNT_T')

# payment stage 
dir_payment_latest = '/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist'

# product holding 

# fs unit base 
dir_fs_unit_base = os.path.join(dir_fs_data_parent, 'fea_unit_base')

# COMMAND ----------

# DBTITLE 1,key
ls_prm_key = ['fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
ls_collection_action_join_key = ['fs_acct_id']
ls_reporting_date_key = ['reporting_date', 'reporting_cycle_type']

# COMMAND ----------

# MAGIC %md
# MAGIC ##### load data

# COMMAND ----------

# DBTITLE 1,load database
# unit base 
df_fs_unitbase = spark.read.format('delta').load(dir_fs_unit_base)

# dedup 
df_base_unit_base = (df_fs_unitbase
        .select(*ls_collection_action_join_key, 'cust_start_date' )
        .distinct()
)

## payment 
df_payment = spark.read.format('delta').load(dir_payment_latest)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Collection Action Features

# COMMAND ----------

df_coll_action  = spark.sql(
        f"""
        select 
            regexp_replace(aa.account_no, '^0+', '') as fs_acct_id,
            a.poid_id0,
            a.scenario_obj_id0, 
            a.poid_type,
            a.created_t, 
            from_utc_timestamp(to_timestamp(a.created_t), 'Pacific/Auckland') as create_dttm_nzt,
            a.mod_t, 
             from_utc_timestamp(to_timestamp(a.mod_t), 'Pacific/Auckland') as mod_dttm_nzt,
            a.due_t, 
            from_utc_timestamp(to_timestamp(a.due_t), 'Pacific/Auckland') as due_dttm_nzt,
            a.completed_t, 
            from_utc_timestamp(to_timestamp(a.completed_t), 'Pacific/Auckland') as complete_dttm_nzt, 
            cca.action_descr, 
            cca.action_name, 
            cca.action_type
        from delta.`{dir_brm_coll_action}` a
        left join delta.`{dir_brm_coll_config_action}` cca on a.config_action_obj_id0 = cca.obj_id0
        left join delta.`{dir_brm_acct}` aa on aa.poid_id0 = a.account_obj_id0
        where 
        a._is_latest = 1 and cca._is_latest = 1 and aa._is_latest = 1
        and completed_t !=0 
        """ 
        )


# COMMAND ----------

# DBTITLE 1,set reporting date
vt_reporting_date = '2024-04-30'

# COMMAND ----------

# MAGIC %md
# MAGIC #### collection category

# COMMAND ----------

# DBTITLE 1,add collection category
## category has been modified according to Shaun's feedback 
# 
#  bar initiated = redirection 
# SMS : NZ DHNR/BRKN Arr Pre-Bar = Swipe Pre Bar Reminder


df_coll_action_01 = (
    df_coll_action
    .join(df_base_unit_base, ['fs_acct_id'], 'inner') 
    .withColumn('complete_date', f.to_date('complete_dttm_nzt') )
    .withColumn('coll_action_category', f.when(f.col('action_name')
                                               .isin(
                                                   'Letter Collection Disclosure Statement'
                                                 , 'Swipe Payment Reminder'
                                                 , 'SMS : NZ Delinquent'
                                                 , 'VMG : NZ Delinquent'
                                                                  )
                                               , 'Early')
                                         .when(f.col('action_name')
                                               .isin(
                                               'Bar Initiated'
                                             , 'Redirection'
                                             , 'SMS : NZ DHNR/BRKN Arr Pre-Bar'
                                             , 'Swipe Pre Bar Reminder'
                                             , 'Letter Data Reminder'
                                             , 'SMS : NZ Delinquent Serious' 
                                                    )
                                               , 'Mid'
                                               )
                                         .when(
                                             f.col('action_name')
                                             .isin(  'InBar'
                                                   , 'Letter Consumer Inbar'
                                                   ,  'Disconnect Initiated'
                                                   ,  'Proposed Disconnect'
                                                   ,  'Disconnect'
                                                   ,  'Letter Collection Final Demand Decision'
                                                   )
                                             , 'Late'
                                             )
                                         .when(
                                             f.col('action_name')
                                             .isin('Writeoff Proposed Auto Action'
                                                   , 'Send to DCA Decision'
                                                   , 'DCA Assignment'
                                                   , 'Complete Worklist'
                                                   )
                                             , 'Pre-writeoff'
                                            )
                                         .when(
                                             f.col('action_name').isNull()
                                             , 'unknown'
                                              )
                                         .otherwise('other')
                                               
                )
    .withColumn('coll_action_category_num'
                , f.when(f.col('coll_action_category') == 'Early', 1)
                 .when(f.col('coll_action_category') == 'Mid', 2)
                 .when(f.col('coll_action_category') == 'Late', 3)
                 .when(f.col('coll_action_category') == 'Pre-writeoff', 4)
                 .when(f.col('coll_action_category') == 'unknown', -1)
                 .when(f.col('coll_action_category') == 'other', 0 )
                )
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### payment 

# COMMAND ----------

df_payment_01 = (
    df_payment
    .join(df_base_unit_base, ['fs_acct_id'], 'inner')
    .select('fs_acct_id'
            , 'payment_effective_date'
            , 'item_total'
            , 'item_poid_id0'
            )
    .filter(f.col('item_poid_type') == '/item/payment')
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### collections actions that pay within 5 days

# COMMAND ----------

## account that have collection actions and then pay within 5 days 
## in collection action level 
df_coll_action_pay = (
    df_coll_action_01
    .join(df_payment_01, ['fs_acct_id'], 'left')
    .withColumn(
        'days_to_payment'
        , f.datediff(f.col('payment_effective_date'), f.col('complete_date'))
    )
    .filter(f.col('days_to_payment').between(0, 5))
    .groupBy(
        'fs_acct_id'
        , 'poid_id0'
        , 'complete_date'
        , 'action_type'
        , 'action_name'
        , 'coll_action_category'
    )
    .agg(
        f.min('payment_effective_date').alias('min_pay_date')
        , f.sum('item_total').alias('total_payment')
        , f.countDistinct('item_poid_id0').alias('cnt_payment')
        , f.avg('days_to_payment').alias('avg_days_to_payment')
    )
    .withColumn('pay_after_coll', f.lit('Y'))
    #.filter(f.col('fs_acct_id') == '1049129')
)

# COMMAND ----------

# DBTITLE 1,data check
display(df_coll_action_01
        .filter(f.col('fs_acct_id') == '351332583')
)

# COMMAND ----------

# DBTITLE 1,data check
display(df_coll_action_pay
        .filter(f.col('fs_acct_id') == '351332583')
        #.filter(f.col('cnt_payment') >10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### collection action 12m version

# COMMAND ----------

#ls_reporting_date = [ '2024-04-30', '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31', '2024-09-30'] 
ls_reporting_date = ['2023-12-31']

df_reporting_dates = spark.createDataFrame([(d,) for d in ls_reporting_date], ["reporting_date"])

# COMMAND ----------

display(df_reporting_dates)

# COMMAND ----------

df_coll_action_12m = (
    df_coll_action_01.alias('a')
    .crossJoin(df_reporting_dates)
    .filter(f.col('complete_date') <= f.col('reporting_date'))
    .filter(f.col('complete_date') >= f.add_months(f.col('reporting_date'), -12))
    .join(df_coll_action_pay, ['fs_acct_id', 'poid_id0'], 'left')
    .withColumn('pay_after_coll_12m',
                f.when(f.col('pay_after_coll').isNull(), f.lit('N'))
                 .when(f.col('pay_after_coll').isNotNull(), f.lit('Y'))
                 .otherwise(f.lit('misc'))
    )
    .groupBy('fs_acct_id', 'reporting_date')
    .agg(
        f.count('poid_id0').alias('cnt_in_coll_action_12m'),
        f.count('pay_after_coll').alias('cnt_pay_after_coll_12m'),
        f.avg('avg_days_to_payment').alias('avg_days_to_payment_12m'),
        f.max('coll_action_category_num').alias('max_coll_action_category_12m'),
        f.max('a.complete_date').alias('last_action_complete_date_12m'),
        f.min('a.complete_date').alias('earliest_action_complete_date_12m')
    )
)

# COMMAND ----------

display(df_coll_action_12m
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             )
        )

# COMMAND ----------

# DBTITLE 1,data check
display(df_coll_action_12m
        .filter(f.col('fs_acct_id') == '351332583')
        )

# COMMAND ----------

# DBTITLE 1,check cnt
display(df_coll_action_12m
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             )
       )

# COMMAND ----------

display(df_coll_action_pay
        .limit(10)
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### collection action curr

# COMMAND ----------

# DBTITLE 1,collection action curr with pay
# Collection action per acct level 
# Aggregate collection action to the latest collection action 
# and count if pay for the latest collection action 
df_coll_action_curr = (
    df_coll_action_01.alias('a')
    .crossJoin(df_reporting_dates)
    .filter(f.col('complete_date') <= f.col('reporting_date'))
    .filter(f.col('complete_date') >= f.date_trunc('MM', f.col('reporting_date')))
    .join(df_coll_action_pay
        , ['fs_acct_id', 'poid_id0']
        , 'left'
    )
    .withColumn('rank'
        , f.row_number().over(
            Window.partitionBy('a.fs_acct_id', 'reporting_date')
                  .orderBy(f.desc('a.complete_date'))
        )
    )
    .filter(f.col('rank') == 1)  # Pick the latest collection action complete within a month 
    .withColumn('latest_pay_after_coll'
        , f.when(f.col('pay_after_coll').isNull(), f.lit('N'))
         .when(f.col('pay_after_coll').isNotNull(), f.col('pay_after_coll'))
         .otherwise(f.lit('misc'))
    )
    .select(
        'fs_acct_id'
        , 'reporting_date'
        , 'poid_id0'
        , f.col('a.action_name').alias('latest_coll_action_name')
        , f.col('a.complete_date').alias('latest_coll_action_date')
        , f.col('a.coll_action_category').alias('latest_coll_action_category')
        , f.col('a.coll_action_category_num').alias('latest_coll_action_category_num')
        , f.col('min_pay_date').alias('min_pay_date_after_coll')
        , f.col('total_payment').alias('total_payment_after_coll')
        , f.col('cnt_payment').alias('cnt_payment_after_coll')
        , f.col('avg_days_to_payment').alias('avg_days_to_payment_after_coll')
        , 'latest_pay_after_coll'
    )
    #.filter(f.col('fs_acct_id') == '351332583')
)

# COMMAND ----------

display(df_coll_action_curr
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id').alias('cnt_fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

# DBTITLE 1,curr joined with 12m and base
df_output_curr = (
    df_fs_unitbase
    .select(*ls_prm_key, *ls_reporting_date_key)
    .filter(f.col('reporting_cycle_type') == 'calendar cycle')
    .distinct()
    .join(
        df_reporting_dates
        , ['reporting_date']
        , 'inner'
    )
    .join(
        df_coll_action_curr
        , ['fs_acct_id', 'reporting_date']
        , 'left'
    )
    .join(
        df_coll_action_12m
        , ['fs_acct_id', 'reporting_date']
        , 'left'
    )
    #.filter(f.col('reporting_date').between('2023-10-31', '2024-07-31'))
    #.filter(f.col('fs_acct_id') == '351332583')
)

# COMMAND ----------

display(df_output_curr
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
)


# COMMAND ----------

display(df_output_curr
        .filter(
                f.col('fs_acct_id') == '351332583')
        )


# COMMAND ----------

display(df_output_curr.limit(10))

display(df_output_curr.count())

# COMMAND ----------

# DBTITLE 1,check profile
display(
        df_output_curr
        .withColumn('coll_action_pay_pct'
                    , f.round(  
                    f.col('cnt_pay_after_coll_12m') / f.col('cnt_in_coll_action_12m')
                    ,2
                      )
                    )
        .groupBy('coll_action_pay_pct', 'max_coll_action_category_12m', 'reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
)



# COMMAND ----------

# MAGIC %md
# MAGIC #### export data

# COMMAND ----------

dbutils.fs.ls('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_coll_action')

# COMMAND ----------

export_data(
            df = df_output_curr
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_coll_action'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### test

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_coll_action')

# COMMAND ----------

display(df_test
        .groupBy('reporting_date')
        .agg(
             f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

display(df_test.limit(10))
