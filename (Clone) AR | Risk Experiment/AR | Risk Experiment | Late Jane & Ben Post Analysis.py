# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 library

# COMMAND ----------

# DBTITLE 1,library
import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 parameter

# COMMAND ----------

# DBTITLE 1,directory
# directory 
dir_audience_conrtol = '/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_control_20241205'
dir_audience_treatment = '/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_treatment_20241205'
#dir_audience_treatment = 'dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/bill_reminder_sent_g3_241209.csv'
dir_audience_base = '/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_campaign_base_20241205/'

# parameters 
ls_joining_key = ['fs_acct_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type']

# COMMAND ----------

# MAGIC %md
# MAGIC #### s03 load data

# COMMAND ----------

# DBTITLE 1,load data
df_audience_control = spark.read.format('delta').load(dir_audience_conrtol)
df_audience_treatment = spark.read.format('delta').load(dir_audience_treatment)
df_audience_base = spark.read.format('delta').load(dir_audience_base) 
df_bill_base_dl = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/bill_base_20241205')

# COMMAND ----------

# DBTITLE 1,actual sent data
# df_sms_sent = spark.read.csv('dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/sms_sent_191124.csv', header=True, inferSchema=True)

df_sms_sent = spark.read.csv('dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/bill_reminder_sent_g3_241209.csv', header = True, inferSchema = True)

# COMMAND ----------

display(df_sms_sent.count())
display(df_sms_sent.limit(10))
display(df_audience_treatment.count()) # 4040 vs. 3960 

# COMMAND ----------

# DBTITLE 1,payment activity
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
    and  payment_effective_date between '2024-11-14' and '2024-12-07'
    """
)

# COMMAND ----------

df_payment_agg = (
        df_payment_base
        .filter(f.col('payment_effective_date') >= '2024-12-06')
        .filter(f.col('payment_effective_date') <= '2024-12-07')
        .groupBy('fs_acct_id')
        .agg(
             f.sum('item_total').alias('total_amt')
             , f.countDistinct('item_poid_id0').alias('payment_cnt')
             , f.min('payment_effective_date').alias('min_payment_date')
             )
        )

# COMMAND ----------

# DBTITLE 1,audience payment
df_audience_payment = (df_audience_base
                      .join(df_payment_agg
                            , ['fs_acct_id']
                            , 'inner')
                      )

# COMMAND ----------

display(
  df_audience_payment
  .limit(10)           
)

# COMMAND ----------

(df_audience_payment
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_payment_20241205')
 )

# COMMAND ----------

df_audience_payment = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/sms_audience_payment_20241205')

# COMMAND ----------

display(df_audience_control)

# COMMAND ----------

# DBTITLE 1,control treatment check
display(df_audience_control
        .join(df_audience_treatment, ['fs_acct_id'], 'inner')
        )

# COMMAND ----------

df_treatment_v1 = (
        df_audience_treatment.alias('a')
        .join(df_sms_sent, f.col('fs_acct_id') == f.col('billing_account_number'), 'inner')
        .join(df_audience_base, ['fs_acct_id', 'fs_srvc_id'], 'inner')
        .select('a.*',df_audience_base['total_due'], df_audience_base['due'] )
)

# COMMAND ----------

display(
  df_treatment_v1
  .agg(
    f.count('*')
    , f.countDistinct('fs_acct_id')
  )
)

# COMMAND ----------

# DBTITLE 1,check cnt
display(
     df_treatment_v1
     .groupBy('L2_combine')
     .agg(
          f.countDistinct('fs_acct_id')
          , f.sum('total_due')
          , f.count('*')
     )
)

# COMMAND ----------

# DBTITLE 1,treatment
display(df_treatment_v1
        .join(df_audience_payment
              .select('fs_acct_id', 'min_payment_date', 'total_amt')
              , ['fs_acct_id'], 'left')
        .withColumn('payment_make_flag'
                    , f.when(f.col('total_amt').isNotNull()
                             , 1
                            )
                       .when(f.col('total_amt').isNull()
                             , 0
                            )
                       .otherwise(-999)
                  )
        #.filter(  f.col('total_amt') >= 0 )
        .groupBy('L2_combine', 'payment_make_flag')
        .agg(  f.countDistinct('fs_acct_id')
             , f.count('*')
             )
      )

# COMMAND ----------

display(df_control_v1
        .join(df_payment_agg, ['fs_acct_id'], 'left')
        .withColumn('payment_make_flag'
                    , f.when(f.col('total_amt').isNotNull()
                             , 1
                            )
                       .when(f.col('total_amt').isNull()
                             , 0
                            )
                       .otherwise(-999)
                  )
        .groupBy('L2_combine', 'payment_make_flag', 'min_payment_date')
        .agg(  f.countDistinct('fs_acct_id')
             , f.count('*')
             )
      )

# COMMAND ----------

display(df_audience_control
        .join(df_audience_payment
              .select('fs_acct_id', 'min_payment_date', 'total_amt')
              , ['fs_acct_id'], 'left')
        .withColumn('payment_make_flag'
                    , f.when(f.col('total_amt').isNotNull()
                             , 1
                            )
                       .when(f.col('total_amt').isNull()
                             , 0
                            )
                       .otherwise(-999)
                  )
        #.filter(f.col('total_amt') >=0 ) 
        .groupBy('L2_combine', 'payment_make_flag')
        .agg(  f.countDistinct('fs_acct_id')
             , f.count('*')
             )
      )

# COMMAND ----------

# DBTITLE 1,treatment pay full %
display(
      df_treatment_v1
      .join(
            df_audience_payment
            .select('fs_acct_id', 'min_payment_date', 'total_amt')
            , ['fs_acct_id'], 'left'
      )
      .withColumn(
            'payment_make_flag'
            , f.when(
                  f.col('total_amt').isNotNull()
                  , 1
            )
            .when(
                  f.col('total_amt').isNull()
                  , 0
            )
            .otherwise(-999)
      )
      .withColumn(
            'pay_full_flag'
            , f.when( 
                  (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') <= f.abs('total_amt'))
                  , 'Y'
            )
            .when (
                  (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') > f.abs('total_amt'))
                  , 'N'
            )
            .otherwise('U')
        )
      .withColumn(
            'days_early'
            , f.date_diff('min_payment_date', f.lit('2024-12-06'))
      )
      .groupBy('pay_full_flag', 'payment_make_flag', 'L2_combine')
      .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             , f.sum('total_amt')
             , f.avg('days_early')
             )
)

# COMMAND ----------

# DBTITLE 1,control pay full %
display(
      df_audience_control
      .join(
            df_audience_payment
            .select('fs_acct_id', 'min_payment_date', 'total_amt')
            , ['fs_acct_id'], 'left'
      )
      .withColumn(
            'payment_make_flag'
            , f.when(
                  f.col('total_amt').isNotNull()
                  , 1
            )
            .when(
                  f.col('total_amt').isNull()
                  , 0
            )
            .otherwise(-999)
      )
      .withColumn(
            'pay_full_flag'
            , f.when( 
                  (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') <= f.abs('total_amt'))
                  , 'Y'
            )
            .when (
                  (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') > f.abs('total_amt'))
                  , 'N'
            )
            .otherwise('U')
      )
      .withColumn(
            'days_early'
            , f.date_diff('min_payment_date', f.lit('2024-11-18'))
      )
      .groupBy('pay_full_flag', 'payment_make_flag', 'L2_combine')
      .agg(
            f.countDistinct('fs_acct_id')
            , f.count('*')
            , f.sum('total_amt')
            , f.avg('days_early')
      )
)

# COMMAND ----------

# DBTITLE 1,check sample - treatment
display(df_treatment_v1
        .join(df_audience_payment
              .select('fs_acct_id', 'min_payment_date', 'total_amt')
              , ['fs_acct_id'], 'left')
        .withColumn('payment_make_flag'
                    , f.when(f.col('total_amt').isNotNull()
                             , 1
                            )
                       .when(f.col('total_amt').isNull()
                             , 0
                            )
                       .otherwise(-999)
                  )
        .withColumn('pay_full_flag', f.when( (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') <= f.abs('total_amt'))
                                             , 'Y'
                                            )
                                      .when ((f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') > f.abs('total_amt'))
                                             , 'N'
                                      )
                                      .otherwise('U')
        )
      #   .groupBy('pay_full_flag', 'payment_make_flag', 'L2_combine', 'min_payment_date' )
      #   .agg(f.countDistinct('fs_acct_id')
      #        , f.count('*')
      #        , f.sum('total_amt')
            # , f.avg('days_early')
             #)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### s04 remove pay 1 day before 

# COMMAND ----------

# DBTITLE 1,new bill base
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

# DBTITLE 1,export new bill base
(
 df_bill_base
 .write
 .mode('overwrite')
 .format('delta')
 .option('header', True)
 .save('/mnt/ml-lab/dev_users/dev_sc/audience_list/bill_base_20241209')
)

# COMMAND ----------

df_bill_base_dl = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/audience_list/bill_base_20241209')

# COMMAND ----------

display(
  df_bill_base_dl
  .filter(f.col('fs_acct_id') == '510484699')
)

# COMMAND ----------

df_pay_t = (
  df_bill_base_dl
  .filter(f.col('bill_due_date') == '2024-12-06')
  .filter(f.col('bill_close_date') <='2024-12-05') 
  .filter(f.col('bill_close_t') != 0 )
  .join(df_treatment_v1, ['fs_acct_id'], 'inner')
)

# COMMAND ----------

display(df_pay_t.count())

# COMMAND ----------

df_pay_c = (
  df_bill_base_dl
  .filter(f.col('bill_due_date') == '2024-12-06')
  .filter(f.col('bill_close_date') <='2024-12-05') 
  .filter(f.col('bill_close_t') != 0 )
  .join(df_audience_control, ['fs_acct_id'], 'inner')
)

# COMMAND ----------

display(df_treatment_v1
        .join(df_audience_payment
              .select('fs_acct_id', 'min_payment_date', 'total_amt')
              , ['fs_acct_id'], 'left')
        .withColumn('payment_make_flag'
                    , f.when(f.col('total_amt').isNotNull()
                             , 1
                            )
                       .when(f.col('total_amt').isNull()
                             , 0
                            )
                       .otherwise(-999)
                  )
        .withColumn('pay_full_flag', f.when( (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') <= f.abs('total_amt'))
                                             , 'Y'
                                            )
                                      .when ((f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') > f.abs('total_amt'))
                                             , 'N'
                                      )
                                      .otherwise('U')
        )
        .join(df_pay_t, ['fs_acct_id'], 'anti')
)

# COMMAND ----------

display(
  df_bill_base
  .filter(f.col('fs_acct_id') == '510484699')
)

# COMMAND ----------

display(
      df_audience_control
      .join(
            df_audience_payment
            .select('fs_acct_id', 'min_payment_date', 'total_amt')
            , ['fs_acct_id'], 'left'
      )
      .withColumn(
            'payment_make_flag'
            , f.when(
                  f.col('total_amt').isNotNull()
                  , 1
            )
            .when(
                  f.col('total_amt').isNull()
                  , 0
            )
            .otherwise(-999)
      )
      .withColumn(
            'pay_full_flag'
            , f.when( 
                  (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') <= f.abs('total_amt'))
                  , 'Y'
            )
            .when(
                  (f.col('payment_make_flag') == 1 ) & ( f.abs('total_due') > f.abs('total_amt'))
                  , 'N'
            )
            .otherwise('U')
      )
      .join(df_pay_c, ['fs_acct_id'], 'anti')
)
