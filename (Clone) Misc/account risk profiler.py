# Databricks notebook source
import pyspark
from pyspark.sql import functions as f
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp, date_format
from pyspark.sql.functions import regexp_replace 

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer')

# COMMAND ----------

df_fs_master = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer')

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/reporting_cycle_type=calendar cycle/')

# COMMAND ----------

display(df_fs_master
        .select('fs_cust_id', 'fs_acct_id', 'fs_srvc_id', 'payment_auto_flag_6cycle', 'reporting_date' )
        .filter(f.col('reporting_cycle_type') == 'rolling cycle')
       #  .filter(f.col('active_flag') == 'Y')
        .groupBy('payment_auto_flag_6cycle', 'reporting_date')
        .agg(f.count('*'))
        )

# COMMAND ----------

dbutils.fs.ls('/mnt/prod_brm/raw/')

# COMMAND ----------

# MAGIC %run "./Function"

# COMMAND ----------

# source table directory 
# dir_item_t_test = 'dbfs:/mnt/ml-lab/dev_users/dev_sc/RAW_PINPAP_ITEM_T' # item_t after partition 
dir_item_t = '/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T' # raw pinpap item t 
#dir_oa_consumer = 'dbfs:/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=calendar cycle'
dir_acct = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T'
dir_event_bill_payment_t = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T'
dir_event_bal_impact_t = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T'
dir_config_pay_type = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_PAY_TYPES_T'
dir_config_channel_map = 'dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_CHANNEL_MAP_T'
dir_prim_oa = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'


# COMMAND ----------

dbutils.fs.ls('dbfs:/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T')

# COMMAND ----------


df_prim_oa = spark.read.format('delta').load(dir_prim_oa)
df_prim_oa_01 = (
    df_prim_oa
    .select('fs_acct_id')
    .filter(col('reporting_date')>='2021-01-01')
    .distinct()
)

# COMMAND ----------

# df_oa_consumer = spark.read.format('delta').load(dir_oa_consumer)
# df_bill_t = spark.read.format('delta').load(dir_bill_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_item_t = spark.read.format('delta').load(dir_item_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_account_t = spark.read.format('delta').load(dir_acct).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_event_bal_impact_t = spark.read.format('delta').load(dir_event_bal_impact_t)
df_event_bill_payment_t = spark.read.format('delta').load(dir_event_bill_payment_t).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_config_pay_type = spark.read.format('delta').load(dir_config_pay_type).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))
df_config_channel_map = spark.read.format('delta').load(dir_config_channel_map).filter((f.col('_is_latest') ==1) & (f.col('_is_deleted') ==0))

# COMMAND ----------

display(df_event_bal_impact_t
        .filter(f.col('ITEM_OBJ_TYPE').isin('/item/payment'))
        .filter(f.col('account_obj_id0') == '1947294679694'))
        

# COMMAND ----------

display(df_event_bal_impact_t
        .filter(f.col('item_obj_type') == '/item/payment')
        # .filter(f.col(''))
        )

# COMMAND ----------

display(df_item_t.count())

# COMMAND ----------

df_account_t = (
    df_account_t
    .filter(col('_is_latest') ==1) # pick latest record
    .filter(col('_is_deleted') ==0) 
    .filter(~col('account_no').startswith('S')) # remove subscription account 
    .withColumn('account_no', f.regexp_replace("account_no", "^0+", "")) # remove leading 0 in account_t 
    .select('account_no', 'poid_id0')
)


# COMMAND ----------

ls_col_payment = ['POID_ID0', 'POID_TYPE', 'CREATED_T', 'MOD_T', 'ACCOUNT_OBJ_ID0', 'EFFECTIVE_T', 'ITEM_NO',  'ITEM_TOTAL', 'BILLINFO_OBJ_ID0']
df_payment = (
    df_item_t
    .select(ls_col_payment)
    .filter(f.col('poid_type').isin('/item/payment', '/item/adjustment'))
)


# COMMAND ----------



# COMMAND ----------

df_oa_payment =  (df_payment.alias('a')
        .join(df_account_t.alias('b'), col('a.account_obj_id0') == col('b.poid_id0'), 'inner' )
        .join(df_prim_oa_01, col('account_no') == col('fs_acct_id'), 'right')
        .drop('account_no')
) 

# 50,092,410 rows 
# 743,383 (inner) vs. 759457 (right)

# COMMAND ----------

ls_time_convert = [c for c in df_payment.columns if c.endswith('_T')]
#ls_time_convert
df_payment = to_nzt(df_payment, ls_time_convert)

# COMMAND ----------

display(df_payment
        .filter(f.col('POID_TYPE') == '/item/payment')
        .groupBy(f.date_format('created_t', 'yyyy-MM').alias('created_month'))
        .agg(f.count('poid_id0'))
        )

# COMMAND ----------

ls_col_impact = ['OBJ_ID0', 'ACCOUNT_OBJ_ID0', 'AMOUNT', 'ITEM_OBJ_ID0', 'ITEM_OBJ_TYPE']

df_bal_impact  = (
    df_event_bal_impact_t
    .filter(f.col('ITEM_OBJ_TYPE').isin('/item/payment'))
    .select(ls_col_impact)
)


ls_col_billpay = ['OBJ_ID0', 'AMOUNT', 'TRANS_ID', 'PAY_TYPE', 'CHANNEL_ID', 'ACCOUNT_NO']
df_bill_payment = (
df_event_bill_payment_t
.select(ls_col_billpay)
)



# COMMAND ----------

ls_col_pay_type = ['REC_ID','PAYINFO_TYPE', 'PAYMENT_EVENT_TYPE']
df_config_pay_type = df_config_pay_type.select(ls_col_pay_type)


ls_col_channel_map = ['CHANNEL_ID','SOURCE','SUBTYPE','TYPE']
df_config_channel_map = df_config_channel_map.select(ls_col_channel_map)

# COMMAND ----------

ls_col = ['aa.ACCOUNT_NO', 'a.ACCOUNT_OBJ_ID0' ,'a.POID_ID0', 'a.POID_TYPE', 'CREATED_T', 'MOD_T', 'EFFECTIVE_T','ITEM_NO' ,
          'a.ITEM_TOTAL', 'PAY_TYPE', 'c.CHANNEL_ID',  'PAYMENT_EVENT_TYPE', 'SOURCE','SUBTYPE', 'TYPE']


df_payment_base = (df_payment.alias('a')
    .join(df_account_t.alias('aa'), f.col('a.account_obj_id0') == f.col('aa.poid_id0'), 'inner')
    .join(df_bal_impact.alias('b'),f.col('a.poid_id0') ==f.col('b.item_obj_id0'),'left')
    .join(df_bill_payment.alias('c'), f.col('b.obj_id0') ==f.col('c.obj_id0'),'left')
    .join(df_config_pay_type.alias('pt'), f.col('c.pay_type')==f.col('pt.rec_id'),'left')
    .join(df_config_channel_map.alias('cm'), f.col('c.channel_id')==f.col('cm.channel_id'), 'left')
    .select(ls_col)
    .withColumn('Created_month', date_format('created_t', 'yyyy-MM'))
)

# COMMAND ----------

display(df_payment.alias('a')
    .join(df_account_t.alias('aa'), f.col('a.account_obj_id0') == f.col('aa.poid_id0'), 'inner')
    .join(df_bal_impact.alias('b'),f.col('a.poid_id0') ==f.col('b.item_obj_id0'),'left')
    .groupBy(f.date_format('CREATED_T', 'yyyy-MM').alias('created_month') , 
             )
    .agg(f.countDistinct('a.POID_ID0'), f.countDistinct('b.item_obj_id0')
         , f.countDistinct('OBJ_ID0')
         )
    )

# COMMAND ----------

std_params = {
    'col_map': {
        'account_no': 'fs_acct_id'
        , 'account_obj_id0': 'bill_account_obj_id0'
        , 'poid_id0': 'item_poid_id0'
        , 'poid_type': 'item_poid_type'
        , 'created_t' : 'rec_created_dttm'
        , 'mod_t': 'rec_mod_dttm'
        , 'effective_t': 'rec_effective_dttm'
        , 'item_no': 'item_no'
        , 'item_total': 'item_amount'
        , 'pay_type': 'pay_type'
        , 'channel_id': 'payment_channel_id'
        , 'payment_event_type': 'payment_event_type'
        , 'source': 'payment_source'
        , 'subtype': 'payment_subtype'
        , 'type': 'payment_type'
        , 'created_month': 'rec_created_month'
    }
}

ls_param_col_map = std_params['col_map']

# COMMAND ----------

df_payment_base = (lower_col_names(df_payment_base))

# COMMAND ----------

df_output_curr = (
    df_payment_base
    .select([f.col(c).alias(ls_param_col_map.get(c, c)) for c in df_payment_base.columns])
)

# COMMAND ----------

dir_oa_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=calendar cycle'
df_oa_prim = spark.read.format('delta').load(dir_oa_prm)

# COMMAND ----------

df_oa_prim_01 = (
    df_oa_prim
    .select('fs_acct_id')
    .filter(f.col('reporting_date')>='2021-01-01')
    .distinct()
)


# COMMAND ----------

ls_param_payment_joining_keys = ['fs_acct_id']
ls_param_payment_int_fields = [
'fs_acct_id', 
'item_id',
'rec_created_dttm', 
'rec_mod_dttm',
'rec_effective_dttm',
'item_amount',
'rec_created_month',
'onenz_auto_pay',
'payment_method',
'item_category'] 


# COMMAND ----------

# inner join with OA consumer to narrow down 
df_payment_int = (
    df_output_curr
    .join(df_oa_prim_01, ls_param_payment_joining_keys, 'inner')
    .withColumn(
        'onenz_auto_pay'
        , f.when(f.col('payment_event_type').isin('/event/billing/payment/dd', '/event/billing/payment/cc' ), 'Y')
        .otherwise('N')
    )
    #.withColumn('payment_method', element_at(split(f.col('payment_event_type'), '/'), size(split(f.col('payment_event_type'), '/'))))
    # .withColumn(
    #     'item_category'
    #     , f.when(f.col('item_poid_type')=='/item/adjustment', f.lit('adjustment'))
    #     .when( f.col('payment_event_type') == '/event/billing/payment/failed', f.lit('fail payment'))
    #             #.when( f.col('item_poid_type').isNull(), f.lit('no payment'))
    #     .when( f.col('item_poid_type') == '/item/payment' , f.lit('payment'))
    #     .otherwise(f.lit('other'))
    # )
    #.withColumnRenamed('item_poid_id0', 'item_id')
    #.select(ls_param_payment_int_fields)
         #.fillna(value = 'other', subset = ['payment_method'])
         # use right join since there is never payer 
)

# COMMAND ----------

display(df_payment_int
        .groupBy('onenz_auto_pay', 'rec_created_month')
        .agg(f.count('*'))
        )

# COMMAND ----------


