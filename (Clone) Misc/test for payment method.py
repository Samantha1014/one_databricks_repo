# Databricks notebook source
import pyspark
from pyspark.sql import functions as f

# COMMAND ----------

dbutils.fs.ls('/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T_PAYMENT')

# COMMAND ----------

df_payment = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_payment_method_cycle_rolling_6')

# COMMAND ----------

dbutils.fs.ls('/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_PAY_TYPES_T')
#RAW_PINPAP_CONFIG_PAYMENT_CHANNEL_MAP_T

# COMMAND ----------

df_congif_payment_type = spark.read.format('delta').load('/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_PAY_TYPES_T')

df_config_payment_channel = spark.read.format('delta').load('/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_CHANNEL_MAP_T')

df_ebit = spark.read.format('delta').load('/mnt/prod_brm/raw/cdc/')

# COMMAND ----------

display(df_config_payment_channel)

# COMMAND ----------

display(df_congif_payment_type)

# COMMAND ----------

display(df_payment
        .filter(f.col('fs_acct_id') == '478958168')
        .filter(f.col('reporting_cycle_type') == 'rolling cycle')
        )

# COMMAND ----------

df_pay_march = (df_payment
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select('payment_method_main_type_6cycle', 'fs_acct_id', 'fs_cust_id', 'reporting_date')
        .filter(  ((f.col('payment_method_main_type_6cycle') == 'external') & 
                 (f.col('reporting_date') == '2024-03-31') ) 
                )
        .distinct()
        )

# COMMAND ----------

df_payment_june = (df_payment
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select('payment_method_main_type_6cycle', 'fs_acct_id', 'fs_cust_id', 'reporting_date')
        .filter(  ((f.col('payment_method_main_type_6cycle') == 'misc') & 
                 (f.col('reporting_date') == '2024-06-30') ) 
                )
        .distinct()
        )

# COMMAND ----------

df_ebit = spark.read.format('delta').load('/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T_PAYMENT')
df_ebpt = spark.read.format('delta').load('/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T')

# COMMAND ----------

display(
    df_ebit
    .filter(f.col('_is_latest') == 1 )
    .filter(f.col('item_obj_type') == '/item/payment')
    .filter(f.col('account_obj_id0') == '1888467740711')      
)

# COMMAND ----------

display(
        df_ebpt.filter(f.col('account_no') == '480973241')
)

# COMMAND ----------

df_test = spark.sql("""
SELECT ebit.ACCOUNT_OBJ_ID0
, ebpt.pay_type
, ebpt.channel_id
, ebpt.account_no
, item_t.created_t
,   FROM_UTC_TIMESTAMP(FROM_UNIXTIME(item_t.created_t), 'Pacific/Auckland') as created_nz_time
FROM delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T_PAYMENT`  ebit
INNER JOIN delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T` ebpt ON ebit.obj_id0 = ebpt.obj_id0 
INNER JOIN delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T` item_t ON item_t.poid_id0 = ebit.item_obj_id0
WHERE item_obj_type IN ('/item/payment')
and item_t._is_latest = 1
and ebpt._is_latest = 1
and ebit._is_latest =1 
AND ebpt.account_no IN ('480973241')
ORDER BY created_t DESC;
""")


# COMMAND ----------

display(df_test)

# COMMAND ----------

df_ebpt = spark.sql("""
SELECT *
from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T` ebpt 
where  1=1  
--ebpt._is_latest = 1
AND ebpt.account_no IN ('480973241')
""")

# COMMAND ----------

display(df_ebpt)

# COMMAND ----------

df_ebit = spark.sql("""
SELECT *
from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T` ebit 
where  
1= 1
--and ebit._is_latest = 1
and item_obj_type IN ('/item/payment')
AND ebit.account_obj_id0 IN ('1888467740711')
""")

# COMMAND ----------

df_ebit_test = spark.sql("""
SELECT *
from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T` ebit 
where  
 ebit.obj_id0 in ('19587958901609')
""")

# COMMAND ----------

display(df_ebit)

# COMMAND ----------

display(df_ebit_test)

# COMMAND ----------

display(
    df_payment_june
    .join(df_pay_march, ['fs_cust_id', 'fs_acct_id'], 'inner')
)

# COMMAND ----------

display(df_payment.limit(3))

# COMMAND ----------



# COMMAND ----------

display(df_payment
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .select('payment_method_main_type_6cycle', 'fs_acct_id', 'fs_srvc_id', 'fs_cust_id', 'reporting_date')
        .groupBy('payment_method_main_type_6cycle', 'reporting_date')
        .agg(f.count('*'))
        )

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/')

# COMMAND ----------

df_base = spark.read.format('delta').load(
    '/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base'
)

# COMMAND ----------

display(df_base 
        .select('fs_acct_id', 'reporting_date', 'reporting_cycle_type', 'fs_srvc_id', 'fs_cust_id')
        .filter(f.col('fs_acct_id') == '505589183')
        )

# COMMAND ----------

display(
    df_base
    .filter(f.col('reporting_cycle_type') == 'rolling cycle')
    .select('fs_acct_id', 'fs_srvc_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type')
    .groupBy('reporting_date', 'reporting_cycle_type')
    .agg(f.count('*'))
)

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/')

# COMMAND ----------

display(
    df_test
    .filter(f.col('reporting_cycle_type') == 'rolling cycle')
    .select('fs_acct_id', 'fs_srvc_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type')
    .groupBy('reporting_date', 'reporting_cycle_type')
    .agg(f.count('*'))
)
