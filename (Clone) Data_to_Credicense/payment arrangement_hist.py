# Databricks notebook source
# MAGIC %md
# MAGIC ### S01 Setup 

# COMMAND ----------

import pyspark
import os
from pyspark.sql import functions as f

# COMMAND ----------

# MAGIC %md
# MAGIC ### S02 Load Data

# COMMAND ----------

dir_prod_brm = 'dbfs:/mnt/prod_brm/raw/cdc'
dir_prod_siebel = 'dbfs:/mnt/prod_siebel/raw/cdc'
df_sbl_s_orgext = spark.read.format('delta').load(os.path.join(dir_prod_siebel, 'RAW_SBL_ORG_EXT'))

# COMMAND ----------

vt_days_back_hist = 180
# vt_days_back_daily = 1

# COMMAND ----------

df_payment_arrange = spark.sql("""
 WITH base AS (
        SELECT 
            acct.account_no
            , ca.created_t as arrangement_t
            , to_date(from_utc_timestamp(from_unixtime(ca.created_t), 'Pacific/Auckland')) as arrangement_date
            , bi.pending_recv AS old_balance
            , cpm.amount AS installment_amount
            ,cpm.descr AS arrangement_type
        FROM DELTA.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_COLLECTIONS_ACTION_T`  ca
           INNER JOIN  DELTA.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_COLLECTIONS_SCENARIO_T`  cs 
            ON ca.scenario_obj_id0 = cs.poid_id0
            AND cs._is_latest = 1 
            AND  cs._is_deleted = 0
            INNER JOIN DELTA.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T` acct 
            ON ca.account_obj_id0 = acct.poid_id0 
            AND acct._is_latest = 1 
            AND acct._is_deleted = 0
            INNER JOIN DELTA.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_VF_COLL_P2P_MILESTONES_T` cpm
            ON cpm.action_obj_id0 = ca.poid_id0 
            AND cpm._is_latest = 1 
            AND cpm._is_deleted = 0
            INNER JOIN DELTA.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILLINFO_T` bi 
            ON bi.account_obj_id0 = acct.poid_id0
            AND bi._is_latest = 1 
            AND bi._is_deleted = 0 
        WHERE 
             ca.poid_type = '/collections_action/promise_to_pay'
            AND ca._is_latest = 1 
            AND ca._is_deleted = 0 
	        )
	SELECT * FROM base
        WHERE arrangement_t >= unix_timestamp(date_sub(current_date(),{0}))
                            """.format(vt_days_back_hist))

# COMMAND ----------

# MAGIC %md
# MAGIC ### S03 Development

# COMMAND ----------

# DBTITLE 1,aggregate payment plans
## payment plan raw data aggregate into requested format 
## ADO link - https://dev.azure.com/vodafonenz/IT/_workitems/edit/308826
## 1st Upload will be for 180days based on Transaction Date followed by daily upload as incremental
 
df_payment_arrange_agg = (
        df_payment_arrange
        .select('account_no'
                , 'arrangement_date'
                , 'arrangement_type'
               #  , f.round('balance',2).alias('balance')
                , f.round('installment_amount',2).alias('installment_amount')
                )
        .groupBy('account_no'
                 , 'arrangement_date'
                 , 'arrangement_type'
                 )
        .agg(
              f.sum('installment_amount').alias('balance') # tamas agree that using sum of installment amount as the proxy of balance 
             ,f.count('*').alias('no_of_installment')
             )
        )

# COMMAND ----------

# DBTITLE 1,get siebel customer id
df_cust_id = (
    df_sbl_s_orgext
    .filter(f.col('_is_latest') == 1)
    # .filter(f.col('_is_deleted') ==0)
    .select(f.col('ou_num').alias('account_no')
            , f.col('par_ou_id').alias('customer_id')
            # , f.col('row_id').alias('billing_acct_id')
           # , 'accnt_type_cd'
            )
    .filter(f.col('accnt_type_cd').isin('Billing'))
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ### S04 Output

# COMMAND ----------

df_output = (
    df_payment_arrange_agg
    .join(df_cust_id, ['account_no'], 'left')   
)


display(df_output)

# COMMAND ----------

display(df_output.count())
