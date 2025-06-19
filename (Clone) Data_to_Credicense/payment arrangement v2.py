# Databricks notebook source


# COMMAND ----------

password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")
options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "prod_edw",
  "sfSchema": "raw",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}

# COMMAND ----------

df_payment_arrange = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
    WITH base AS (
        SELECT 
            acct.account_no
            , ca.created_t as arrangement_t
           -- , ca.due_t as installment_due_t
            , bi.pending_recv AS balance
            , cpm.amount AS installment_amount
            ,cpm.descr AS arrangement_type
    
        FROM PROD_BRM.RAW.PINPAP_COLLECTIONS_ACTION_T ca
            INNER JOIN PROD_BRM.RAW.PINPAP_COLLECTIONS_SCENARIO_T  cs 
            ON ca.scenario_obj_id0 = cs.poid_id0
            AND cs._is_latest = 1 
            AND  cs._is_deleted = 0
            INNER JOIN PROD_BRM.RAW.PINPAP_ACCOUNT_T acct 
            ON ca.account_obj_id0 = acct.poid_id0 
            AND acct._is_latest = 1 
            AND acct._is_deleted = 0
            INNER JOIN PROD_BRM.RAW.PINPAP_VF_COLL_P2P_MILESTONES_T cpm
            ON cpm.action_obj_id0 = ca.poid_id0 
            AND cpm._is_latest = 1 
            AND cpm._is_deleted = 0
            INNER JOIN PROD_BRM.RAW.PINPAP_BILLINFO_T bi 
            ON bi.account_obj_id0 = acct.poid_id0
            AND bi._is_latest = 1 
            AND bi._is_deleted = 0 
        WHERE arrangement_t >= (date_part(epoch_second, current_timestamp()) - 60*60*24*180)  --'2023-10-01' 
            AND ca.poid_type = '/collections_action/promise_to_pay'
            AND ca._is_latest = 1 
            AND ca._is_deleted = 0 
	)
	SELECT * FROM base; 
         """
    )
    .load()
)

# cnt = 40125

# COMMAND ----------

## payment plan raw data aggregate into requested format 
## ADO link - https://dev.azure.com/vodafonenz/IT/_workitems/edit/318883
## 1st Upload will be for 180days based on Transaction Date followed by daily upload as incremental
 
df_payment_arrange_agg = (
        df_payment_arrange
        .withColumn('arrangement_date', 
                    f.date_format(f.from_utc_timestamp(
                        f.from_unixtime(f.col('arrangement_t')), "Pacific/Auckland"
                                        ) , 'yyyy-MM-dd'
                                  )
                    ) # convert to nzt to match with brm data 
        .select('account_no'
                , 'arrangement_date'
                , 'arrangement_type'
                , f.round('balance',2).alias('balance')
                , f.round('installment_amount',2).alias('installment_amount')
                )
        .groupBy('account_no'
                 , 'arrangement_date'
                 , 'arrangement_type'
                 , 'balance')
        .agg(
              f.sum('installment_amount').alias('arrangement_total')
             ,f.count('*').alias('no_of_installment')
             )
        .filter(f.col('arrangement_date') )
        )
