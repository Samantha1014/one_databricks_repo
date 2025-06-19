# Databricks notebook source
# MAGIC %md
# MAGIC ### S01 Set up 

# COMMAND ----------

from pyspark.sql import functions as f

# COMMAND ----------

# MAGIC %md
# MAGIC ### S02 Data Load

# COMMAND ----------

df_prm_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_unit_base/reporting_cycle_type=rolling cycle')

# COMMAND ----------

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

df_bb_base = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """ 
        select 
                d_snapshot_date_key
                ,s.service_id as fs_srvc_id
                , billing_account_number as fs_acct_id
                , c.customer_source_id as fs_cust_id
                , service_access_type_name
                , s.proposition_product_name
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
                and f.d_snapshot_date_key in ('20240507'); 
         """
    )
    .load()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### S03 Development 

# COMMAND ----------

# DBTITLE 1,Check BB Base
display(df_bb_base.limit(100))
display(df_bb_base
        .groupBy('service_access_type_name')
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('FS_SRVC_ID')
             , f.count('*')
             )
        )

## compare with CAR 
##
## select BB_PLAN_NAME, count(1) from Staging.dbo.staging_fix_merge_car_daily
## group by bb_plan_name;
## check count with this PBI report https://app.powerbi.com/groups/988cd51e-9a82-4a79-b12a-cb47c3f7c0ba/reports/3fb990c8-357f-4769-a274-5992bcc66882/ReportSectione3a516fb90c0615a03c3?experience=power-bi

# COMMAND ----------

# DBTITLE 1,get llatest oa reporting date
## get latest oa reporting date 
display(df_prm_oa
        .agg(f.max('reporting_date'))
        )

# COMMAND ----------

# DBTITLE 1,oa base filter
df_oa_base = (df_prm_oa
        .select('fs_acct_id', 'fs_cust_id', 'fs_srvc_id', 'reporting_date')
        .distinct()
        .filter(f.col('reporting_date') == '2024-05-05')
        )

# COMMAND ----------

# DBTITLE 1,check oa base
display(df_oa_base.count())
display(df_oa_base.limit(100))

# COMMAND ----------

# DBTITLE 1,anti join oa msisdn field

## anti on  converage on oa misdin 
df_oa_list_layer01 = (df_oa_base.alias('a')
        .join(df_bb_base.alias('b'), 
         f.col('a.fs_srvc_id') == f.col('CONVERGED_OA_MSISDN'), 'anti')
        )

# COMMAND ----------

# converge on same account number 

df_oa_converge = (
    df_oa_base
    .join(df_bb_base, ['fs_acct_id', 'fs_cust_id'], 'inner')
    #.count()
)

display(df_oa_converge.count())

# COMMAND ----------

# DBTITLE 1,anti join account level converge
df_oa_list_layer02 = (
    df_oa_list_layer01
    .join(df_oa_converge, ['fs_acct_id', 'fs_cust_id'], 'anti')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### S04 Output

# COMMAND ----------

df_output = (df_oa_list_layer02)

# COMMAND ----------

# DBTITLE 1,check sample
display(df_output.limit(100))

# COMMAND ----------

# DBTITLE 1,check output
display(df_output
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.countDistinct('fs_cust_id')
             )       
)

