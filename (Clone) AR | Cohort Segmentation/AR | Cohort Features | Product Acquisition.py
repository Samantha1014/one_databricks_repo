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

# COMMAND ----------

# MAGIC %run "./utility_functions"

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

# MAGIC %md
# MAGIC ##### directory

# COMMAND ----------

dir_edw_data_parent = "/mnt/prod_edw/raw/cdc"
dir_brm_data_parent = "/mnt/prod_brm/raw/cdc"
dir_fs_data_parent = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer"
dir_fs_data_stg = '/mnt/feature-store-prod-lab/d200_staging/d299_src'
dir_fs_data_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/'
dir_fs_data_srvc = '/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/'

# COMMAND ----------

# MAGIC %md
# MAGIC #### load data

# COMMAND ----------

# DBTITLE 1,oa base
df_master = spark.read.format('delta').load(os.path.join(dir_fs_data_srvc,'reporting_cycle_type=calendar cycle'))

# COMMAND ----------

# DBTITLE 1,bb base
df_bb_base = (
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
            , billing_account_number as fs_acct_id
            , c.customer_source_id as bb_fs_cust_id
            , service_access_type_name
            , s.proposition_product_name
            , s.plan_name
            , s.service_start_date_time
            , s.service_first_activation_date
    from prod_pdb_masked.modelled.f_service_activity_daily_snapshot f 
    inner join prod_pdb_masked.modelled.d_service s on f.d_service_key = s.d_service_key 
    inner join prod_pdb_masked.modelled.d_billing_account b on b.d_billing_account_key = s.d_billing_account_key 
    inner join prod_pdb_masked.modelled.d_customer c on c.d_customer_key = b.d_customer_key
    where  
            s.service_type_name in ('Broadband')
            and c.market_segment_name in ('Consumer')
            and f.d_snapshot_date_key in ('20240131', '20240229', '20240331')
            and 
            service_access_type_name not in ('LLR')
            and 
            ( 
            (service_access_type_name is not null)
            
            or 
            ( s.service_type_name in ('Broadband')
            and service_access_type_name is null   
            and s.proposition_product_name in ('home phone wireless broadband discount proposition'
            , 'home phone plus broadband discount proposition')
            )
            ) """
    ).load()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### parameter

# COMMAND ----------

ls_prm_key = ['fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
#ls_product_join_key = ['fs_acct_id']
ls_reporting_date_key = ['reporting_date', 'reporting_cycle_type']

# COMMAND ----------

vt_reporting_date = '2024-01-31'

# COMMAND ----------

# DBTITLE 1,product base
df_product_base = (
    df_master
        .filter(f.col('reporting_date') == f.lit(vt_reporting_date))
        .filter(f.col('reporting_cycle_type') == 'calendar cycle') 
        .select( *ls_prm_key
                , *ls_reporting_date_key
                , 'num_of_active_srvc_cnt'
                , 'num_of_active_acct_cnt'
                , 'srvc_start_date'
                , 'billing_acct_open_date'
                , 'cust_start_date'
                , 'first_activation_date'
                , 'plan_name'
                , 'ifp_prm_dvc_term_start_date'
                , 'ifp_prm_dvc_flag'
                , 'ifp_acct_accs_flag'
                , 'ifp_acct_accs_cnt'
                , 'ifp_acct_dvc_cnt'
        )
)

# COMMAND ----------

display(df_bb_base
        .limit(10)
        )

# COMMAND ----------

df_bb_base_acct = (
    df_bb_base
    .withColumn('index', f.row_number().over(
                        Window
                        .partitionBy('fs_acct_id', 'bb_fs_cust_id', 'reporting_date')
                        .orderBy(f.desc('SERVICE_START_DATE_TIME'))
                        )
                ) ## latest bb service start time if multi service per billing account 
    .filter(f.col('index') == 1)
    .drop('index')
)

# COMMAND ----------

# display(df_bb_base_acct
#         .groupBy('reporting_date')
#         .agg(f.countDistinct('fs_acct_id')
#              , f.count('*')
#              , f.countDistinct('bb_fs_srvc_id')
#              , f.countDistinct('bb_fs_cust_id')
#              )
#         )


# COMMAND ----------

# MAGIC %md
# MAGIC ##### product holding & recent acquisition

# COMMAND ----------

df_output_curr = (
     df_product_base
        .join(df_bb_base_acct
              .filter(f.col('reporting_date') == vt_reporting_date )
              , ['fs_acct_id', 'reporting_date'], 'left'
               )
        .withColumn('product_holding'
                    , f.when(f.col('PROPOSITION_PRODUCT_NAME').isNull() 
                    , 'OA only')
                       .otherwise(f.lit('OA+BB'))
        )
        .withColumn('recent_acq_6mp'
                     , f.when( f.col('SERVICE_START_DATE_TIME')
                                .between( f.add_months(f.col('reporting_date'), -6), f.col('reporting_date'))
                                ,f.lit('Y')
                     )
                     .when(f.col('ifp_prm_dvc_term_start_date')
                            .between(f.add_months(f.col('reporting_date'), -6), f.col('reporting_date'))
                            , f.lit('Y')
                           )
                     .when(f.col('srvc_start_date')
                            .between(f.add_months(f.col('reporting_date'), -6), f.col('reporting_date'))
                            , f.lit('Y')
                          )
                     .otherwise('N')
       )
      .withColumn('bb_add_in_6mp'
                   ,  f.when( f.col('SERVICE_START_DATE_TIME')
                               .between( f.add_months(f.col('reporting_date'), -6), f.col('reporting_date'))
                               , 1
                      )
                      .otherwise(0)
     )
     .withColumn( 'oa_plan_add_in_6mp'
                   ,  f.when(f.col('srvc_start_date')
                              .between(f.add_months(f.col('reporting_date'), -6), f.col('reporting_date'))
                              ,1
                      )
                       .otherwise(0)
     )
     .withColumn('ifp_add_in_6mp'
                  ,  f.when(f.col('ifp_prm_dvc_term_start_date')
                             .between(f.add_months(f.col('reporting_date'), -6), f.col('reporting_date'))
                             ,1
                     )
                  .otherwise(0)
     )
     .withColumn('cnt_product_add_in_6mp'
                , f.col('oa_plan_add_in_6mp') + f.col('ifp_add_in_6mp') + f.col('bb_add_in_6mp')
                )
     .select('fs_acct_id'
             , 'reporting_date'
             , 'reporting_cycle_type'
             , 'fs_cust_id'
             , 'fs_srvc_id'
             , 'num_of_active_srvc_cnt'
             , 'ifp_prm_dvc_flag'
             , 'ifp_acct_accs_flag'
             , 'ifp_acct_accs_cnt'
             , 'ifp_acct_dvc_cnt'
             , 'product_holding'
             , 'recent_acq_6mp'
             , 'cnt_product_add_in_6mp'
             )
        )

# COMMAND ----------

display(df_output_curr.limit(10))

# COMMAND ----------

display(df_output_curr.count())

# COMMAND ----------

display(df_product_base.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### export data

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

dbutils.fs.ls('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/')

# COMMAND ----------

export_data(
            df = df_output_curr
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_product_acq'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_product_acq')

# COMMAND ----------

display(df_test
        .groupBy('reporting_date')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

display(
    df_test
    .groupBy(
             'reporting_date'
             ,'product_holding'
             , 'ifp_prm_dvc_flag'
             , 'recent_acq_6mp'
             , 'cnt_product_add_in_6mp'
             )
    .agg(f.countDistinct('fs_acct_id'))
)
