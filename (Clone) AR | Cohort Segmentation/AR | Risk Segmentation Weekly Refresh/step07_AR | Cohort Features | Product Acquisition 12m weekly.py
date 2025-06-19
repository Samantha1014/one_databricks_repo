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

# MAGIC %run "./s99_utility_functions"

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
dir_fs_unit_base = '/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base'

# COMMAND ----------

# MAGIC %md
# MAGIC #### load data

# COMMAND ----------

# DBTITLE 1,oa base
df_master = spark.read.format('delta').load(os.path.join(dir_fs_data_srvc,'reporting_cycle_type=rolling cycle'))
df_fea_ifp = spark.read.format('delta').load(os.path.join(dir_fs_data_parent,'fea_ifp_device_account'))
df_fea_ifp_on_bill = spark.read.format('delta').load(os.path.join(dir_fs_data_parent,'fea_ifp_device_on_bill'))
df_fea_ifp_on_srvc = spark.read.format('delta').load(os.path.join(dir_fs_data_parent,'fea_ifp_device_on_service'))
df_fea_accs = spark.read.format('delta').load(os.path.join(dir_fs_data_parent,'fea_ifp_accessory_account'))
df_prm_ifp_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_bill')
df_prm_ifp_srvc = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_service')

# COMMAND ----------

vt_reporting_cycle_type = 'rolling cycle'
vt_reporting_date = (df_master
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select('reporting_date')
       .agg(f.max('reporting_date'))
       .collect()[0][0]
        )

vt_reporting_date_bb = (df_master
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select('reporting_date')
       .agg(f.max('reporting_date').alias('max_reporting_date'))
       .withColumn('reporting_date', f.date_format('max_reporting_date', 'yyyyMMdd'))
       .select('reporting_date')
       .collect()[0][0]
        )



# COMMAND ----------

bb_query = f""" 
    select 
            TO_DATE(TO_VARCHAR(d_snapshot_date_key), 'YYYYMMDD') as reporting_date
            ,s.service_id as bb_fs_srvc_id
            , billing_account_number as fs_acct_id
            , c.customer_source_id as bb_fs_cust_id
            , service_access_type_name 
            , s.proposition_product_name
            , s.plan_name as bb_plan_name
            , s.service_start_date_time as bb_service_start_date_time
            , s.service_first_activation_date as bb_service_first_activation_date
    from prod_pdb_masked.modelled.f_service_activity_daily_snapshot f 
    inner join prod_pdb_masked.modelled.d_service s on f.d_service_key = s.d_service_key 
    inner join prod_pdb_masked.modelled.d_billing_account b on b.d_billing_account_key = s.d_billing_account_key 
    inner join prod_pdb_masked.modelled.d_customer c on c.d_customer_key = b.d_customer_key
    where  
            s.service_type_name in ('Broadband')
            and c.market_segment_name in ('Consumer')
            and f.d_snapshot_date_key  = '{vt_reporting_date_bb}'
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

# COMMAND ----------

# DBTITLE 1,bb base
df_bb_base = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , bb_query
    ).load()
)

# COMMAND ----------

#display(df_bb_base.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### parameter

# COMMAND ----------

ls_prm_key = ['fs_cust_id', 'fs_srvc_id', 'fs_acct_id']
ls_product_join_key = ['fs_acct_id']
ls_reporting_date_key = ['reporting_date', 'reporting_cycle_type']

# COMMAND ----------

# MAGIC %md
# MAGIC ####IFP ALL

# COMMAND ----------

# DBTITLE 1,ifp_history
# get rid of reporting date boundary 
df_ifp_history= (df_prm_ifp_srvc
        .select(
                         'fs_acct_id'
                        , 'fs_cust_id'
                        ,'fs_ifp_id' 
                        , 'ifp_model'
                        , 'ifp_level'
                        , 'ifp_type'
                        , 'ifp_term_start_date'
                        , 'ifp_term_end_date'
                  )
        .distinct()
        .union(
                df_prm_ifp_bill
                .select(
                        'fs_acct_id'
                        , 'fs_cust_id'
                        ,'fs_ifp_id'
                        , 'ifp_model'
                        , 'ifp_level' 
                        , 'ifp_type'
                        , 'ifp_term_start_date'
                        , 'ifp_term_end_date'
                        )
                .distinct()
        ) ## aggregate into per account level 
        .filter(f.col('ifp_term_start_date') >= f.add_months(f.lit(vt_reporting_date), -36))
        .filter(f.col('ifp_term_start_date') <= vt_reporting_date )
        .filter(f.col('ifp_term_end_date') >= vt_reporting_date)
        )

# COMMAND ----------

# DBTITLE 1,aggregate into account level
df_ifp_history_agg = (df_ifp_history
        # .filter(f.col('ifp_term_start_date') >= f.add_months(f.lit(vt_reporting_date), -12))
        # .filter(f.col('ifp_term_start_date') <= vt_reporting_date )
        # .filter(f.col('ifp_term_end_date') >= vt_reporting_date)
        .groupBy('fs_acct_id', 'fs_cust_id')
        .agg(
            f.countDistinct('fs_ifp_id').alias('ifp_total_cnt_hist')
            , f.sum(f.when(f.col('ifp_type') == 'accessory', 1)
                        .otherwise(0)
                      )
                .alias('ifp_accessory_cnt_hist')
            , f.sum(f.when(f.col('ifp_type') == 'device', 1)
                        .otherwise(0)
                      )
                .alias('ifp_device_cnt_hist')
            , f.min('ifp_term_start_date').alias('min_ifp_start_date_hist')
            , f.max('ifp_term_start_date').alias('max_ifp_start_date_hist')
        )
        #.filter(f.col('fs_acct_id') == '501520730')
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ifp history

# COMMAND ----------

df_ifp_base_36m= (df_prm_ifp_srvc
        #.filter(f.col('ifp_srvc_dvc_flag') == 'Y')
        .filter(f.col('reporting_date') == vt_reporting_date)
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select(*ls_reporting_date_key
                        , 'fs_acct_id'
                        , 'fs_cust_id'
                        ,'fs_ifp_id' 
                        , 'ifp_model'
                        , 'ifp_level'
                        , 'ifp_type'
                        , 'ifp_term_start_date'
                        , 'ifp_term_end_date'
                  )
        .distinct()
        .union(
                df_prm_ifp_bill
                #.filter(f.col('ifp_bill_dvc_flag') == 'Y')
                .filter(f.col('reporting_date') == vt_reporting_date)
                .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
                .select( *ls_reporting_date_key
                        , 'fs_acct_id'
                        , 'fs_cust_id'
                        ,'fs_ifp_id'
                        , 'ifp_model'
                        , 'ifp_level' 
                        , 'ifp_type'
                        , 'ifp_term_start_date'
                        , 'ifp_term_end_date'
                        )
                .distinct()
        ) ## aggregate into per account level 
        .filter(f.col('ifp_term_start_date') >= f.add_months(f.col('reporting_date'), -36))
        .filter(f.col('ifp_term_end_date') >= f.col('reporting_date'))
        .groupBy(*ls_reporting_date_key, 'fs_acct_id', 'fs_cust_id')
        .agg(
            f.countDistinct('fs_ifp_id').alias('ifp_total_cnt_36m')
            , f.sum(f.when(f.col('ifp_type') == 'accessory', 1)
                        .otherwise(0)
                      )
                .alias('ifp_accessory_cnt_36m')
            , f.sum(f.when(f.col('ifp_type') == 'device', 1)
                        .otherwise(0)
                      )
                .alias('ifp_device_cnt_36m')
            , f.min('ifp_term_start_date').alias('min_ifp_start_date_36m')
            , f.max('ifp_term_start_date').alias('max_ifp_start_date_36m')
        )
       # .filter(f.col('fs_acct_id') == '477943749')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### ifp add in 12 months 

# COMMAND ----------

# DBTITLE 1,combine ifp on srvc + on bill at device lvl
df_ifp_base = (df_prm_ifp_srvc
        #.filter(f.col('ifp_srvc_dvc_flag') == 'Y')
        .filter(f.col('reporting_date') == vt_reporting_date)
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
        .select(*ls_reporting_date_key
                        , 'fs_acct_id'
                        , 'fs_cust_id'
                        ,'fs_ifp_id' 
                        , 'ifp_model'
                        , 'ifp_level'
                        , 'ifp_type'
                        , 'ifp_term_start_date'
                        , 'ifp_term_end_date'
                  )
        .distinct()
        .union(
                df_prm_ifp_bill
                #.filter(f.col('ifp_bill_dvc_flag') == 'Y')
                .filter(f.col('reporting_date') == vt_reporting_date)
                .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type)
                .select( *ls_reporting_date_key
                        , 'fs_acct_id'
                        , 'fs_cust_id'
                        ,'fs_ifp_id'
                        , 'ifp_model'
                        , 'ifp_level' 
                        , 'ifp_type'
                        , 'ifp_term_start_date'
                        , 'ifp_term_end_date'
                        )
                .distinct()
        ) ## aggregate into per account level 
        .filter(f.col('ifp_term_start_date') >= f.add_months(f.col('reporting_date'), -12))
        .groupBy(*ls_reporting_date_key, 'fs_acct_id', 'fs_cust_id')
        .agg(
            f.countDistinct('fs_ifp_id').alias('ifp_total_cnt_12m')
            , f.sum(f.when(f.col('ifp_type') == 'accessory', 1)
                        .otherwise(0)
                      )
                .alias('ifp_accessory_cnt_12m')
            , f.sum(f.when(f.col('ifp_type') == 'device', 1)
                        .otherwise(0)
                      )
                .alias('ifp_device_cnt_12m')
            , f.min('ifp_term_start_date').alias('min_ifp_start_date_12m')
            , f.max('ifp_term_start_date').alias('max_ifp_start_date_12m')
        )
       # .filter(f.col('fs_acct_id') == '477943749')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### combine into product base

# COMMAND ----------

# DBTITLE 1,product base
df_product_base = (
    df_master
        .filter(f.col('reporting_date') == f.lit(vt_reporting_date))
        .filter(f.col('reporting_cycle_type') == vt_reporting_cycle_type) 
        .select( *ls_prm_key
                , *ls_reporting_date_key
                , 'num_of_active_srvc_cnt'
                , 'num_of_active_acct_cnt'
                , 'srvc_start_date'
                , 'billing_acct_open_date'
                , 'cust_start_date'
                , 'first_activation_date'
                , 'plan_name'
        )
        .join(df_ifp_history_agg, ['fs_acct_id', 'fs_cust_id'], 'left' )
        # .join(df_ifp_base, ['fs_acct_id', 'fs_cust_id'] + ls_reporting_date_key, 'left')
        # .join(df_ifp_base_36m, ['fs_acct_id', 'fs_cust_id'] + ls_reporting_date_key, 'left')
)

# COMMAND ----------

# display(df_product_base
#         .agg(f.count('*')
#              , f.countDistinct('fs_srvc_id')
#              , f.countDistinct('fs_acct_id')
#              , f.sum(f.when(f.col('ifp_total_cnt_hist').isNotNull(), 1)
#                        .otherwise(0)
#                        )
#              , f.sum(f.when(f.col('ifp_total_cnt_hist').isNotNull(), 1)
#                        .otherwise(0)
#                        )
#              )
#         )

# COMMAND ----------

# display(df_product_base
#        .limit(10)
#         #.filter(f.col('fs_acct_id') == '477943749')
#         )        

# COMMAND ----------

# display(df_bb_base
#         .limit(10)
#         )

# COMMAND ----------

# MAGIC %md
# MAGIC #### bb base aggregation

# COMMAND ----------

df_bb_base_acct = (
    df_bb_base
    .withColumn('index', f.row_number().over(
                        Window
                        .partitionBy('fs_acct_id', 'bb_fs_cust_id', 'reporting_date')
                        .orderBy(f.desc('bb_SERVICE_START_DATE_TIME'))
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

# DBTITLE 1,output curr
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
        .withColumn('recent_acq_12mp_flag'
                     , f.when( f.col('bb_SERVICE_START_DATE_TIME')
                                .between( f.add_months(f.col('reporting_date'), -12), f.col('reporting_date'))
                                ,f.lit('Y')
                     )
                     .when(f.col('min_ifp_start_date_hist')
                                .between( f.add_months(f.col('reporting_date'), -12), f.col('reporting_date'))
                            , f.lit('Y')
                           )
                     .when(f.col('max_ifp_start_date_hist')
                                .between( f.add_months(f.col('reporting_date'), -12), f.col('reporting_date'))
                            , f.lit('Y')
                           )
                     # .when(f.col('min_ifp_start_date_hist').isNotNull(), f.lit('Y'))
                     .otherwise('N')
       )
      .withColumn('bb_add_in_12mp'
                   ,  f.when( f.col('bb_SERVICE_START_DATE_TIME')
                               .between( f.add_months(f.col('reporting_date'), -12), f.col('reporting_date'))
                               , 1
                      )
                      .otherwise(0)
     )
     .withColumn('ifp_add_in_12mp'
                  ,  f.when( f.col('min_ifp_start_date_hist')
                               .between( f.add_months(f.col('reporting_date'), -12), f.col('reporting_date'))
                             ,1
                     )
                  .when( f.col('max_ifp_start_date_hist')
                               .between( f.add_months(f.col('reporting_date'), -12), f.col('reporting_date'))
                             ,1
                     )
                  # , f.when(f.col('min_ifp_start_date_hist').isNotNull()
                  #          , 1
                  #          )
                  .otherwise(0)
     )
     .withColumn('cnt_product_add_in_12mp'
                , f.col('ifp_add_in_12mp') + f.col('bb_add_in_12mp')
                )
     .select('fs_acct_id'
             , 'fs_cust_id'
             , 'fs_srvc_id'
             , 'reporting_date'
             , 'reporting_cycle_type'
             , 'num_of_active_srvc_cnt'
             , 'bb_add_in_12mp'
             , 'ifp_add_in_12mp'
             , 'cnt_product_add_in_12mp'
             , 'recent_acq_12mp_flag'
             , 'product_holding'
             , 'bb_SERVICE_START_DATE_TIME'
             , 'ifp_total_cnt_hist'
             , 'ifp_accessory_cnt_hist'
             , 'ifp_device_cnt_hist'
             , 'min_ifp_start_date_hist'
             , 'max_ifp_start_date_hist'
             )
        )

# COMMAND ----------

display(df_output_curr
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             , f.countDistinct('fs_srvc_id')
        )
)

# COMMAND ----------

display(df_output_curr
        .groupBy('recent_acq_12mp_flag', 'cnt_product_add_in_12mp', 'ifp_add_in_12mp' , 'reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

display(df_output_curr
        #.groupBy('recent_acq_12mp_flag', 'cnt_product_add_in_12mp', 'ifp_add_in_12mp')
        .agg(f.min('min_ifp_start_date_hist')
             , f.max('min_ifp_start_date_hist')
            , f.min('max_ifp_start_date_hist')
            , f.max('max_ifp_start_date_hist')
              )
        )


# COMMAND ----------

display(df_output_curr.count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### export data

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

#dbutils.fs.rm('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_product_acq_hist', recurse=True)

# COMMAND ----------

export_data(
            df = df_output_curr
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_product_acq_hist'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/cohort_seg/fea_product_acq_hist')

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

# display(
#     df_test
#     .groupBy(
#              'reporting_date'
#              ,'product_holding'
#              #, 'ifp_prm_dvc_flag'
#              , 'recent_acq_12mp_flag'
#              , 'cnt_product_add_in_12mp'
#              , 'ifp_add_in_12mp'
#              )
#     .agg(f.countDistinct('fs_acct_id'))
# )

# COMMAND ----------

# MAGIC %md
# MAGIC #### export ifp transaction in snowflake

# COMMAND ----------

# # ------------ login to snowflake
# password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

# options = {
#   "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
#   "sfUser": "SVC_LAB_DS_DATABRICKS",
#   "pem_private_key": password.replace('\\n', '\n'),
#   "sfDatabase": "LAB_ML_STORE",
#   "sfSchema": "SANDBOX",
#   "sfWarehouse": "LAB_DS_WH_SCALE"
# }

# COMMAND ----------

# display(df_ifp_history)

# COMMAND ----------

# (
#     df_ifp_history
#     .write
#     .format("snowflake")
#     .options(**options)
#     .option("dbtable", "lab_ml_store.sandbox.sc_one_off_ifp_hist_march24")
#     .mode("overwrite")
#     .save()
# )
