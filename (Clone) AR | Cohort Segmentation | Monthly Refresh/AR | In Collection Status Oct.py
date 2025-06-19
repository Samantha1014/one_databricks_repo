# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 set up 

# COMMAND ----------

# DBTITLE 1,library
import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta

# COMMAND ----------

# DBTITLE 1,utilities
# MAGIC %run "../utility_functions"

# COMMAND ----------

# DBTITLE 1,directory
dir_fs_data_parent = '/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer'
dir_fs_data_stg = '/mnt/feature-store-prod-lab/d200_staging/d299_src'
dir_fs_data_prm = '/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/'
dir_aod30d_mvnt = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_v3'  #history from 2023-01

# COMMAND ----------

# DBTITLE 1,parameters
ls_param_unit_base_fields = ["fs_cust_id", "fs_acct_id"]
ls_param_aod30_joining_keys = ["fs_acct_id"]
ls_reporting_date_key = ['reporting_date', 'reporting_cycle_type']
ls_param_snapshot_base_join_key = ["fs_cust_id", "fs_acct_id", "fs_srvc_id", "reporting_date", "reporting_cycle_type"]

# COMMAND ----------

# DBTITLE 1,load data
df_fea_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_data_parent, 'fea_unit_base'))
df_base_aod30_event = spark.read.format('delta').load(dir_aod30d_mvnt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 dev

# COMMAND ----------

# DBTITLE 1,unit base
df_base_unit_base = (df_fea_unit_base
        .select( *ls_param_snapshot_base_join_key)
        .filter(f.col('reporting_date') >= '2023-01-01') 
        )

# COMMAND ----------

display(df_base_aod30_event.limit(10))

# COMMAND ----------

df_aod_status_01 = (df_fea_unit_base
        .filter(f.col('reporting_date') >= '2023-01-01')
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        #.filter(f.col('fs_acct_id') == '474984036')
        .select(*ls_param_snapshot_base_join_key)
        .join(df_base_aod30_event
              .drop(*ls_reporting_date_key)
              , ls_param_unit_base_fields, 'left') 
        .withColumn('aod30_status', 
                    f.when(  
                           (f.col('bill_close_flag') == f.lit('N'))
                           & ( f.col('reporting_date') >= f.col('aod_30_date'))
                           , 'Y'
                    )
                    .when( 
                          (f.col('bill_close_flag') == f.lit('Y')) 
                          & ( 
                               (f.col('reporting_date') <= f.last_day('bill_close_date')) 
                                & (f.col('reporting_date') >= f.last_day('aod_30_date')) 
                            )
                          , 'Y' )
                    .when(
                            ( f.col('bill_close_flag') == f.lit('Y'))
                            & (f.col('reporting_date') > f.last_day('bill_close_date'))
                            , 'N' 
                        )   
                )
        .filter(f.col('aod30_status') == 'Y')
        .select('fs_acct_id', 'fs_cust_id', 'fs_srvc_id', 'reporting_date'
                , 'reporting_cycle_type', 'aod30_status', 'bill_no', 
                'bill_due_date',
                'bill_close_date',
                'aod_30_date',
                'bill_close_flag',
                'bill_close_days'
                )
               )

# COMMAND ----------

 display(df_fea_unit_base
        .filter(f.col('reporting_date') >= '2023-01-01')
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        #.filter(f.col('fs_acct_id') == '474984036')
        .select(*ls_param_snapshot_base_join_key)
        .join(df_base_aod30_event
              .drop(*ls_reporting_date_key)
              , ls_param_unit_base_fields, 'left') 
        .withColumn('aod30_status', 
                    f.when(  
                           (f.col('bill_close_flag') == f.lit('N'))
                           & ( f.col('reporting_date') >= f.col('aod_30_date'))
                           , 'Y'
                    )
                    .when( 
                          (f.col('bill_close_flag') == f.lit('Y')) 
                          & ( 
                               (f.col('reporting_date') <= f.last_day('bill_close_date')) 
                                & (f.col('reporting_date') >= f.last_day('aod_30_date')) 
                            )
                          , 'Y' )
                    .when(
                            ( f.col('bill_close_flag') == f.lit('Y'))
                            & (f.col('reporting_date') > f.last_day('bill_close_date'))
                            , 'N' 
                        )   
                )
       #  .filter(f.col('fs_acct_id') == '1139059')
        .select('fs_acct_id', 'fs_cust_id', 'fs_srvc_id', 'reporting_date'
                , 'reporting_cycle_type', 'aod30_status', 'bill_no', 
                'bill_due_date',
                'bill_close_date',
                'aod_30_date',
                'bill_close_flag',
                'bill_close_days'
                )
               )

# COMMAND ----------

display(df_fea_unit_base
        .filter(f.col('reporting_date') >= '2023-01-01')
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .filter(f.col('fs_acct_id') =='1139059')
        )

# COMMAND ----------

df_aod_status_02 = (df_aod_status_01
        .groupBy(*ls_param_snapshot_base_join_key, 'aod30_status') 
        .agg(   
             f.countDistinct('bill_no').alias('aod30d_bill_cnt')
             , f.min('aod_30_date').alias('min_aod_30_date')
             , f.max('aod_30_date').alias('max_aod_30_date')
             , f.min('bill_close_date').alias('min_close_date')
             , f.max('bill_close_date').alias('max_bill_close_date')
             , f.avg('bill_close_days').alias('avg_bill_close_days')
             )
                  )

# COMMAND ----------

display(df_aod_status_02
        .filter(f.col('fs_acct_id') == '474984036')
        )

# COMMAND ----------

# DBTITLE 1,check data
display(df_aod_status_02
        .groupBy('reporting_date', 'reporting_cycle_type')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

df_aod_status_02_oct = (
    df_aod_status_02
    .filter(f.col('reporting_date') == '2024-10-31')
                        )

# COMMAND ----------

export_data(
            df = df_aod_status_02_oct
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/99_misc/df_aod_stauts_v3'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["reporting_date"]
        )

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/99_misc/df_aod_stauts_v3')

# COMMAND ----------

display(df_test.limit(10))

# COMMAND ----------

display(df_test
        .groupBy('reporting_date', 'reporting_cycle_type')
        .agg(f.count('*')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

display(df_fea_unit_base
        .filter(f.col('reporting_date') >= '2023-01-01')
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .groupBy(f.col('reporting_date'), f.col('reporting_cycle_type'))
        .agg(f.count('*'))
)

# COMMAND ----------

display(df_fea_unit_base
        .filter(f.col('reporting_date') >= '2023-01-01')
        .filter(f.col('reporting_cycle_type') == 'calendar cycle')
        .join(df_test,  ls_param_snapshot_base_join_key , 'left')
        .filter(f.col('aod30_status').isNull())
        .groupBy(f.col('reporting_date'), f.col('reporting_cycle_type'))
        .agg(f.count('*')
             , f.countDistinct('fs_srvc_id')
             , f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_cust_id')
             )
        )
