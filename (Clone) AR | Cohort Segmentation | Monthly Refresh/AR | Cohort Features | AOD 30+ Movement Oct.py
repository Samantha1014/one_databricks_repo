# Databricks notebook source
# MAGIC %md
# MAGIC #### s01 set up

# COMMAND ----------

# DBTITLE 1,library
import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
from delta.tables import *

# COMMAND ----------

# DBTITLE 1,utility function
# MAGIC %run "./utility_functions"

# COMMAND ----------

# DBTITLE 1,directory
dir_fs_stg_bill = "/mnt/feature-store-prod-lab/d200_staging/d299_src"
dir_fs_prm_base  = "/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer" 
dir_fs_fea_base = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer" 

# COMMAND ----------

# DBTITLE 1,load data
df_stg_bill = spark.read.format('delta').load(os.path.join(dir_fs_stg_bill, 'stg_brm_bill_t'))
df_prm_bill = spark.read.format('delta').load(os.path.join(dir_fs_prm_base, 'prm_bill_cycle_billing_6'))
df_prm_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_prm_base, 'prm_unit_base'))
df_fea_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_fea_base, 'fea_unit_base'))

# COMMAND ----------

# DBTITLE 1,parameters
ls_param_unit_base_fields = ["fs_cust_id", "fs_acct_id"]
ls_param_aod30d_joining_keys = ["fs_acct_id"]
ls_param_export_fields = [
'reporting_date',
'reporting_cycle_type',
'fs_cust_id',
'fs_acct_id',
'bill_no',
'bill_end_date', 
'bill_due_date',
'bill_close_date',
'aod_30_date',
'data_update_date',
'data_update_dttm',
'bill_close_flag',
'bill_close_days',
'previous_total', 
'movement_type',
'movement_date'
]

## 
vt_stg_bill_start_period = '2023-01-01'
vt_calendar_cycle_type = 'calendar cycle'

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 dev

# COMMAND ----------

# DBTITLE 1,OA mobile account ever exist
df_base_unit_base = (
            df_prm_unit_base
            .select(*ls_param_unit_base_fields, "cust_start_date")
            .distinct()
            .withColumn('index'
                        , f.row_number().over(
                          Window
                         .partitionBy('fs_acct_id')
                         .orderBy(f.desc('cust_start_date'))
                                              )
                        )
            .filter(f.col('index') == 1)
        )

# COMMAND ----------

# DBTITLE 1,movement part 1
df_base_aod30_00_curr = ( 
    df_stg_bill
    .filter(f.col('bill_no').isNotNull())
    .filter(f.col('bill_close_t').isNotNull())
    .filter(f.col('total_due')>0)
    .withColumn('index'
                , f.row_number().over(
                  Window
                 .partitionBy('bill_no')
                 .orderBy(f.desc('bill_mod_date'))
                                    )
     )
    .filter(f.col('index') == 1)
    .drop('index')
    .withColumn('aod_30_date'
                , f.date_add('bill_due_date', 30)
               )
    .filter(f.col('aod_30_date') >= vt_stg_bill_start_period )   # only look for 2023-01 onwards 
    .withColumn('bill_close_flag' 
                , f.when(
                  f.col('bill_close_date') <= '1970-01-31', 'N'
                        )
                   .otherwise('Y')
                )
    .withColumn('bill_close_days'
                , f.when(
                        f.col('bill_close_flag') == 'Y'
                        ,f.datediff('bill_close_date', 'bill_due_date')
                        )
                   .when(
                        f.col('bill_close_flag') == 'N'
                        ,f.datediff(f.current_date(), 'bill_due_date')
                        )
               )
    .join(
          df_base_unit_base
         ,ls_param_aod30d_joining_keys
         ,'inner'
         ) # filteron OA mobile customers only 
    .select('fs_acct_id'
            , 'fs_cust_id'
            , 'bill_no'
            , 'bill_end_date'
            , 'bill_due_date'
            , 'bill_close_date'
            , 'aod_30_date'
            , 'bill_close_flag'
            , 'bill_close_days'
            , 'previous_total'
            , 'total_due'
            )
)

# COMMAND ----------

# DBTITLE 1,movement part 2
df_base_aod30_01_curr = (df_base_aod30_00_curr
        .filter(f.col('bill_close_days')>30)
        .withColumn('reporting_date', f.last_day('aod_30_date'))
        .withColumn('reporting_cycle_type', f.lit(vt_calendar_cycle_type))
        #.withColumn('aod30_flag', f.lit('Y'))
        # .filter(f.col('fs_acct_id') == '1038833')
        )

# COMMAND ----------

# DBTITLE 1,agg into account level
# one account could have more than 1 bill that went into aod 30 for the same reporting month 
df_base_aod30_agg=(df_base_aod30_01_curr
        .groupBy('fs_acct_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type')
        .agg(f.min('aod_30_date').alias('min_movement_date')
             , f.max('aod_30_date').alias('max_movement_date')
             , f.min('bill_close_date').alias('min_close_date')
             , f.max('bill_close_date').alias('max_close_date')
             , f.count('bill_no').alias('aod_bill_cnt')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ####s03 export

# COMMAND ----------

# DBTITLE 1,agg
df_output_curr = (
    df_base_aod30_agg
    .withColumn('movement_date', f.col('min_movement_date'))
    .withColumn('movement_type', f.lit('AOD30'))
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
    .filter(f.col('reporting_date') == '2024-10-31')
)

# COMMAND ----------

df_output_curr_01 = (
    df_base_aod30_01_curr
    .withColumn('movement_date', f.col('aod_30_date'))
    .withColumn('movement_type', f.lit('AOD30'))
    .withColumn('data_update_date', f.current_date())
    .withColumn('data_update_dttm', f.current_timestamp())
    .select(*ls_param_export_fields)
    .filter(f.col('reporting_date') == '2024-10-31')
)

# COMMAND ----------

display(df_output_curr_01
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

# DBTITLE 1,check agg count
display(df_output_curr
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

# DBTITLE 1,check count
display(df_output_curr.filter(f.col('fs_acct_id') == '1038833'))

display(df_output_curr.count())

# COMMAND ----------

# DBTITLE 1,delete  previous reporting date

# delta_run_node_meta = DeltaTable.forPath(spark,  '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_v3')
# delta_run_node_meta.delete(
#      (f.col("reporting_date") == "2024-10-31")
# )

# COMMAND ----------

#dbutils.fs.rm( '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_v3', True)

# COMMAND ----------

# DBTITLE 1,export aod30 movement
export_data(
            df = df_output_curr_01
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_v3'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["movement_date"]
        )

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_v3')

display(df_test
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

#dbutils.fs.rm( '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg', True)

# COMMAND ----------

# delta_run_node_meta = DeltaTable.forPath(spark,  '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg')
# delta_run_node_meta.delete(
#      (f.col("reporting_date") == "2024-10-31")
# )

# COMMAND ----------

# DBTITLE 1,export aod30 agg
export_data(
            df = df_output_curr
            , export_path = '/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg'
            , export_format = "delta"
            , export_mode = "append"
            , flag_overwrite_schema = True
            , flag_dynamic_partition = True
            , ls_dynamic_partition = ["movement_date"]
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### s04 test

# COMMAND ----------

# DBTITLE 1,load
df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_v3')
df_test_2 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg')

# COMMAND ----------

display(df_output_curr.limit(10))

# COMMAND ----------

display(df_test_2.limit(10))

# COMMAND ----------

display(df_test
        .groupBy(f.date_format('movement_date', 'yyyy-MM'))
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
        )

# COMMAND ----------

display(df_test_2
        .groupBy(f.date_format('movement_date', 'yyyy-MM'))
        .agg(f.countDistinct('fs_acct_id'))
        )

# COMMAND ----------

display(df_test_2
        .filter(f.col('aod_bill_cnt') >1)
        .count()
        )

# COMMAND ----------

# DBTITLE 1,check count
display(df_test_2
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.count('*')
             )
)

display(df_test_2.count())
