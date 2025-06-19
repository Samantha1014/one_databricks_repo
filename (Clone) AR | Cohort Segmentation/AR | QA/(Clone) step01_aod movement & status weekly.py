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

# COMMAND ----------

# DBTITLE 1,diretory
dir_fs_stg_bill = "/mnt/feature-store-prod-lab/d200_staging/d299_src"
dir_fs_prm_base  = "/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer" 
dir_fs_fea_base = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer" 

# COMMAND ----------

# DBTITLE 1,load data
df_stg_bill = spark.read.format('delta').load(os.path.join(dir_fs_stg_bill, 'stg_brm_bill_t'))
# df_prm_bill = spark.read.format('delta').load(os.path.join(dir_fs_prm_base, 'prm_bill_cycle_billing_6'))
df_prm_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_prm_base, 'prm_unit_base'))
df_fea_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_fea_base, 'fea_unit_base'))

# COMMAND ----------

# DBTITLE 1,parameters
ls_param_unit_base_fields = ["fs_cust_id", "fs_acct_id"]
ls_param_aod30d_joining_keys = ["fs_acct_id"]
ls_param_snapshot_base_join_key = ["fs_cust_id", "fs_acct_id", "fs_srvc_id", "reporting_date", "reporting_cycle_type"]

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
vt_calendar_cycle_type = 'rolling cycle'

# COMMAND ----------

# MAGIC %md
# MAGIC #### s02 dev

# COMMAND ----------

# DBTITLE 1,OA Account ever exist
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

# DBTITLE 1,aod 30d movement part 1
## get bill close days and aod 30 date at per bill level 
df_base_aod30_00_curr = (
    df_stg_bill
    .filter(f.col('bill_no').isNotNull())
    .filter(f.col('bill_close_t').isNotNull())
    .filter(f.col('total_due') > 0)  
    # actual overdue, not credit
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
    .filter(f.col('aod_30_date') >= vt_stg_bill_start_period) 
     # only look for 2023-01 onwards
    .withColumn('bill_close_flag'
        , f.when(f.col('bill_close_date') <= '1970-01-31', 'N')
         .otherwise('Y')
    )
    .withColumn('bill_close_days'
        , f.when(f.col('bill_close_flag') == 'Y'
            , f.datediff('bill_close_date', 'bill_due_date')
        )
        .when(f.col('bill_close_flag') == 'N'
            , f.datediff(f.current_date(), 'bill_due_date')
        )
    )
    .join(
        df_base_unit_base
        , ls_param_aod30d_joining_keys
        , 'inner'
    )  # filter on OA mobile customers only
    .select(
        'fs_acct_id'
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

# filter on those one that has aod > 30 days and weekly reporting date 

df_base_aod30_01_curr = (
    df_base_aod30_00_curr
    .filter(f.col('bill_close_days') > 30)
    #.withColumn('reporting_date', f.last_day('aod_30_date'))
    .withColumn('reporting_date', f.next_day(f.col('aod_30_date'), 'Sunday'))
    .withColumn('reporting_cycle_type', f.lit(vt_calendar_cycle_type))
)


# COMMAND ----------

# one account could have more than 1 bill that went into aod 30 for the same reporting date period  
df_base_aod30_agg = (
     df_base_aod30_01_curr
     .groupBy('fs_acct_id', 'fs_cust_id', 'reporting_date', 'reporting_cycle_type')
     .agg(f.min('aod_30_date').alias('min_movement_date')
          , f.max('aod_30_date').alias('max_movement_date')
          , f.min('bill_close_date').alias('min_close_date')
          , f.max('bill_close_date').alias('max_close_date')
          , f.count('bill_no').alias('aod_bill_cnt')
     )
)

# COMMAND ----------

display(
    df_base_aod30_agg
    .groupBy('min_movement_date')
    .agg(
        f.countDistinct('fs_acct_id')
       # , f.countDistinct('fs_srvc_id') 
    )     
)

# COMMAND ----------

display(
    df_base_aod30_agg
    .filter(f.col('fs_acct_id') == '464705106')        
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### qa danny 

# COMMAND ----------

dir_danny_path = "dbfs:/mnt/feature-store-dev/dev_users/dev_dw/24q4_fs_fundation/"
dir_data_parent_shared = os.path.join(dir_danny_path, "d400_feature/d401_mobile_oa_consumer")
dir_data_parent_mvmt = os.path.join(dir_danny_path, "d500_movement/d501_mobile_oa_consumer")
dir_data_parent_stag = os.path.join(dir_danny_path, "d200_staging/d299_src")

# COMMAND ----------

df_fea_coll_action = spark.read.format('delta').load(os.path.join(dir_data_parent_shared ,'fea_coll_action_cycle_12mnth'))
df_fea_product_acq = spark.read.format('delta').load(os.path.join(dir_data_parent_shared, 'fea_product_acquisition_cycle_billing_12'))
df_fea_late_pay = spark.read.format('delta').load(os.path.join(dir_data_parent_shared,"fea_late_pay_cycle_billing_6"))
df_mvmt_aod = spark.read.format('delta').load(os.path.join(dir_data_parent_mvmt, 'mvmt_aod30d'))

# COMMAND ----------

display(
    df_mvmt_aod
    .groupBy('movement_date')
    .agg(f.countDistinct('fs_acct_id'))     
)

# COMMAND ----------

display(
    df_base_aod30_agg
    .filter(f.col('min_movement_date') == '2024-10-23')
    .join(
        df_mvmt_aod
        .filter(f.col('movement_date') == '2024-10-23')
        , ['fs_acct_id']
        , 'anti'
    )
   # .filter(f.col('fs_acct_id') == '440162484')    
)

# COMMAND ----------

display(
    df_mvmt_aod
    .filter(f.col('movement_date') == '2024-10-23')
    .join(
        df_base_aod30_agg
        .filter(f.col('min_movement_date') == '2024-10-23')
        , ['fs_acct_id']
        , 'anti'
    )
   # .filter(f.col('fs_acct_id') == '440162484')    
)

# COMMAND ----------

display(
    df_mvmt_aod
    .filter(f.col('fs_acct_id') == '484731086')
)

# COMMAND ----------

display(
    df_base_aod30_agg
    .filter(f.col('fs_acct_id') == '474507481')
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### s03 current week's aod movement

# COMMAND ----------

# DBTITLE 1,get current week
vt_reporting_date = (
    df_base_aod30_agg
    .select(
        f.max(
            f.when(
                f.col('reporting_date') <= f.current_timestamp(),
                f.col('reporting_date')
            )
        ).alias('latest_reporting_date') 
    )
    .collect()[0]['latest_reporting_date']
)

# COMMAND ----------

vt_reporting_date

# COMMAND ----------

# DBTITLE 1,get output for current week
df_output_curr = (
    df_base_aod30_agg
    .filter(f.col('reporting_date') == vt_reporting_date)
    .withColumn('movement_date', f.col('min_movement_date'))
    .withColumn('movement_type', f.lit('AOD30'))
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
)

# COMMAND ----------

display(df_output_curr.count())

# COMMAND ----------

display(df_output_curr)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,export current week
(df_output_curr
    .write
    .format("delta")
    .mode("overwrite")
    .save('/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg_curr')
)

# COMMAND ----------

(df_output_curr
    .write
    .format("delta")
    .mode("append")
    .save('/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg_weekly')
)

# COMMAND ----------

df_test2 = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg_curr')

# COMMAND ----------

display(df_test2
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             #, f.countDistinct('fs_srvc_id')
             , f.count('*')
             )
        )

# COMMAND ----------

df_test = spark.read.format('delta').load('/mnt/ml-lab/dev_users/dev_sc/aod30_mvnt_acct_agg_weekly')

# COMMAND ----------

display(df_test
        .groupBy('reporting_date', 'reporting_cycle_type')
        .agg(f.count('*'))
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### s04 current week's aod status 

# COMMAND ----------

df_aod_status_01 = (df_fea_unit_base
        .filter(f.col('reporting_date') >= '2023-01-01')
        .filter(f.col('reporting_cycle_type') == 'rolling cycle')
        #.filter(f.col('fs_acct_id') == '474984036')
        .select(*ls_param_snapshot_base_join_key)
        .join(df_base_aod30_01_curr
              .drop('reporting_date', 'reporting_cycle_type')
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
                               (f.col('reporting_date') <= f.next_day(
                                   f.col('bill_close_date') ,'Sunday')
                                ) 
                                & (f.col('reporting_date') >= f.next_day(
                                    f.col('aod_30_date'), 'Sunday')
                                   ) 
                            )
                          , 'Y' )
                    .when(
                            ( f.col('bill_close_flag') == f.lit('Y'))
                            & (f.col('reporting_date') > f.next_day(
                                f.col('bill_close_date'), 'Sunday')
                               )
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
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             )
        )

# COMMAND ----------

df_aod_status_curr = (
    df_aod_status_02
    .filter(f.col('reporting_date') == vt_reporting_date)
    .withColumn("data_update_date", f.current_date())
    .withColumn("data_update_dttm", f.current_timestamp())
)

# COMMAND ----------

display(df_aod_status_curr
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
       )

# COMMAND ----------

# MAGIC %md
# MAGIC #### export aod30 status for next step

# COMMAND ----------

#dbutils.fs.rm('/mnt/ml-lab/dev_users/dev_sc/df_aod_stauts_curr')

# COMMAND ----------

# DBTITLE 1,current week's aod status
(df_aod_status_curr
    .write
    .format("delta")
    .mode("overwrite")
    .save('/mnt/ml-lab/dev_users/dev_sc/df_aod_stauts_curr')
)

# COMMAND ----------

df_test3 = spark.read.format("delta").load('/mnt/ml-lab/dev_users/dev_sc/df_aod_stauts_curr')

# COMMAND ----------

display(df_test3
        .groupBy('reporting_date')
        .agg(f.countDistinct('fs_acct_id'))
        )
