# Databricks notebook source
# MAGIC %md
# MAGIC **Objective**
# MAGIC
# MAGIC - to understand churn rate by different cohort (within ifp terms vs. beyond ifp terms vs no terms )
# MAGIC - ifp term separated by 12 months, 24 months, 36 months, 36 months only started sicne march 2019 
# MAGIC

# COMMAND ----------

import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab"

# COMMAND ----------

dir_fs_data_meta = os.path.join(dir_fs_data_parent, 'd000_meta')
dir_fs_data_raw =  os.path.join(dir_fs_data_parent, 'd100_raw')
dir_fs_data_int =  os.path.join(dir_fs_data_parent, "d200_intermediate")
dir_fs_data_prm =  os.path.join(dir_fs_data_parent, "d300_primary")
dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")
dir_fs_data_target = os.path.join(dir_fs_data_parent, "d500_movement")
dir_fs_data_serv = os.path.join(dir_fs_data_parent, "d600_serving")

# COMMAND ----------

vt_param_reporting_date = "2024-06-30"
vt_param_reporting_cycle_type = "calendar cycle"

# COMMAND ----------

df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))
df_fs_ifp_device = spark.read.format('delta').load(os.path.join(dir_fs_data_fea, 'd401_mobile_oa_consumer/fea_ifp_device_account'))
df_fs_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_data_fea, 'd401_mobile_oa_consumer/fea_unit_base'))
df_fs_deact = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation')
# /mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation

# meta for fields 
df_fs_meta = spark.read.format('delta').load(os.path.join(dir_fs_data_meta,'d004_fsr_meta','fsr_field_meta'))

# COMMAND ----------

display(df_fs_deact
       .limit(100)
)


# COMMAND ----------

display(df_fs_deact
        .select('fs_cust_id'
                , 'fs_acct_id'
                ,  'fs_srvc_id'
                , 'movement_date'
                )
        .groupBy(f.date_format('movement_date', 'yyyy-MM'))
        .agg(f.countDistinct('fs_cust_id', 'fs_acct_id', 'fs_srvc_id'))
        )

# COMMAND ----------

vt_param_plan_only_join_key = ['fs_cust_id', 'fs_acct_id', 'fs_srvc_id']

# COMMAND ----------

# cohort base 

# never ever have ifp , have ifp on bill, have ifp on service 

# ifp base 
df_prm_ifp_base = (
    df_fs_master
        .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
        .filter(f.col('srvc_start_date') >= '2019-01-01')
        .select(  
                 'fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id' 
                , 'fs_ifp_prm_dvc_id'
                , 'srvc_start_date'
                , 'ifp_prm_dvc_term_start_date'
                , 'ifp_prm_dvc_term_end_date'
                , 'ifp_prm_dvc_term'
               # , 'fs_ifp_prm_dvc_order_id'
                )
        .distinct()
        .filter(f.col('ifp_prm_dvc_flag') == f.lit('Y'))
)


# plan only base 
df_plan_only = (
    df_fs_master
    .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type) 
    .filter(f.col('srvc_start_date') >= '2019-01-01')
    .select('fs_cust_id'
            , 'fs_acct_id'
            , 'fs_srvc_id'
            , 'fs_ifp_prm_dvc_id' 
            #, 'srvc_start_date'
            )
    .distinct()
    .join(df_prm_ifp_base, ['fs_cust_id', 'fs_acct_id', 'fs_srvc_id']
        , 'anti'
          )
)


# COMMAND ----------

# df_plan_churned= spark.read.format('csv').option("header", "true").load("dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/df_plan_churned.csv")

#df_plan_churned.write.format("csv").option("header", "true").save("dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/df_plan_churned.csv")

# COMMAND ----------

# DBTITLE 1,plan only churn date
df_plan_churned = (df_fs_master
        .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
        .filter(f.col('srvc_start_date') >= '2019-01-01')
        .select('reporting_date'
                , 'fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id'
                , 'srvc_start_date'
                )
        .join(df_plan_only,vt_param_plan_only_join_key, 'inner' )
        .withColumn('index', f.row_number().over(
                                    Window.partitionBy(vt_param_plan_only_join_key)
                                    .orderBy(f.desc('reporting_date'))
                                                 )
                    )
        .filter(f.col('index') == f.lit(1))
        .withColumn('plan_tenure', f.round(
            f.months_between('reporting_date'
                             , 'srvc_start_date'),0)
                    )
        .withColumnRenamed('reporting_date', 'last_active_rpt_date')
        .drop('index')
        .withColumn('status', f.when(
                                    f.col('last_active_rpt_date') == vt_param_reporting_date,
                                    f.lit('Active')
                                )
                                .when(
                                    f.col('last_active_rpt_date') < vt_param_reporting_date,
                                    f.lit('Churned Plan Only')
                                )
                                .otherwise(f.lit('Unknown'))
                    )
       # .groupBy('status', 'plan_tenure')
       # .agg(f.count('*'))
        )

display(df_plan_churned
        .groupBy('status', 'plan_tenure')
        .agg(f.count('*'))
        )

# COMMAND ----------

# df_plan_churned.write.format("csv").option("header", "true").save("dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/df_plan_churned.csv")

# COMMAND ----------

# DBTITLE 1,ifp churn date
df_ifp_churned = (
    df_fs_master
    .filter(f.col('reporting_cycle_type') == vt_param_reporting_cycle_type)
    .filter(f.col('srvc_start_date') >= '2019-01-01')
    .select('reporting_date'
               , 'fs_cust_id'
               , 'fs_acct_id'
               , 'fs_srvc_id'
                )
    .join(df_prm_ifp_base, vt_param_plan_only_join_key ,'inner')
    .withColumn('index', f.row_number().over(Window.partitionBy(vt_param_plan_only_join_key)
                                             .orderBy(f.desc('reporting_date'))
                                             )
                )
    .filter(f.col('index') == f.lit(1))
    .withColumnRenamed('reporting_date', 'last_active_rpt_date')
    .drop('index')
    .withColumn('status', f.when(
                                    f.col('last_active_rpt_date') == vt_param_reporting_date,
                                    f.lit('Active')
                                )
                            .when(
                                    (f.col('last_active_rpt_date') < vt_param_reporting_date) & 
                                    (f.col('last_active_rpt_date') <= f.col('ifp_prm_dvc_term_end_date'))
                                    ,f.lit('Churned Within Term')
                                )
                            .when(
                                    (f.col('last_active_rpt_date') < vt_param_reporting_date) & 
                                    (f.col('last_active_rpt_date') > f.col('ifp_prm_dvc_term_end_date'))
                                    ,f.lit('Churned Beyond Term')
                                )
                            .otherwise(f.lit('Unknown'))
                 
                )
    .withColumn('tenure',  
                f.round(
                    f.months_between(
                        'last_active_rpt_date', 'ifp_prm_dvc_term_start_date'
                    ) ,0
                )
                 )
    #.groupBy('status', 'ifp_prm_dvc_term', 'tenure')
    #.agg(f.count('*'))
)

display(df_ifp_churned
        .groupBy('status', 'ifp_prm_dvc_term', 'tenure')
        .agg(f.count('*'))
        )

# COMMAND ----------

display(df_ifp_churned)

# COMMAND ----------

display(df_plan_churned)

# COMMAND ----------

display(df_prm_ifp_base.limit(100))

# COMMAND ----------

display( df_prm_ifp_base
        .withColumn('cnt', f.count('*').over(Window.partitionBy('fs_acct_id'
                                                                    , 'fs_cust_id'
                                                                    , 'fs_srvc_id'
                                                                    )
        ))
        .filter(f.col('cnt') == 2)
       # .groupBy('cnt')
       # .agg(f.count('*'))
)
