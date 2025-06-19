# Databricks notebook source
import pyspark 
import os
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window
from typing import Optional, List, Dict, Tuple

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab"
dir_mls_data_parent = "/mnt/ml-store-prod-lab/classification"

dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")

# COMMAND ----------

# mvnt base 

df_srvc_mvnt = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation/reporting_cycle_type=rolling cycle')

# plan 
df_plan = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_plan')

# fea master 

df_master = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/reporting_cycle_type=rolling cycle') 


# unit base
df_fea_unitbase = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_unit_base/reporting_cycle_type=rolling cycle')

# ifp base 
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))

df_fs_ifp_acct = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_ifp_device_account')


# COMMAND ----------

df_srvc_mvnt = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation')

display(
    df_srvc_mvnt
    .groupBy('reporting_date')
    .agg(f.count('*'))
)

# COMMAND ----------

display(
    df_srvc_mvnt
    .filter(f.col('deactivate_reason').isin('Collections-Last                                  ', 'Collections                                       '
                                            , 'COLL - Uneconomical To Pursue                     '
                                            , 'COLL - Lost Job                                   '
                                            , 'COLL - Collections                                '
                                            
                                            ))
    .groupBy( f.next_day(f.date_add(f.col("movement_date"), -7), "Saturday"), 'deactivate_type')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.countDistinct('fs_srvc_id')
    )
)

# COMMAND ----------

display(df_fea_unitbase.limit(10))

# COMMAND ----------

df_stage_one = (
    df_srvc_mvnt
    .filter(f.col('movement_date').between('2024-07-01', '2025-02-28'))
    .filter(f.col('deactivate_type') == 'Involuntary')
    .join(df_master, ['fs_acct_id', 'fs_srvc_id', 'fs_cust_id'], 'left')
    .withColumn(
        'rnk'
        , f.row_number().over(
            Window.partitionBy('fs_cust_id', 'fs_acct_id', 'fs_srvc_id')
            .orderBy(f.desc(df_master.reporting_date))
            )
    )
    .filter(f.col('rnk')==1)
    .select( 'fs_cust_id'
            ,'fs_srvc_id', 'fs_acct_id', 'srvc_start_date', 'movement_date','deactivate_type'
            , 'deactivate_reason', 'plan_proposition_name', 'plan_name', 'plan_amt', 'bill_due_amt'
            , df_master.reporting_date, 'service_sales_channel_branch', 'service_sales_channel_group'
            , 'service_sales_channel'
            )
)

# COMMAND ----------

# DBTITLE 1,check ifp or plan only
df_ifp_acct_lvl = (df_fs_ifp_acct
        .filter(f.col('ifp_acct_dvc_term_end_date_min').isNotNull())
        .select('fs_acct_id', 'fs_cust_id', 'ifp_acct_dvc_term_end_date_max')
        .filter(f.col('ifp_acct_dvc_term_end_date_max') >= '2024-07-01')
        .distinct()
        )

# COMMAND ----------

# DBTITLE 1,pre2post
df_pre2post= (
    df_master
    .filter(f.col('pre2post_flag') == 'Y')
    .select('fs_acct_id', 'fs_cust_id', 'fs_srvc_id', 'pre2post_flag', 'pre2post_date')
    .distinct()
)

# COMMAND ----------

df_output = (
    df_stage_one
    .join(df_ifp_acct_lvl, ['fs_acct_id', 'fs_cust_id'], 'left')
    .join(df_pre2post, ['fs_acct_id', 'fs_cust_id', 'fs_srvc_id'], 'left')
    .withColumn(
        'ifp_flag'
        , f.when(
            f.col('ifp_acct_dvc_term_end_date_max') >= f.col('movement_date')
            , 'IFP'
        )
        .otherwise('Plan Only')
    )
    .withColumn(
        'pre_2_post'
        , f.when(
            f.col('pre2post_date')<= f.col('movement_date')
            , 'Y'
        )
        .otherwise('N')
    )
)
        

# COMMAND ----------

display(df_output.count())

# COMMAND ----------

display(df_output) 

# COMMAND ----------

# Involuntary churn deactivation in Dec24
# Plan (eg Endless $45)
# Siebel ID (so I can link those with multi plan)
# Amount owed at time of disconnection
# Connection date (although not essential)

# COMMAND ----------

display(
    df_srvc_mvnt
    .limit(10)
)
