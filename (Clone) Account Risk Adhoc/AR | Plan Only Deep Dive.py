# Databricks notebook source
# MAGIC %md
# MAGIC #### library

# COMMAND ----------

import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC #### directory

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer"

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d200_intermediate/d299_shared/int_mobile_rated_usage_monthly')

# COMMAND ----------

df_int_usage = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d203_dp_dh/stg_dh_rated_usage_daily')
df_int_usage_daily = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_intermediate/d299_shared/int_mobile_rated_usage_daily')
df_int_usage_monthly = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_intermediate/d299_shared/int_mobile_rated_usage_monthly')

# COMMAND ----------

display(df_int_usage.limit(10))

# COMMAND ----------

df_prm_usage = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_rated_usage_cycle_calendar')
df_plan = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_plan')

# COMMAND ----------

# DBTITLE 1,usage aggregate
df_output_curr = (
                df_int_usage_daily
                .filter(f.col('usg_start_date') >= '2024-09-01')
                .withColumn(
                    "usg_type"
                    , f.lower(f.col("prod_usg_prod_grp_desc_lvl3"))
                )
                .withColumn(
                    "usg_type"
                    , f.when(f.col("usg_type") == "messaging", "msg").otherwise(f.col("usg_type"))
                )
                .withColumn(
                    "tariff_type"
                    , f.when(f.col("usg_rev_aft_disc_incl_gst_amt") > 0, f.lit("charged")).otherwise(f.lit("others"))
                )
                .groupBy(
                    "fs_srvc_id"
                    , "usg_type"
                    , "prod_usg_prod_name"
                    , "prod_tariff_prod_name"
                    , "usg_dirctn_code"
                    , "dist_cat"
                    , "rmr_class_name"
                    , "tariff_type"
                )
                .agg(
                    f.min("usg_start_date").alias("usg_date_min")
                    , f.max("usg_start_date").alias("usg_date_max")
                    , f.countDistinct("usg_start_date").alias("active_days")
                    , f.max("usg_start_date").alias("last_activity_date")
                    , f.collect_set("usg_start_date").alias("active_date_list")
                    , f.sum("chrgd_vol_qty").alias("usg_vol")
                    , f.sum("dur_tm").alias("usg_dur")
                    , f.sum("rated_usg_qty").alias("usg_qty")
                    , f.sum("usg_rev_aft_disc_incl_gst_amt").alias("usg_rev")   
                )
            )


# COMMAND ----------

display(
        df_output_curr
        .filter(f.col('fs_srvc_id') == '64211290392')
        #.select('rmr_class_name')
        #.distinct()
       #.filter(f.col('rmr_class_name') == 'Roaming')
       )

# COMMAND ----------

# DBTITLE 1,roaming
df_roaming = (
        df_output_curr
        .groupBy('fs_srvc_id')
        .pivot('rmr_class_name')
        .agg( f.sum('usg_qty')
             #,  f.sum('active_days')
             )
       )

# COMMAND ----------

# DBTITLE 1,usage type
df_usage_type = (
        df_output_curr
        .groupBy('fs_srvc_id')
        .pivot('usg_type')
        .agg( f.sum('usg_qty')
             #,  f.sum('active_days')
             )
        #.filter(f.col('fs_srvc_id')== '64211290392' )
       )

# COMMAND ----------

# DBTITLE 1,dist cat
df_dist_cat = (
        df_output_curr
        .groupBy('fs_srvc_id')
        .pivot('dist_cat')
        .agg( f.sum('usg_qty')
             #,  f.sum('active_days')
             )
       )

# COMMAND ----------

# MAGIC %md
# MAGIC #### data load

# COMMAND ----------

df_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_data_parent, 'fea_unit_base'))
df_prm_ifp_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_bill')
df_prm_ifp_srvc = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_ifp_main_on_service')

# COMMAND ----------

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
                .filter(f.col('ifp_type') == 'device')
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
        .filter(f.col('ifp_term_start_date') >= '2024-09-01')
        )

# COMMAND ----------

df_ifp_hist_agg = (df_ifp_history
        .groupBy('fs_acct_id')
        .agg(
            f.min('ifp_term_start_date').alias('min_ifp_start_date')
            , f.max('ifp_term_start_date').alias('max_ifp_start_date')
            ,  f.collect_set("ifp_model").alias('ifp_model_list')
            , f.countDistinct('fs_ifp_id').alias('cnt_ifp')
             )
)

# COMMAND ----------

df_bill_acct = (df_unit_base
        .filter(f.col('billing_acct_open_date') >= '2024-09-01')
        .select('fs_cust_id'
                , 'fs_acct_id'
                , 'fs_srvc_id'
               , 'cust_start_date'
                , 'billing_acct_open_date'
                , 'srvc_start_date'
              #  , 'first_activation_date'
                )
        .distinct()
        .withColumn('index', f.row_number().over(Window.partitionBy('fs_acct_id', 'fs_cust_id', 'fs_srvc_id')
                                                 .orderBy(f.desc('billing_acct_open_date'))
                                                 )
                    )
        .filter(f.col('index') == 1)
        )


# COMMAND ----------

display(df_bill_acct
        .agg(f.countDistinct('fs_acct_id')
             , f.countDistinct('fs_srvc_id')
             , f.count('*')
             )
        )

# COMMAND ----------

display(df_bill_acct
        .join(df_ifp_hist_agg, on='fs_acct_id', how='left')
        .join(df_roaming, ['fs_srvc_id'], 'left')
        .join(df_usage_type, ['fs_srvc_id'], 'left')
        .join(df_dist_cat, ['fs_srvc_id'], 'left')
        .withColumn('date_diff', f.date_diff('min_ifp_start_date', 'billing_acct_open_date'))
        #.withColumn('usage_col', f.when(f.col('fs_srvc_id').isNotNull(), 1).otherwise(0))
)
