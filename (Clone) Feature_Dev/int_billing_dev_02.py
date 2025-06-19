# Databricks notebook source
# MAGIC %run "./stg_billing_dev"

# COMMAND ----------

# MAGIC %run "../Function"

# COMMAND ----------

dir_global_calendar_meta = 'dbfs:/mnt/feature-store-dev/dev_users/dev_sc/d000_meta/d001_global_cycle_calendar'
df_global_calendar_meta = spark.read.format('delta').load(dir_global_calendar_meta)

# COMMAND ----------



# COMMAND ----------

vt_param_ssc_reporting_date = '2022-07-31'
vt_param_ssc_reporting_cycle_type = 'calendar cycle'
#vt_param_ssc_prod_name = 'billing'
vt_param_lookback_cycles = 6
vt_param_lookback_cycle_unit_type = "months"
vt_param_lookback_units_per_cycle = 1


# COMMAND ----------

display(df_global_calendar_meta.filter(col('cycle_type') =='calendar cycle'))

# COMMAND ----------

# derive the closest finished billing cycles based on the current snapshot
df_param_partition_curr = get_lookback_cycle_meta(
    df_calendar=df_global_calendar_meta
    , vt_param_date=vt_param_ssc_reporting_date
    , vt_param_lookback_cycles=vt_param_lookback_cycles
    , vt_param_lookback_cycle_unit_type=vt_param_lookback_cycle_unit_type
    , vt_param_lookback_units_per_cycle=vt_param_lookback_units_per_cycle
)

display(df_param_partition_curr)

# COMMAND ----------

df_param_partition_summary_curr = (
    df_param_partition_curr
    .agg(
        f.min("partition_date").alias("date_min")
        , f.max("partition_date").alias("date_max")
    )
)

# COMMAND ----------

display(df_param_partition_summary_curr)
