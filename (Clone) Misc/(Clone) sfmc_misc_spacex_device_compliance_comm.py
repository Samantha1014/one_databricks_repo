# Databricks notebook source
# MAGIC %md
# MAGIC ### s1 environment setup

# COMMAND ----------

# DBTITLE 1,libraries
import pyspark
import os
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import monotonically_increasing_id, row_number

import numpy as np

import pandas as pd
import datetime as dt
from jinja2 import Template

from datetime import datetime
from datetime import date
from datetime import timedelta

# ------------- Use snowflake utility
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils

# COMMAND ----------

# DBTITLE 1,parameters
vt_param_current_date_index = datetime.now().strftime("%Y%m%d")
vt_param_current_date_index

# COMMAND ----------

# DBTITLE 1,sf connection
# ------------ login to snowflake
password = dbutils.secrets.get(scope = "auea-kv-sbx-dxdtlprdct01", key = "sfdbrsdskey")

options = {
  "sfUrl": "vodafonenz_prod.australia-east.azure.snowflakecomputing.com/", 
  "sfUser": "SVC_LAB_DS_DATABRICKS",
  "pem_private_key": password.replace('\\n', '\n'),
  "sfDatabase": "LAB_ML_STORE",
  "sfSchema": "RAW",
  "sfWarehouse": "LAB_DS_WH_SCALE"
}


# COMMAND ----------

# DBTITLE 1,utility functions
# MAGIC %run "./utility_functions/utility_functions"

# COMMAND ----------

# DBTITLE 1,directories
dir_data_master = "/mnt/ml-lab/dev_shared/tactical_solutions/spacex_device_comm"
dir_data_sfmc = "/mnt/prod_sfmc/imports/ml_store"
dir_data_sfmc_misc = os.path.join(dir_data_sfmc, "misc")

# COMMAND ----------

# MAGIC %md
# MAGIC ### s2 data import

# COMMAND ----------

# DBTITLE 1,network device extract
dir_data_wip_spacex_comm_srvc = os.path.join(dir_data_master, "raw_spacex_comm_srvc")

df_raw_spacex_comm_srvc = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            with base_srvc_consumer as (
                --- get latest active mobile service from MARTECH DB
                select 
                    customer_id
                    , service_id
                    , mobile_number
                    , service_type_name
                    , service_access_type_name
                from prod_mar_tech.serving.export_service as a
                where 
                    1 = 1
                    and service_access_type_name = 'Mobile'
                    and service_type_name != 'Broadband'
                qualify row_number() over (
                    partition by service_id
                    order by 
                        connection_activation_date desc
                ) = 1
            )

            , base_srvc_ent as (
                --- get latest active mobile service from MARTECH DB
                select 
                    customer_id
                    , service_id
                    , mobile_number
                    , service_type_name
                    , service_access_type_name
                from prod_mar_tech.serving.export_service_ent as a
                where 
                    1 = 1
                    and service_access_type_name = 'Mobile'
                    and service_type_name != 'Broadband'
                qualify row_number() over (
                    partition by service_id
                    order by 
                        connection_activation_date desc
                ) = 1
            )

            , base_srvc_00 as (
                select * from base_srvc_consumer
                union all
                select * from base_srvc_ent
            )

            , base_srvc as (
                select * from base_srvc_00
                qualify row_number() over (
                    partition by service_id
                    order by 1
                ) = 1
            )

            , base_gsma as (
                --- get latest GSMA device model from GSMA DB
                select
                    tac
                    , gsma__marketing_name as network_dvc_model
                    , standardised_marketing_name as network_dvc_model_marketing
                    , _updated_timestamp
                    , _valid_from_timestamp
                    , _valid_to_timestamp
                from prod_gsma.raw.devicemap
                qualify row_number() over (
                    partition by tac
                    order by 
                        _updated_timestamp desc
                        , _valid_to_timestamp desc
                        , _valid_from_timestamp desc
                    
                ) = 1
            )

            , base_dvc_latest as (
                --- get latest network device record from CDR in WNI DB
                select
                    base.customer_id
                    , base.service_id
                    , base.mobile_number
                    , base.service_type_name
                    , base.service_access_type_name
                    , dvc.datetime as network_dvc_record_dttm
                    , dvc.imei
                    , left(dvc.imei, 8) as tac
                from prod_wni.serving_pii.f_netscout_call_records as dvc
                inner join base_srvc as base
                    on cast(dvc.msisdn as string) = cast(base.service_id as string)
                where 
                    1 = 1
                    and dvc.datetime between dateadd(day, -1, current_date) and dateadd(day, 1, current_date)
                    and cast(left(dvc.imsi, 5) as string) = '53001'
                qualify row_number() over (
                    partition by dvc.msisdn
                    order by dvc.datetime desc
                ) = 1
            )

            , base_dvc_hist as (
                --- get network device record between 31 days ago and 1 day ago from CDR in WNI DB
                select
                    base.customer_id
                    , base.service_id
                    , base.mobile_number
                    , base.service_type_name
                    , base.service_access_type_name
                    , dvc.datetime as network_dvc_record_dttm
                    , dvc.imei
                    , left(dvc.imei, 8) as tac
                from prod_wni.serving_pii.f_netscout_call_records as dvc
                inner join base_srvc as base
                    on cast(dvc.msisdn as string) = cast(base.service_id as string)
                left join base_dvc_latest as latest
                    on cast(dvc.msisdn as varchar) = cast(latest.service_id as varchar) 
                where 
                    1 = 1
                    and dvc.datetime between dateadd(day, -31, current_date) and dateadd(day, -1, current_date)
                    and cast(left(dvc.imsi, 5) as string) = '53001'
                    and latest.service_id is null
                qualify row_number() over (
                    partition by dvc.msisdn
                    order by dvc.datetime desc
                ) = 1
            )

            , base_dvc as (
                --- combine latest and historical network device record
                select * from base_dvc_latest
                union all
                select * from base_dvc_hist
            )

            , final as (
                --- get final output
                select
                    dvc.customer_id
                    , dvc.service_id
                    , dvc.service_access_type_name
                    , dvc.service_type_name
                    , dvc.imei
                    , dvc.tac
                    , gsma.network_dvc_model
                    , gsma.network_dvc_model_marketing
                    , dvc.network_dvc_record_dttm
                    , current_date as data_update_date
                    , current_timestamp as data_update_dttm
                from base_dvc as dvc
                left join base_gsma as gsma
                    on dvc.tac = gsma.tac
                where
                    1 = 1
                    and dvc.imei is not null
            )

            select * from final
        """
    )
    .load()
)

#df_raw_spacex_comm_srvc = lower_col_names(df_raw_spacex_comm_srvc)

# export_data(
#     df = df_raw_spacex_comm_srvc
#     , export_path = dir_data_wip_spacex_comm_srvc
#     , export_format = "delta"
#     , export_mode = "overwrite"
#     , flag_overwrite_schema = True
#     , flag_dynamic_partition = False    
# )

df_raw_spacex_comm_srvc = spark.read.format("delta").load(dir_data_wip_spacex_comm_srvc)


# COMMAND ----------

# DBTITLE 1,data check
print("df_raw_spacex_comm_srvc")
display(df_raw_spacex_comm_srvc.count())
display(df_raw_spacex_comm_srvc.limit(10))

# COMMAND ----------

# DBTITLE 1,martech service base
dir_data_wip_martech_srvc = os.path.join(dir_data_master, "raw_martech_srvc")

df_raw_martech_srvc = (
    spark
    .read
    .format("snowflake")
    .options(**options)
    .option(
        "query"
        , """
            with consumer as (
                select
                    customer_id
                    , service_id
                    , mobile_number
                    , service_type_name
                    , service_access_type_name
                from prod_mar_tech.serving.export_service
                where 
                    1 = 1
                    and service_access_type_name = 'Mobile'
                    and service_type_name != 'Broadband'
                qualify row_number() over (
                    partition by service_id
                    order by 
                        connection_activation_date desc
                ) = 1
            )
            , ent as (
                select
                    customer_id
                    , service_id
                    , mobile_number
                    , service_type_name
                    , service_access_type_name
                from prod_mar_tech.serving.export_service_ent
                where 
                    1 = 1
                    and service_access_type_name = 'Mobile'
                    and service_type_name != 'Broadband'
                qualify row_number() over (
                    partition by service_id
                    order by 
                        connection_activation_date desc
                ) = 1
            )

            , base as (
                select * from consumer
                union all
                select * from ent
            )

            select *
            from base
            qualify row_number() over (
                partition by service_id
                order by 1
            ) = 1
        """
    )
    .load()
)

df_raw_martech_srvc = lower_col_names(df_raw_martech_srvc)

# export_data(
#     df = df_raw_martech_srvc
#     , export_path = dir_data_wip_martech_srvc
#     , export_format = "delta"
#     , export_mode = "overwrite"
#     , flag_overwrite_schema = True
#     , flag_dynamic_partition = False    
# )

df_raw_martech_srvc = spark.read.format("delta").load(dir_data_wip_martech_srvc)


# COMMAND ----------

# DBTITLE 1,data check 01
print("df_raw_martech_srvc")
display(df_raw_martech_srvc.count())
display(df_raw_martech_srvc.limit(10))

# COMMAND ----------

# DBTITLE 1,data check 02
display(
    df_raw_martech_srvc
    .groupBy(
        "service_type_name"
    )
    .agg(
        f.count("*")
        , f.countDistinct("service_id")
        , f.countDistinct("mobile_number")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### s3 data transform

# COMMAND ----------

df_output = (
    df_raw_spacex_comm_srvc
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy("service_id")
            .orderBy(f.desc("network_dvc_record_dttm"))
        )
    )
    .filter(f.col("index") == 1)
    .join(
        df_raw_martech_srvc
        .select(
            f.col("customer_id")
            , "service_id"
        )
        , ["customer_id", "service_id"]
        , "left"
    )
    .select(
        f.col("customer_id").alias("contact_id")
        , "service_id"
        , "service_access_type_name"
        , "service_type_name"
        , "imei"
        , "tac"
        , "network_dvc_model"
        , "network_dvc_model_marketing"
        , "network_dvc_record_dttm"
        , "data_update_date"
        , "data_update_dttm"
    )
)

# COMMAND ----------

# DBTITLE 1,output creation
df_output = (
    df_raw_spacex_comm_srvc
    .withColumn(
        "index"
        , f.row_number().over(
            Window
            .partitionBy("service_id")
            .orderBy(f.desc("network_dvc_record_dttm"))
        )
    )
    .filter(f.col("index") == 1)
    .join(
        df_raw_martech_srvc
        .select(
            f.col("customer_id")
            , "service_id"
        )
        , ["customer_id", "service_id"]
        , "inner"
    )
    .select(
        f.col("customer_id").alias("contact_id")
        , "service_id"
        , "service_access_type_name"
        , "service_type_name"
        , "imei"
        , "tac"
        , "network_dvc_model"
        , "network_dvc_model_marketing"
        , "network_dvc_record_dttm"
        , "data_update_date"
        , "data_update_dttm"
    )
)

# COMMAND ----------

# DBTITLE 1,data check
display(df_output.count())
display(
    df_output
    .groupBy("service_type_name")
    .agg(
        f.count("*")
        , f.countDistinct("service_id")
    )
)
display(df_output.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### s4 data export

# COMMAND ----------

# DBTITLE 1,data export
vt_param_segment = "spacex_device_compliance_comm"

os.makedirs(f"/dbfs{dir_data_sfmc_misc}/{vt_param_segment}", exist_ok=True)

df_export = df_output

print(df_export.count())
display(df_export.limit(100))

(
    df_export
    .toPandas()
    .to_csv(f"/dbfs{dir_data_sfmc_misc}/{vt_param_segment}/network_dvc_{vt_param_current_date_index}.csv", index=False)
)

# COMMAND ----------

display(
    df_export
    .withColumn(
        "cnt"
        , f.count("*").over(Window.partitionBy("service_id"))
    )
    .filter(f.col("cnt") > 1)
    .limit(100)
)
