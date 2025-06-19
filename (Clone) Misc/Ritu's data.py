# Databricks notebook source
import os
import pyspark
from pyspark import sql
from pyspark.sql import functions as f
from pyspark.sql import Window

# COMMAND ----------

df_ritu = spark.read.format('csv').option('header', 'true').load('dbfs:/FileStore/mnt/ml-lab/dev_users/dev_sc/Ritu_Data.csv')

# COMMAND ----------

dir_fs_data_parent = "/mnt/feature-store-prod-lab"

# COMMAND ----------

# dir_fs_data_meta = os.path.join(dir_fs_data_parent, 'd000_meta')
# dir_fs_data_raw =  os.path.join(dir_fs_data_parent, 'd100_raw')
# dir_fs_data_int =  os.path.join(dir_fs_data_parent, "d200_intermediate")
# dir_fs_data_prm =  os.path.join(dir_fs_data_parent, "d300_primary")
dir_fs_data_fea =  os.path.join(dir_fs_data_parent, "d400_feature")
dir_fs_data_target = os.path.join(dir_fs_data_parent, "d500_movement")
dir_fs_data_serv = os.path.join(dir_fs_data_parent, "d600_serving")

# COMMAND ----------

#df_fs_master = spark.read.format("delta").load(os.path.join(dir_fs_data_serv, "serv_mobile_oa_consumer"))
df_fs_ifp_srvc = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_service"))
df_fs_ifp_bill = spark.read.format("delta").load(os.path.join(dir_fs_data_fea, "d401_mobile_oa_consumer/fea_ifp_device_on_bill"))
#df_fs_ifp_device = spark.read.format('delta').load(os.path.join(dir_fs_data_fea, 'd401_mobile_oa_consumer/fea_ifp_device_account'))
df_fs_unit_base = spark.read.format('delta').load(os.path.join(dir_fs_data_fea, 'd401_mobile_oa_consumer/fea_unit_base'))
df_fs_deact = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation')
# /mnt/feature-store-prod-lab/d500_movement/d501_mobile_oa_consumer/mvmt_service_deactivation

# meta for fields 
#df_fs_meta = spark.read.format('delta').load(os.path.join(dir_fs_data_meta,'d004_fsr_meta','fsr_field_meta'))

# COMMAND ----------

dbutils.fs.ls('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_ifp_accessory_on_bill')

# COMMAND ----------

df_ifp_on_bill_acc = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_ifp_accessory_account')

df_ifp_access_on_bill = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_ifp_accessory_on_bill')

# COMMAND ----------

df_accessory = (df_ifp_access_on_bill
        .select('fs_acct_id'
                , 'ifp_bill_accs_term_start_date'
                , 'ifp_bill_accs_term_end_date'
                , 'ifp_bill_accs_type'
                , 'ifp_bill_accs_sales_channel'
                , 'ifp_bill_accs_sales_channel_group'
                )
        .distinct()
        )

# COMMAND ----------

display(df_accessory
        .filter(f.col('fs_acct_id') == '504074907')
)

# COMMAND ----------

display(df_ritu
        .join(df_fs_unit_base
              .select('fs_acct_id')
              .distinct()
              , f.col('Account Ref No') == f.col('fs_acct_id'), 'inner')
        )

# COMMAND ----------

display(df_fs_ifp_bill.limit(3))

# COMMAND ----------

df_ifp_on_serv= (df_fs_ifp_srvc
        .filter(f.col('fs_ifp_srvc_dvc_id').isNotNull())
        .select('fs_acct_id'
                , 'fs_srvc_id'
                , 'fs_ifp_srvc_dvc_id'
                , 'ifp_srvc_dvc_term_start_date'
                , 'ifp_srvc_dvc_term_end_date'
                , 'ifp_srvc_dvc_sales_channel_group'
                , 'ifp_srvc_dvc_sales_channel'
                , 'ifp_srvc_dvc_sales_channel_branch'
                )
        .distinct()
)

# COMMAND ----------

display(df_ritu
        .join(df_fs_ifp_bill
              .filter(f.col('ifp_bill_dvc_flag') == 'Y')
              .select('fs_acct_id'
                      , 'ifp_bill_dvc_term_start_date'
                      , 'ifp_bill_dvc_term_end_date'
                      , 'ifp_bill_dvc_term'
                      , 'ifp_bill_dvc_sales_channel_group'
                      , 'ifp_bill_dvc_sales_channel'
                      )
              .distinct()
              , f.col('Account Ref No') == f.col('fs_acct_id'), 'left')
        .join(df_ifp_on_serv.alias('b'), f.col('Account Ref No') == f.col('b.fs_acct_id'), 'left')
        .join(df_accessory.alias('c'), f.col('Account Ref No') == f.col('c.fs_acct_id') , 'left')
        
        .withColumn('combine_start_date', f.coalesce('ifp_srvc_dvc_term_start_date', 'ifp_bill_dvc_term_start_date'))
        .withColumn('combine_end_date', f.coalesce('ifp_srvc_dvc_term_end_date', 'ifp_bill_dvc_term_end_date'))
        .withColumn('combine_sales_channel', f.coalesce('ifp_srvc_dvc_sales_channel', 'ifp_bill_dvc_sales_channel' ))
        .withColumn('combine_sales_channel_group', f.coalesce('ifp_srvc_dvc_sales_channel_group', 'ifp_bill_dvc_sales_channel_group'))
        )

# COMMAND ----------

display(df_ritu
        .join(df_accessory, f.col('Account Ref No') == f.col('fs_acct_id'), 'left')
        )

# COMMAND ----------

display(

    df_ifp_access_on_bill.filter(f.col('fs_acct_id') == '504074907')
)


