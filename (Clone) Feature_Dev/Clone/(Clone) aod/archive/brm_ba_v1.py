# Databricks notebook source
# MAGIC %md ## s1 environment setup

# COMMAND ----------

# DBTITLE 1,libraries
### libraries
import pyspark
import os

import re
import numpy as np

from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import monotonically_increasing_id, row_number

from itertools import islice, cycle

# COMMAND ----------

# MAGIC %md ### utility functions

# COMMAND ----------

# DBTITLE 1,spkdf
# MAGIC %run "../../../utility_functions/spkdf_utils"

# COMMAND ----------

# DBTITLE 1,qa
# MAGIC %run "../../../utility_functions/qa_utils"

# COMMAND ----------

# DBTITLE 1,misc
# MAGIC %run "../../../utility_functions/misc"

# COMMAND ----------

# MAGIC %md ### directories

# COMMAND ----------

# DBTITLE 1,data lake
dir_data_dl_brm = "/mnt/prod_brm/raw/cdc"
dir_data_dl_edw = "/mnt/prod_edw/raw/cdc"

# COMMAND ----------

# DBTITLE 1,feature store 01
dir_data_parent = "/mnt/feature-store-dev"
dir_data_parent_shared = os.path.join(dir_data_parent, "dev_shared")
dir_data_parent_users = os.path.join(dir_data_parent, "dev_users/dev_el/2024q4_mobile_oa_account_risk")

# COMMAND ----------

# DBTITLE 1,feature store 02
dir_data_raw =  os.path.join(dir_data_parent_shared, 'd100_raw')
dir_data_meta = os.path.join(dir_data_parent_users, 'd000_meta')
dir_data_int =  os.path.join(dir_data_parent_users, "d200_intermediate")
dir_data_prm =  os.path.join(dir_data_parent_users, "d300_primary/d302_mobile_pp")
dir_data_fea =  os.path.join(dir_data_parent_users, "d400_feature/d402_mobile_pp")
dir_data_mvmt = os.path.join(dir_data_parent_users, "d500_movement/d502_mobile_pp")
dir_data_serv = os.path.join(dir_data_parent_users, "d600_serving")
dir_data_tmp =  os.path.join(dir_data_parent_users, "d999_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## s1 data import

# COMMAND ----------

df_raw_brm_bill_t = spark.read.format("delta").load(os.path.join(dir_data_dl_brm, "RAW_PINPAP_BILL_T"))

# COMMAND ----------

display(
    df_raw_brm_bill_t
    .limit(100)
)

# COMMAND ----------

df_raw_brm_bill_t.createOrReplaceTempView("vw_temp")

# COMMAND ----------

spark = (
    SparkSession
    .builder
    .config("spark.sql.caseSensitive", "false")
    .getOrCreate()
)

# COMMAND ----------

df_check = spark.sql(
    f"""
        select
            regexp_replace(acct.account_no, '^0+', '') as fs_acct_id
            , bill.poid_id0
            , bill.account_obj_id0
            , acct.account_no
            , bill.invoice_obj_id0
            , bill.billinfo_obj_id0
            , bill.ar_billinfo_obj_id0
            , bill.bill_no
            , bill.created_t
            , from_utc_timestamp(from_unixtime(bill.created_t), 'Pacific/Auckland') as created_dttm
            , bill.mod_t
            , from_utc_timestamp(from_unixtime(bill.mod_t), 'Pacific/Auckland') as mod_dttm
            , bill.start_t
            , from_utc_timestamp(from_unixtime(bill.start_t), 'Pacific/Auckland') as start_dttm
            , bill.end_t
            , from_utc_timestamp(from_unixtime(bill.end_t), 'Pacific/Auckland') as end_dttm
            , bill.due_t
            , from_utc_timestamp(from_unixtime(bill.due_t), 'Pacific/Auckland') as due_dttm
            , bill.closed_t
            , from_utc_timestamp(from_unixtime(bill.closed_t), 'Pacific/Auckland') as close_dttm
            , bill.currency
            , bill.previous_total
            , bill.current_total
            , bill.subords_total
            , bill.total_due
            , bill.due
            , bill.recvd
            , bill.writeoff
            , bill.transferred
        from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_BILL_T` as bill
        inner join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T` as acct
            on bill.account_obj_id0 = acct.poid_id0
            and acct._is_latest = 1
            and acct._is_deleted = 0
            and acct.account_no not like 'S%'
        where
            1 = 1
            and bill._is_latest = 1
            and bill._is_deleted = 0
            and from_utc_timestamp(from_unixtime(bill.mod_t), 'Pacific/Auckland') between '2023-01-02' and '2023-01-05'
        limit 100
    """
)

# COMMAND ----------

display(df_check.limit(100))

# COMMAND ----------

df_check = spark.sql(
    f"""
        select
            regexp_replace(acct.account_no, '^0+', '') as fs_acct_id
            , item.poid_id0 as item_poid_id0
            , item.item_no            
            , item.poid_type as item_poid_type
            , item.billinfo_obj_id0
            , item.account_obj_id0
            , acct.account_no
            
            , item.created_t
            , from_utc_timestamp(from_unixtime(item.created_t), 'Pacific/Auckland') as created_dttm
            , item.mod_t
            , from_utc_timestamp(from_unixtime(item.mod_t), 'Pacific/Auckland') as mod_dttm
            , item.effective_t
            , from_utc_timestamp(from_unixtime(item.effective_t), 'Pacific/Auckland') as effective_dttm
            
            , item.item_total

            , ebit.obj_id0 as ebit_obj_id0
            ---, ebit.account_obj_id0
            , ebit.amount as ebit_amount
            ---, ebit.item_obj_id0
            ---, ebit.item_obj_type
            ---, ebpt.obj_id0
            ---, ebpt.amount
            , ebpt.trans_id as ebpt_trans_id
            , ebpt.pay_type as ebpt_rec_id
            ---, ebpt.account_no
            ---, cpt.rec_id
            , cpt.payinfo_type
            , cpt.payment_event_type
            , cpt.refund_event_type
            
            , ebpt.channel_id as ebpt_channel_id
            ---, ccm.channel_id
            , ccm.source as payment_source
            , ccm.subtype as payment_subtype
            , ccm.type as payment_type

        from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T` as item
        inner join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T` as acct
            on item.account_obj_id0 = acct.poid_id0
            and acct._is_latest = 1
            and acct._is_deleted = 0
            and acct.account_no not like 'S%'

        left join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BAL_IMPACTS_T` as ebit
            on item.poid_id0 = ebit.item_obj_id0
            and ebit._is_latest = 1
            and ebit._is_deleted = 0
            and ebit.item_obj_type in ('/item/payment', '/item/adjustment')

        left join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_EVENT_BILLING_PAYMENT_T` as ebpt
            on ebit.obj_id0 = ebpt.obj_id0
            and ebpt._is_latest = 1
            and ebpt._is_deleted = 0

        left join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_PAY_TYPES_T` cpt
            on ebpt.pay_type = cpt.rec_id
            and cpt._is_latest = 1
            and cpt._is_deleted = 0

        left join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_CONFIG_PAYMENT_CHANNEL_MAP_T` ccm
            on ebpt.channel_id = ccm.channel_id
            and ccm._is_latest = 1
            and ccm._is_deleted = 0

        where
            1 = 1
            and item._is_latest = 1
            and item._is_deleted = 0
            and item.poid_type in ('/item/payment', '/item/adjustment')
            and from_utc_timestamp(from_unixtime(item.mod_t), 'Pacific/Auckland') between '2023-01-02' and '2023-01-05'
        limit 100
    """
)

# COMMAND ----------

display(
    df_check
    .limit(100)
)

# COMMAND ----------

df_check = spark.sql(
    f"""
        select
        regexp_replace(acct.account_no, '^0+', '') as fs_acct_id
        , item.poid_id0 as item_poid_id0
        , item.item_no            
        , item.poid_type as item_poid_type
        , item.account_obj_id0
        , acct.account_no
        , item.created_t as writeoff_create_t
        , from_unixtime(item.created_t) as writeoff_create_dttm
        , to_date(from_unixtime(item.created_t)) as writeoff_create_date
        , item.mod_t as writeoff_mod_t
        , from_unixtime(item.mod_t) as writeoff_mod_dttm
        , to_date(from_unixtime(item.mod_t)) as writeoff_mod_date
        , item.effective_t as writeoff_effective_t
        , from_unixtime(item.effective_t) as writeoff_effective_dttm
        , to_date(from_unixtime(item.effective_t)) as writeoff_effective_date
        , item.item_total as writeoff_amt

    from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T` as item
    inner join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T` as acct
        on item.account_obj_id0 = acct.poid_id0
        and acct._is_latest = 1
        and acct._is_deleted = 0
        and acct.account_no not like 'S%'
    where
        1 = 1
        and item._is_latest = 1
        and item._is_deleted = 0
        and item.poid_type in ('/item/writeoff')
            and from_utc_timestamp(from_unixtime(item.mod_t), 'Pacific/Auckland') between '2023-01-02' and '2023-01-05'
        limit 100
    """
)

# COMMAND ----------

display(df_check.limit(100))
