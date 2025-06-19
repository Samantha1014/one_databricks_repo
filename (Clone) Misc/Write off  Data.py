# Databricks notebook source
import pyspark
from pyspark.sql import functions as f

# COMMAND ----------

vt_param_date_start = '2024-03-01'

# COMMAND ----------

dir_data_dl_brm = 'dbfs:/mnt/prod_brm/raw/cdc'

# COMMAND ----------

df_db_brm_wo = spark.sql(
    f"""
        with acct as (
            select 
                distinct account_no, poid_id0
            from delta.`{dir_data_dl_brm}/RAW_PINPAP_ACCOUNT_T`
        )
        
        , base_wo_hist as (
            select *
            from
                delta.`{dir_data_dl_brm}/RAW_PINPAP_ITEM_T`
            where
                _is_latest = 1
                and _is_deleted = 0
                and poid_type = '/item/writeoff'
        )
        , ranked_wo_hist as (
            select
                *
                , row_number() over (
                    partition by account_obj_id0
                    order by created_t asc, item_total asc
                ) as rnk
            from
                base_wo_hist
        )
        , filtered_wo_hist as (
            select
                *
                , date_format(
                    from_utc_timestamp(from_unixtime(created_t), 'Pacific/Auckland'),
                    'yyyy-MM-dd'
                ) as created_date
            from
                ranked_wo_hist 
            where
                rnk = 1
                --and item_total <= -500
        )
        , joined_wo_hist as (
            select
                a.account_no
                , b.created_date
                , b.item_total
            from
                filtered_wo_hist as b
            inner join
                acct as a
            on
                b.account_obj_id0 = a.poid_id0
        )
        select
            account_no
            , created_date
            , item_total
        from
            joined_wo_hist
        where
            created_date >= '{vt_param_date_start}'
    """
)

# display(df_db_brm_wo.limit(10))

# COMMAND ----------

# DBTITLE 1,check monthly wo acct
display(df_db_brm_wo
        .withColumn('date_month', f.date_format('created_date', 'yyyy-MM'))
        # .filter(f.col('item_total')<-50)
        .groupBy('date_month')
        .agg(f.countDistinct('account_no'))
        )


# COMMAND ----------

display(df_db_brm_wo)
