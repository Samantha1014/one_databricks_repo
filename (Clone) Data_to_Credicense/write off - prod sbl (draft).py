# Databricks notebook source
dir_prod_brm = 'dbfs:/mnt/prod_brm/raw/cdc'
dir_prod_siebel = 'dbfs:/mnt/prod_siebel/raw/cdc'

# brm 
df_item_t = spark.read.format('delta').load(os.path.join(dir_prod_brm, 'RAW_PINPAP_ITEM_T'))
df_account_t = spark.read.format('delta').load(os.path.join(dir_prod_brm,'RAW_PINPAP_ACCOUNT_T'))


# COMMAND ----------

 #siebel 
#df_sbl_org_ext = spark.read.format('delta').load(os.path.join(dir_prod_siebel, 'RAW_SBL_ORG_EXT'))
#df_sbl_contact = spark.read.format('delta').load(os.path.join(dir_prod_siebel, 'RAW_SBL_CONTACT'))


# COMMAND ----------

### wo aggreated into requsted format 
### ADO Link - https://dev.azure.com/vodafonenz/IT/_workitems/edit/318883
### ## 1st Upload will be for 5 years history based on write off Date followed by daily upload as incremental

df_wo_base_hist = spark.sql("""
    with temp_01 as (
        select
            acct.account_no
            ,  to_date(from_utc_timestamp(from_unixtime(item.created_t), 'Pacific/Auckland')) as writeoff_create_date
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
), temp_02 as 
    (select *
    from temp_01
    qualify row_number() over(partition by(account_no) order by writeoff_create_date asc,writeoff_amt asc) = 1
)
        select * from temp_02 
        where writeoff_amt <= -100 
        and writeoff_create_date >= add_months(current_date(), -60)
""")


# COMMAND ----------

# MAGIC %sql
# MAGIC with temp_01 as (
# MAGIC     select
# MAGIC         acct.account_no
# MAGIC         ,  to_date(from_utc_timestamp(from_unixtime(item.created_t), 'Pacific/Auckland')) as writeoff_create_date
# MAGIC         , item.item_total as writeoff_amt
# MAGIC     from delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ITEM_T` as item
# MAGIC     inner join delta.`/mnt/prod_brm/raw/cdc/RAW_PINPAP_ACCOUNT_T` as acct
# MAGIC         on item.account_obj_id0 = acct.poid_id0
# MAGIC         and acct._is_latest = 1
# MAGIC         and acct._is_deleted = 0
# MAGIC         and acct.account_no not like 'S%'
# MAGIC     where
# MAGIC         1 = 1
# MAGIC         and item._is_latest = 1
# MAGIC         and item._is_deleted = 0
# MAGIC         and item.poid_type in ('/item/writeoff')
# MAGIC ), temp_02 as 
# MAGIC (select *
# MAGIC from temp_01
# MAGIC   qualify row_number() over(partition by(account_no) order by writeoff_create_date asc,writeoff_amt asc) = 1
# MAGIC )
# MAGIC select * from temp_02 
# MAGIC where writeoff_amt <= -100 
# MAGIC and writeoff_create_date >= add_months(current_date(), -60)
# MAGIC

# COMMAND ----------

df_cust_id = (
    df_sbl_org_ext
    .filter(f.col('_is_latest') == 1)
    .filter(f.col('_is_deleted') ==0)
    .select(f.col('ou_num').alias('account_no')
            , f.col('row_id').alias('billing_acct_id')
            , f.col('par_ou_id').alias('customer_id')
            , 'master_ou_id')
    .filter(f.col('accnt_type_cd').isin('Billing'))
            )

# COMMAND ----------

display(df_sbl_org_ext
        .filter(f.col('_is_latest') == 1)
        .filter(f.col('_is_deleted') ==0)
        .select(
            f.col('row_id').alias('cust_id')
            ,'par_ou_id'
            ,'master_ou_id' 
        )
        .filter(f.col('accnt_type_cd').isin('Customer'))
        .filter(f.col('cust_id') =='1-K0F2RBB' )
        )

# COMMAND ----------

display(df_cust_id
        .filter(f.col('account_no') == '474858260'))

# COMMAND ----------

df_dob = (df_sbl_org_ext
        .filter(f.col('_is_latest') == 1)
        .filter(f.col('_is_deleted') == 0)
        .select(
            f.col('row_id').alias('customer_id')
            , 'pr_con_id', 'accnt_type_cd')
        .distinct()
        .filter(f.col('accnt_type_cd').isin('Customer'))
        .join(df_sbl_contact
              .filter(f.col('_is_latest') ==1)
              .filter(f.col('_is_deleted') ==0)
              .select('row_id' ,'FST_NAME', 'last_name', 'mid_name', 'birth_dt')
              .distinct()
              , f.col('pr_con_id') == f.col('row_id')
              , 'left'         
            )
        )

# COMMAND ----------

df_wo_hist = (df_wo_base.alias('a')
        .join(df_cust_id.alias('b'), f.col('a.account_no') == f.col('b.account_no'), 'left')
        .join(df_dob.alias('c'), f.col('b.customer_id') == f.col('c.customer_id'),'left')
        .select( 'a.account_no' , 'create_date', 'b.customer_id', 'FST_NAME', 'mid_name', 'last_name', 
                 'birth_dt', 'item_total'
                 )
        )
        # 104272 

# COMMAND ----------

display(df_wo_hist
        .withColumn('cnt' , f.count('*').over(Window.partitionBy('account_no')))
        .filter(f.col('cnt') >=2)
)

# COMMAND ----------

display(df_wo_base
        .withColumn('create_month', f.date_format('create_date','yyyy-MM'))
        .groupBy(f.col('create_month'))
        .agg(f.countDistinct('account_no'))
        )
