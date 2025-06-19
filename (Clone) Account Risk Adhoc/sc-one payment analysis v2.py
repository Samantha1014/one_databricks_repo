# Databricks notebook source
import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql.functions import from_unixtime,from_utc_timestamp
from pyspark.sql.functions import date_format

# COMMAND ----------

df_wo_base = spark.read.format('delta').load('/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/WRITEOFF_BASE/')
df_payment_base = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d200_staging/d299_src/stg_brm_payment_hist')

# COMMAND ----------

df_fea_oa = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d600_serving/serv_mobile_oa_consumer/reporting_cycle_type=calendar cycle')

# COMMAND ----------

df_fea_ifp_accessory = spark.read.format('delta').load('/mnt/feature-store-prod-lab/d400_feature/d401_mobile_oa_consumer/fea_ifp_accessory_account/reporting_cycle_type=calendar cycle')

df_payment_arrange = spark.read.format('delta').load('/mnt/feature-store-dev/dev_users/dev_sc/d100_raw/payment_arrange') 

# COMMAND ----------

df_prm_bill = spark.read.format("delta").load("/mnt/feature-store-prod-lab/d300_primary/d301_mobile_oa_consumer/prm_bill_cycle_billing_6")

# COMMAND ----------

df_ifp_base = (
    df_fea_oa.select(
        "fs_acct_id",
        "ifp_prm_dvc_term_start_date",
        "ifp_prm_dvc_term_end_date",
        "ifp_prm_dvc_sales_channel_group",
        "ifp_prm_dvc_model",
        "ifp_prm_dvc_term",
        "fs_ifp_prm_dvc_id",
        "network_dvc_type",
        "ifp_prm_dvc_discount",
        "ifp_prm_dvc_discount_flag",
        "ifp_prm_dvc_pmt_monthly",
        "ifp_prm_dvc_pmt_net_monthly", 
        "ifp_acct_dvc_pmt_monthly_tot",
        "ifp_acct_accs_pmt_monthly_tot",
        "network_dvc_hardware_rating",

    )
    .filter(f.col("reporting_date") >= "2023-01-01")
    .filter(f.col("ifp_prm_dvc_term_start_date").isNotNull())
    .distinct()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write off Trend

# COMMAND ----------

display(df_wo_base
        .select('fs_acct_id', 'write_off_item_amount', 'rec_created_dttm')
        .distinct()
        .withColumn('rnk', f.row_number().over(Window.partitionBy('fs_acct_id')
                                               .orderBy(f.asc('rec_created_dttm'), f.asc('write_off_item_amount')))
        )
        .filter(f.col('rnk') == 1)
        .filter(f.col('write_off_item_amount') <=-50)
        .groupBy(f.date_format('rec_created_dttm', 'yyyy-MM'))
        .agg(f.countDistinct('fs_acct_id')
             , f.sum('write_off_item_amount')
             )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Late Payment Trend

# COMMAND ----------

display(
    df_prm_bill
    .filter(f.col("total_due") > 0)
    .select("bill_no", "fs_acct_id", "bill_due_date", "bill_close_date")
    .filter(f.col("bill_due_date") >= "2023-04-01")
    .filter(f.col("bill_due_date") <= "2024-03-31")
    .distinct()
    .withColumn("date_month", f.date_format("bill_due_date", "yyyy-MM"))
    # .filter(f.col('bill_close_date') != '1970-01-01')
    .withColumn("od_days", f.datediff("bill_close_date", "bill_due_date"))
    .withColumn(
        "reporting_od_days",
        f.when(
            f.col("bill_close_date") == "1970-01-01",
            f.datediff(f.last_day("bill_due_date"), f.col("bill_due_date")),
        ).otherwise(f.col("od_days")),
    )
    .filter(f.col("reporting_od_days") > 1)
    .groupBy("date_month")
    .agg(
        f.count("bill_no"),
        f.countDistinct("fs_acct_id"),
        f.avg("reporting_od_days"),
        f.median("reporting_od_days"),
    )
)

# COMMAND ----------

# DBTITLE 1,late pay bill proportion
display(
    df_prm_bill
    .filter(f.col("total_due") > 0)
    .select("bill_no", "fs_acct_id", "bill_due_date", "bill_close_date")
    .filter(f.col("bill_due_date") >= "2023-04-01")
    .filter(f.col("bill_due_date") <= "2024-03-31")
    .distinct()
    .withColumn("date_month", f.date_format("bill_due_date", "yyyy-MM"))
    # .filter(f.col('bill_close_date') != '1970-01-01')
    .withColumn("od_days", f.datediff("bill_close_date", "bill_due_date"))
    .withColumn(
        "reporting_od_days",
        f.when(
            f.col("bill_close_date") == "1970-01-01",
            f.datediff(f.last_day("bill_due_date"), f.col("bill_due_date")),
        ).otherwise(f.col("od_days")),
    )
    .withColumn(
        "overdue_flag"
        , f.when(
            f.col('reporting_od_days') > 1
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .groupBy("date_month", 'overdue_flag')
    .agg(
        f.count("bill_no"),
        f.countDistinct("fs_acct_id"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### wo base  payment

# COMMAND ----------

df_one_payment = (
    df_payment_base
    .filter(f.col("item_poid_type") == "/item/payment")
    .filter(f.col("item_total") < 0)
    .join(df_wo_base, ["fs_acct_id"], "inner")
    .filter(f.col("payment_create_dttm") <= f.col("rec_created_dttm"))
    .groupBy("fs_acct_id")
    .agg(
        f.countDistinct("item_no").alias("payment_cnt")
        # , f.countDistinct('fs_acct_id')
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### IFP and WO

# COMMAND ----------

df_wo_ifp = (
    df_wo_base
    .select("rec_created_dttm", "fs_acct_id", "write_off_item_amount")
    # .withColumn('wo_month', f.date_format('rec_created_dttm', 'yyyy-MM'))
    .join(df_ifp_base.alias("b"), ["fs_acct_id"], "inner")
    .filter(f.col("rec_created_dttm") >= "2023-01-01")
    .filter(f.col("ifp_prm_dvc_term_start_date") < f.col("rec_created_dttm"))
    .filter(f.col("ifp_prm_dvc_term_end_date") > f.col("rec_created_dttm"))
    .withColumn(
        "ifp_tenure(month)",
        f.round(
            f.datediff(f.col("rec_created_dttm"), f.col("ifp_prm_dvc_term_start_date"))
            / 30.25,
            1,
        ),
    )
    .withColumn(
        "ifp_period_left",
        f.round(
            f.datediff(f.col("ifp_prm_dvc_term_end_date"), f.col("rec_created_dttm"))
            / 30.25,
            1,
        ),
    )
    .filter(f.col("ifp_period_left") >= 1)
    .withColumn(
        "device_cnt",
        f.count("fs_ifp_prm_dvc_id").over(Window.partitionBy("fs_acct_id")),
    )
    .withColumnRenamed("rec_created_dttm", "writeoff_date")
)

# COMMAND ----------

# DBTITLE 1,accesory wo base
df_wo_acces = (
    df_wo_base
    .select("rec_created_dttm", "fs_acct_id", "write_off_item_amount")
    .join(df_fea_ifp_accessory
        .select('fs_acct_id', 'ifp_acct_accs_flag', 'ifp_acct_accs_cnt'
                , 'ifp_acct_accs_term_end_date_max'
                )
        .distinct()
        .filter(f.col('ifp_acct_accs_flag') == f.lit('Y'))
        , ['fs_acct_id'], 'left'
         )
    .filter( (f.col('rec_created_dttm') <= f.col('ifp_acct_accs_term_end_date_max')) | 
       (f.col('ifp_acct_accs_term_end_date_max').isNull()))
    .withColumn('acc_flag', 
                f.when(f.col('ifp_acct_accs_flag').isNull(), 'no accesory'
                )
                .when(f.col('ifp_acct_accs_cnt') == 1,  'one accesory'
                      )
                .when(f.col('ifp_acct_accs_cnt')>1 , 'multi accesory')
                .otherwise(f.lit('misc'))
)
    .filter(f.col('rec_created_dttm')>='2023-01-01')
    .filter(f.col('write_off_item_amount') <=-50)
    .select('fs_acct_id',  'ifp_acct_accs_flag', 'ifp_acct_accs_cnt', 'ifp_acct_accs_term_end_date_max', 'acc_flag')
    .distinct()
)

# COMMAND ----------

# DBTITLE 1,wo with payment, ifp, accsoery
df_wo_ifp_final = (
    df_wo_base
    .select("fs_acct_id", "rec_created_dttm", f.col('write_off_item_amount').alias('base_wo_amt'))
    .withColumn(
          "wo_month"
          , date_format("rec_created_dttm", "yyyy-MM")
      )
    .join(df_wo_ifp, ["fs_acct_id"], "left")
    .withColumn(
        "ifp_cat",
        f.when(f.col("device_cnt").isNull(), f.lit("no_device"))
        .when(f.col("device_cnt") == 1, f.lit("one_device"))
        .when(f.col("device_cnt") > 1, f.lit("multi_device"))
        .otherwise("unknown"),
    )
    .join(df_one_payment, ["fs_acct_id"], "left")
    .withColumn(
        "payment_cat",
        f.when(f.col("payment_cnt").isNull(), f.lit("neverpay"))
        .when(f.col("payment_cnt") == 1, f.lit("one_pay"))
        .when(f.col("payment_cnt") > 1, f.lit("one_plus"))
        .otherwise(f.lit("unknown")),
    )
    .join(df_wo_acces
          , ['fs_acct_id'], 'left')
    .withColumn('acces_in_wo', f.when(f.col('ifp_acct_accs_flag').isNull(), f.lit('N'))
                                .otherwise(f.lit('Y'))
                )
    .filter(f.col("wo_month") >= "2023-01")
    .filter(f.col('base_wo_amt') <=-50)
    # .groupBy('wo_month', 'ifp_cat')
    # .agg(f.countDistinct('fs_acct_id')
    #      , f.countDistinct('fs_ifp_prm_dvc_id')
    #     )
)

# COMMAND ----------

display(df_wo_ifp_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ###payment arrangement 

# COMMAND ----------

display(df_payment_arrange
        .select('ARR_DESCR')
        .distinct()
        )

# COMMAND ----------

display(
    df_wo_ifp_final
    # .filter(f.col('payment_cat') == 'one_plus')
    .join(
        df_payment_arrange
        .filter(f.col('ARR_DESCR')!='Collection Hold')
        .groupBy('fs_acct_id')
        .agg(
            f.min('TRANSACTION_DATE').alias('min_trans_date')
            , f.max('TRANSACTION_DATE').alias('max_trans_date')
            , f.min('INSTALLMENT_DUE_DATE').alias('min_installment_date')
            , f.max('INSTALLMENT_DUE_DATE').alias('max_installment_date')
            , f.min('ARR_STATUS').alias('min_status')
            , f.max('ARR_STATUS').alias('max_status')
        )
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'wo_month_6p'
        , f.add_months('rec_created_dttm', -6)
    )
    .withColumn(
        'has payment arrangement 6 month before write-off'
        , f.when(
            (f.col('min_trans_date') >= f.col('wo_month_6p')) & 
          (f.col('min_trans_date') <= f.col('rec_created_dttm') )
            , f.lit('Y')
                ).otherwise(f.lit('N'))
                )
    .groupBy('wo_month', 'has payment arrangement 6 month before write-off', 'payment_cat')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*')
        , f.mean('ifp_tenure(month)')
    )
    .filter(f.col('wo_month') != '2023-01')
)

# COMMAND ----------

df_wo_arrange = (
    df_wo_ifp_final
    # .filter(f.col('payment_cat') == 'one_plus')
    .join(
        df_payment_arrange
        .filter(f.col('ARR_DESCR')!='Collection Hold')
        .groupBy('fs_acct_id')
        .agg(
            f.min('TRANSACTION_DATE').alias('min_trans_date')
            , f.max('TRANSACTION_DATE').alias('max_trans_date')
            , f.min('INSTALLMENT_DUE_DATE').alias('min_installment_date')
            , f.max('INSTALLMENT_DUE_DATE').alias('max_installment_date')
            , f.min('ARR_STATUS').alias('min_status')
            , f.max('ARR_STATUS').alias('max_status')
        )
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'wo_month_6p'
        , f.add_months('rec_created_dttm', -6)
    )
    .withColumn(
        'has payment arrangement 6 month before write-off'
        , f.when(
            (f.col('min_trans_date') >= f.col('wo_month_6p')) & 
          (f.col('min_trans_date') <= f.col('rec_created_dttm') )
            , f.lit('Y')
                ).otherwise(f.lit('N'))
                )
    # .groupBy('wo_month', 'has payment arrangement 6 month before write-off')
    # .agg(
    #     f.countDistinct('fs_acct_id')
    #     , f.count('*')
    #     , f.mean('ifp_tenure(month)')
    # )
    .filter(f.col('wo_month') != '2023-01')
)

# COMMAND ----------

display(df_wo_arrange
        .filter(f.col('payment_cat') == 'neverpay')
        .filter(f.col('has payment arrangement 6 month before write-off')== 'Y')
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sales Channel

# COMMAND ----------

ls_top5 = ['Vodafone Retail', 'Online', 'Retailer', 'Telesales', 'Franchisee']

display(
    df_wo_ifp_final
    .withColumn(
        'reporting_date'
        , f.last_day('writeoff_date')
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
                , 'payment_cat'
            )
        )
    )
    .withColumn(
        'top 5 sales channel group'
        , f.col('ifp_prm_dvc_sales_channel_group')
    )
    .groupBy('reporting_date', 'top 5 sales channel group')
    .pivot(
        'payment_cat'
    )
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    .filter(f.col('top 5 sales channel group').isin(ls_top5))
    .filter(f.col('reporting_date') > '2023-01-01')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### sales channel base

# COMMAND ----------

df_ifp_base = (
    df_fea_oa.select(
        "fs_acct_id",
        "ifp_prm_dvc_term_start_date",
        "ifp_prm_dvc_term_end_date",
        "ifp_prm_dvc_sales_channel_group",
        "ifp_prm_dvc_model",
        "ifp_prm_dvc_term",
        "fs_ifp_prm_dvc_id",
        "network_dvc_type",
        "ifp_prm_dvc_discount",
        "ifp_prm_dvc_discount_flag",
        "ifp_prm_dvc_pmt_monthly",
        "ifp_prm_dvc_pmt_net_monthly", 
        "ifp_acct_dvc_pmt_monthly_tot",
        "ifp_acct_accs_pmt_monthly_tot",
        "network_dvc_hardware_rating",
        'reporting_date'
    )
    .filter(f.col("reporting_date") >= "2023-01-01")
    .filter(f.col("ifp_prm_dvc_term_start_date").isNotNull())
    .distinct()
)

# COMMAND ----------

ls_top5 = ['Vodafone Retail', 'Online', 'Retailer', 'Telesales', 'Franchisee']

display(
    df_ifp_base
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
            )
        )
    )
    .withColumn(
        'top 5 sales channel group'
        , f.col('ifp_prm_dvc_sales_channel_group')
    )
    .groupBy('reporting_date', 'top 5 sales channel group')
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    .filter(f.col('top 5 sales channel group').isin(ls_top5))
    .filter(f.col('reporting_date') > '2023-01-01')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### sale channel per tenure 

# COMMAND ----------

ls_top5 = ['Vodafone Retail', 'Online', 'Retailer', 'Telesales', 'Franchisee']

display(
    df_wo_ifp_final
    .withColumn(
        'reporting_date'
        , f.last_day('writeoff_date')
    )
    .withColumn(
        'top 5 sales channel group'
        , f.col('ifp_prm_dvc_sales_channel_group')
    )
    .groupBy('reporting_date', 'top 5 sales channel group')
    .agg(
        f.mean('ifp_tenure(month)')
    )
    .filter(f.col('top 5 sales channel group').isin(ls_top5))
    .filter(f.col('reporting_date') > '2023-01-01')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Device Rating Tier

# COMMAND ----------

display(
    df_wo_ifp_final
    .filter(f.col('ifp_cat') != 'no_device')
    # .filter(f.col('network_dvc_hardware_rating').isNotNull())
    # .filter(f.col('network_dvc_hardware_rating') != 'Entry Level')
    .withColumn(
        'Device Rating'
        , f.col('network_dvc_hardware_rating')
    )
    .groupBy('wo_month', 'Device Rating')
    .agg(
        f.mean('ifp_tenure(month)')
        , f.sum('device_cnt')
        , f.mean('ifp_prm_dvc_term')
        , f.median('ifp_prm_dvc_term')
    )
)
