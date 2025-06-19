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

# DBTITLE 1,late pay days trend
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

display(
    df_fea_ifp_accessory
    .filter(f.col('ifp_acct_accs_flag') == 'Y')
    .select('reporting_date', 'fs_acct_id', 'ifp_acct_accs_pmt_monthly_tot', 'ifp_acct_accs_value_med')
    .distinct()
    .agg(
        f.mean('ifp_acct_accs_pmt_monthly_tot')
        , f.mean('ifp_acct_accs_value_med')
    )
    # .filter(f.col('ifp_acct_accs_flag') == 'Y')
    # .limit(100)
)

# COMMAND ----------

display(
    df_wo_base
    # .filter(f.col('ifp_acct_accs_flag') == 'Y')
    .limit(100)
)

# COMMAND ----------

df_fea_oa.columns

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

df_ifp_base.columns

# COMMAND ----------

# DBTITLE 1,ifp base monthly payment base
display(
    df_ifp_base
    .filter(f.col('fs_ifp_prm_dvc_id').isNotNull())
    .withColumn(
        'reporting_date'
        , f.last_day('ifp_prm_dvc_term_start_date')
    )
    .select('reporting_date', 'fs_acct_id', 'ifp_acct_dvc_pmt_monthly_tot')
    .distinct()
    .withColumn(
        'monthly payment group'
        , f.when(
            f.col('ifp_acct_dvc_pmt_monthly_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 100) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 80) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 60) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 40) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 20) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'monthly payment group'
    )
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    .filter(f.col('reporting_date') > '2023-01-01')
)

# COMMAND ----------

display(
    df_ifp_base
    .withColumn(
        'reporting_date'
        , f.last_day('ifp_prm_dvc_term_start_date')
    )
    .select('reporting_date', 'fs_acct_id', 'ifp_acct_dvc_pmt_monthly_tot', 'fs_ifp_prm_dvc_id')
    .distinct()
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('ifp_acct_dvc_pmt_monthly_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 100) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 80) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 60) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 40) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('ifp_acct_dvc_pmt_monthly_tot') <= 20) & (f.col('ifp_acct_dvc_pmt_monthly_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        "device_cnt",
        f.count("fs_ifp_prm_dvc_id").over(Window.partitionBy("fs_acct_id")),
    )
    .withColumn(
        "ifp_cat",
        f.when(f.col("device_cnt").isNull(), f.lit("no_device"))
        .when(f.col("device_cnt") == 1, f.lit("one_device"))
        .when(f.col("device_cnt") > 1, f.lit("multi_device"))
        .otherwise("unknown"),
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
                , 'monthly_payment_cat'
            )
        )
    )
    .filter(f.col('reporting_date') > '2023-01-01')
    .groupBy(
        'reporting_date'
        , 'ifp_cat'
    )
    .pivot('monthly_payment_cat')
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
)

# COMMAND ----------

display(
    df_ifp_base
    .select('fs_acct_id', 'ifp_acct_dvc_pmt_monthly_tot')
    # .groupBy('fs_acct_id')
    .agg(
        f.percentile_approx("ifp_acct_dvc_pmt_monthly_tot", 0.25),
        f.median("ifp_acct_dvc_pmt_monthly_tot"),
        f.percentile_approx("ifp_acct_dvc_pmt_monthly_tot", 0.75),
    )
)
    

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

display(df_one_payment.limit(10))

# COMMAND ----------

# DBTITLE 1,IFP and WO
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

display(df_wo_ifp.limit(10))

# COMMAND ----------

display(
    df_wo_ifp
    .withColumn(
         "wo_month"
         , f.date_format("writeoff_date", "yyyy-MM")
     )
    .groupBy("wo_month")
    .agg(
         f.countDistinct("fs_acct_id")
         , f.countDistinct("fs_ifp_prm_dvc_id")
     )
)

# COMMAND ----------

# DBTITLE 1,left join to write off base
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
    .filter(f.col("rec_created_dttm") >= "2023-01-01")
    # .groupBy('wo_month', 'ifp_cat')
    # .agg(f.countDistinct('fs_acct_id')
    #      , f.countDistinct('fs_ifp_prm_dvc_id')
    #     )
)

# COMMAND ----------

# DBTITLE 1,wo_ifp_final_summary
display(
    df_wo_ifp_final
    .agg(
        f.min("ifp_prm_dvc_discount"),
        f.max("ifp_prm_dvc_discount"),
        f.percentile_approx("ifp_prm_dvc_discount", 0.25),
        f.median("ifp_prm_dvc_discount"),
        f.percentile_approx("ifp_prm_dvc_discount", 0.75),
        # f.min('wo_month')
        # , f.max('wo_month') 
    )
)

# COMMAND ----------

# DBTITLE 1,network device
display(
    df_wo_ifp_final
    .filter(f.col("device_cnt") == 1)
    .groupBy("wo_month", "network_dvc_hardware_rating")
    .agg(
         f.countDistinct("fs_acct_id")
         , f.countDistinct("fs_ifp_prm_dvc_id")
     )
)

display(
    df_wo_ifp_final
    .filter(f.col("device_cnt") > 1)
    .groupBy("wo_month", "network_dvc_hardware_rating")
    .agg(
         f.countDistinct("fs_acct_id")
         , f.countDistinct("fs_ifp_prm_dvc_id")
     )
)

# COMMAND ----------

# DBTITLE 1,WO ifp with disocunt
display(
    df_wo_ifp_final
    .filter(f.col("device_cnt") == 1)
    .groupBy("wo_month", "ifp_prm_dvc_discount_flag")
    .agg(
         f.countDistinct("fs_acct_id")
         , f.countDistinct("fs_ifp_prm_dvc_id")
     )
)

display(
    df_wo_ifp_final
    .filter(f.col("device_cnt") > 1)
    .groupBy("wo_month", "ifp_prm_dvc_discount_flag")
    .agg(
         f.countDistinct("fs_acct_id")
         , f.countDistinct("fs_ifp_prm_dvc_id")
     )
)

# COMMAND ----------

# DBTITLE 1,device breakdown
display(
    df_wo_ifp_final
    .withColumn(
        "wo_sum"
        , f.count("fs_acct_id").over(Window.partitionBy())
    )
    .filter(f.col("device_cnt") >= 1)
    .filter(f.col("ifp_prm_dvc_discount_flag") == f.lit("Y"))
    .groupBy("wo_month", "ifp_cat")
    .agg(
         f.countDistinct("fs_acct_id")
         , f.countDistinct("fs_ifp_prm_dvc_id")
     )
)

# COMMAND ----------

# DBTITLE 1,WO ifp device
display(
    df_wo_ifp_final
    .groupBy("wo_month", "ifp_cat")
    .agg(
        f.countDistinct("fs_acct_id")
        , f.countDistinct("fs_ifp_prm_dvc_id")
    )
)

# COMMAND ----------

display(
    df_wo_ifp_final
    .filter(f.col("ifp_cat").isin("one_device", "multi_device"))
    .groupBy(
        "wo_month", "ifp_cat"
    )
    # .pivot('ifp_cat')
    .agg(
        f.avg("ifp_tenure(month)"),
        f.percentile_approx("ifp_tenure(month)", 0.25).alias("25th"),
        f.median("ifp_tenure(month)"),
        f.percentile_approx("ifp_tenure(month)", 0.75).alias("75th"),
        f.percentile_approx("ifp_tenure(month)", 0.9).alias("90th"),
    )
)

# COMMAND ----------

display(
    df_wo_base
    .withColumn(
        "wo_month"
        , date_format("rec_created_dttm", "yyyy-MM")
    )
    .groupBy("wo_month")
    .agg(f.countDistinct("fs_acct_id"))
    .filter(f.col("wo_month") >= "2023-01")
)

# COMMAND ----------

# DBTITLE 1,Full Base for WO
display(
    df_wo_base
    .withColumn(
        "wo_month"
        , date_format("rec_created_dttm", "yyyy-MM")
    )
    .join(df_one_payment, ["fs_acct_id"], "left")
    .withColumn(
        "payment_cat",
        f.when(f.col("payment_cnt").isNull(), f.lit("neverpay"))
        .when(f.col("payment_cnt") == 1, f.lit("one_pay"))
        .when(f.col("payment_cnt") > 1, f.lit("one_plus"))
        .otherwise(f.lit("unknown")),
    )
    .filter(f.col("rec_created_dttm") >= "2023-01-01")
    .groupBy("payment_cat", "wo_month")
    .agg(
        f.countDistinct("fs_acct_id")
        , f.count("*")
    )
)

# COMMAND ----------

# DBTITLE 1,Mobile OA Consumer WO
display(
    df_wo_base
    .withColumn(
        "wo_month"
        , date_format("rec_created_dttm", "yyyy-MM")
    )
    .join(
        df_fea_oa
        .select("fs_acct_id")
        .distinct()
        , ["fs_acct_id"]
        , "inner"
    )
    .join(df_one_payment, ["fs_acct_id"], "left")
    .withColumn(
        "payment_cat",
        f.when(f.col("payment_cnt").isNull(), f.lit("neverpay"))
        .when(f.col("payment_cnt") == 1, f.lit("one_pay"))
        .when(f.col("payment_cnt") > 1, f.lit("one_plus"))
        .otherwise(f.lit("unknown")),
    )
    .filter(f.col("rec_created_dttm") >= "2023-01-01")
    .groupBy("payment_cat", "wo_month")
    .agg(
        f.countDistinct("fs_acct_id")
        , f.count("*")
    )
)

# COMMAND ----------

display(
    df_wo_base
    .withColumn(
        "wo_month"
        , date_format("rec_created_dttm", "yyyy-MM")
    )
    .join(df_one_payment, ["fs_acct_id"], "left")
    .withColumn(
        "payment_cat",
        f.when(f.col("payment_cnt").isNull(), f.lit("neverpay"))
        .when(f.col("payment_cnt") == 1, f.lit("one_pay"))
        .when(f.col("payment_cnt") > 1, f.lit("one_plus"))
        .otherwise(f.lit("unknown")),
    )
    .filter(f.col("rec_created_dttm") >= "2023-01-01")
    .filter(f.col("payment_cat") == "one_pay")
)

# COMMAND ----------

display(
    df_wo_base
    .withColumn(
        "wo_month"
        , date_format("rec_created_dttm", "yyyy-MM")
    )
    .join(df_one_payment, ["fs_acct_id"], "left")
    .withColumn(
        "payment_cat",
        f.when(f.col("payment_cnt").isNull(), f.lit("neverpay"))
        .when(f.col("payment_cnt") == 1, f.lit("one_pay"))
        .when(f.col("payment_cnt") > 1, f.lit("one_plus"))
        .otherwise(f.lit("unknown")),
    )
    .join(df_wo_ifp_final, ["fs_acct_id"], "left")
)

# COMMAND ----------

# DBTITLE 1,accessories monthly payment
display(
    df_wo_ifp_final
    .filter(f.col('ifp_cat') != 'no_device')
    .select('wo_month', 'fs_acct_id', 'ifp_acct_dvc_pmt_monthly_tot', 'ifp_acct_accs_pmt_monthly_tot', 'ifp_cat')
    .distinct()
    .join(
        df_fea_ifp_accessory
        .filter(f.col('ifp_acct_accs_flag') == 'Y')
        .select('ifp_acct_accs_term_end_date_min', 'fs_acct_id', 'ifp_acct_accs_flag', 'ifp_acct_accs_cnt', 'ifp_acct_accs_value_tot')
        .distinct()
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'have accessory flag'
        , f.when(
            # (f.col('ifp_prm_dvc_term_end_date') >= f.col('ifp_acct_accs_term_end_date_min')) &
            # (f.col('ifp_prm_dvc_term_start_date') <= f.col('ifp_acct_accs_term_end_date_min')) &
            (f.col('ifp_acct_accs_flag') == 'Y')
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .fillna(0, subset=['ifp_acct_dvc_pmt_monthly_tot', 'ifp_acct_accs_pmt_monthly_tot'])
    .withColumn(
        'monthly_payment_tot'
        , f.col('ifp_acct_dvc_pmt_monthly_tot') + f.col('ifp_acct_accs_pmt_monthly_tot')
    )
    .groupBy('wo_month', 'have accessory flag')
    .pivot('ifp_cat')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*')
        , f.mean('monthly_payment_tot')
        , f.median('monthly_payment_tot')
        , f.sum('monthly_payment_tot')/f.countDistinct('fs_acct_id')
    )
    .filter(f.col('wo_month') != '2023-01')
)

# COMMAND ----------

# DBTITLE 1,accessory on multi device
display(
    df_wo_ifp_final
    .filter(f.col('ifp_cat') != 'no_device')
    .join(
        df_fea_ifp_accessory
        .filter(f.col('ifp_acct_accs_flag') == 'Y')
        .select('ifp_acct_accs_term_end_date_min', 'fs_acct_id', 'ifp_acct_accs_flag', 'ifp_acct_accs_cnt', 'ifp_acct_accs_value_tot')
        .distinct()
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'have accessory flag'
        , f.when(
            # (f.col('ifp_prm_dvc_term_end_date') >= f.col('ifp_acct_accs_term_end_date_min')) &
            (f.col('ifp_prm_dvc_term_start_date') <= f.col('ifp_acct_accs_term_end_date_min')) &
            (f.col('ifp_acct_accs_flag') == 'Y')
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .groupBy('wo_month', 'have accessory flag')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*')
        , f.mean('ifp_tenure(month)')
        , f.median('ifp_tenure(month)')
    )
    .filter(f.col('wo_month') != '2023-01')
)

# COMMAND ----------

display(
    df_wo_ifp_final
    .filter(f.col('ifp_cat')!='no_device')
    .groupBy('wo_month')
    .pivot('ifp_cat')
    .agg(
        f.mean('ifp_tenure(month)')
        , f.median('ifp_tenure(month)')
    )
)

# COMMAND ----------

display(
    df_wo_ifp_final
    .filter(f.col('ifp_cat') != 'no_device')
    .join(
        df_fea_ifp_accessory
        .filter(f.col('ifp_acct_accs_flag') == 'Y')
        .select('ifp_acct_accs_term_end_date_min', 'fs_acct_id', 'ifp_acct_accs_flag', 'ifp_acct_accs_cnt', 'ifp_acct_accs_value_tot')
        .distinct()
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'have accessory flag'
        , f.when(
            # (f.col('ifp_prm_dvc_term_end_date') >= f.col('ifp_acct_accs_term_end_date_min')) &
            (f.col('ifp_prm_dvc_term_start_date') <= f.col('ifp_acct_accs_term_end_date_min')) &
            (f.col('ifp_acct_accs_flag') == 'Y')
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .groupBy('wo_month', 'have accessory flag')
    .pivot('ifp_cat')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*')
        , f.mean('ifp_tenure(month)')
    )
    .filter(f.col('wo_month') != '2023-01')
)

# COMMAND ----------

# DBTITLE 1,ifp wo monthly payment - mean ifp tenure
display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('wo_month').alias('wo_month')
        , f.mean('ifp_tenure(month)').alias('ifp_tenure(month)')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('wo_month')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
    )
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'monthly_payment_cat'
    )
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
        , f.mean('ifp_tenure(month)')
    )
    # .limit(10)
)

# COMMAND ----------

display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('wo_month').alias('wo_month')
        , f.mean('ifp_acct_dvc_pmt_monthly_tot').alias('ifp_acct_dvc_pmt_monthly_tot')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('wo_month')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('ifp_acct_dvc_pmt_monthly_tot')
        # , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
    )
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'monthly_payment_cat'
    )
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    # .limit(10)
)

# COMMAND ----------

# DBTITLE 1,ifp wo monthly payment
display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('wo_month').alias('wo_month')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('wo_month')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
    )
    .withColumn(
        'monthly payment group'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'monthly payment group'
    )
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    # .limit(10)
)

# COMMAND ----------

# DBTITLE 1,monthly payment band vs. multi/single device
display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('wo_month').alias('wo_month')
        , f.first('ifp_cat').alias('ifp_cat')
        , f.mean('ifp_acct_dvc_pmt_monthly_tot').alias('ifp_acct_dvc_pmt_monthly_tot')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('wo_month')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('ifp_acct_dvc_pmt_monthly_tot')
        # , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
        # + f.abs(f.col('ifp_acct_accs_pmt_monthly_tot'))
    )
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
                , 'ifp_cat'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'monthly_payment_cat'
    )
    .pivot('ifp_cat')
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    # .limit(10)
)

# COMMAND ----------



# COMMAND ----------

display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('wo_month').alias('wo_month')
        , f.first('ifp_cat').alias('ifp_cat')
        , f.mean('ifp_acct_dvc_pmt_monthly_tot').alias('ifp_acct_dvc_pmt_monthly_tot')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('wo_month')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('ifp_acct_dvc_pmt_monthly_tot')
        # , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
        # + f.abs(f.col('ifp_acct_accs_pmt_monthly_tot'))
    )
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
                , 'monthly_payment_cat'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'ifp_cat'
    )
    .pivot('monthly_payment_cat')
    .agg(
        f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    # .limit(10)
)

# COMMAND ----------

# DBTITLE 1,monthly payment band with/without accessories
display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('writeoff_date').alias('writeoff_date')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('writeoff_date')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .join(
        df_fea_ifp_accessory
        .filter(f.col('ifp_acct_accs_flag') == 'Y')
        .select('ifp_acct_accs_term_end_date_max', 'fs_acct_id', 'ifp_acct_accs_cnt', 'ifp_acct_accs_value_tot', 'ifp_acct_accs_pmt_monthly_tot')
        .distinct()
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'ifp_acct_accs_flag'
        , f.when(
            (f.col('writeoff_date') <= f.col('ifp_acct_accs_term_end_date_max')) &
            (f.col('ifp_acct_accs_term_end_date_max').isNotNull())
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .fillna(0, subset=['ifp_acct_accs_pmt_monthly_tot'])
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
        + f.abs(f.col('ifp_acct_accs_pmt_monthly_tot'))
    )
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
                , 'ifp_acct_accs_flag'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'monthly_payment_cat'
    )
    .pivot('ifp_acct_accs_flag')
    .agg(
        f.count('*')
        , f.countDistinct('fs_acct_id')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
    # .limit(10)
)

# COMMAND ----------

display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('writeoff_date').alias('writeoff_date')
        , f.first('ifp_cat').alias('ifp_cat')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('writeoff_date')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .join(
        df_fea_ifp_accessory
        .filter(f.col('ifp_acct_accs_flag') == 'Y')
        .select('ifp_acct_accs_term_end_date_max', 'fs_acct_id', 'ifp_acct_accs_cnt', 'ifp_acct_accs_value_tot', 'ifp_acct_accs_pmt_monthly_tot')
        .distinct()
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'ifp_acct_accs_flag'
        , f.when(
            (f.col('writeoff_date') <= f.col('ifp_acct_accs_term_end_date_max')) &
            (f.col('ifp_acct_accs_term_end_date_max').isNotNull())
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .fillna(0, subset=['ifp_acct_accs_pmt_monthly_tot'])
    # .filter(f.col('ifp_acct_accs_flag') == 'Y')
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
        + f.abs(f.col('ifp_acct_accs_pmt_monthly_tot'))
    )
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
                , 'monthly_payment_cat'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'ifp_acct_accs_flag'
    )
    .pivot('monthly_payment_cat')
    .agg(
        f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
)

# COMMAND ----------

display(
    df_wo_ifp_final
    .groupBy('fs_acct_id')
    .agg(
        f.max('base_wo_amt').alias('base_wo_amt')
        , f.max('write_off_item_amount').alias('write_off_item_amount')
        , f.max('ifp_period_left').alias('ifp_period_left')
        , f.min('ifp_prm_dvc_term_start_date').alias('ifp_prm_dvc_term_start_date')
        , f.min('writeoff_date').alias('writeoff_date')
        , f.first('ifp_cat').alias('ifp_cat')
    )
    .withColumn(
        'reporting_date'
        , f.last_day('writeoff_date')
    )
    .filter(f.col('ifp_period_left').isNotNull())
    .withColumn(
        'ifp_monthly_payment_impute'
        , f.col('write_off_item_amount')/f.col('ifp_period_left')
    )
    .join(
        df_fea_ifp_accessory
        .filter(f.col('ifp_acct_accs_flag') == 'Y')
        .select('ifp_acct_accs_term_end_date_max', 'fs_acct_id', 'ifp_acct_accs_cnt', 'ifp_acct_accs_value_tot', 'ifp_acct_accs_pmt_monthly_tot')
        .distinct()
        , ['fs_acct_id']
        , 'left'
    )
    .withColumn(
        'ifp_acct_accs_flag'
        , f.when(
            (f.col('writeoff_date') <= f.col('ifp_acct_accs_term_end_date_max')) &
            (f.col('ifp_acct_accs_term_end_date_max').isNotNull())
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .fillna(0, subset=['ifp_acct_accs_pmt_monthly_tot'])
    # .filter(f.col('ifp_acct_accs_flag') == 'Y')
    .withColumn(
        'monthly_payment_tot'
        , f.abs(f.col('ifp_monthly_payment_impute'))
        + f.abs(f.col('ifp_acct_accs_pmt_monthly_tot'))
    )
    .withColumn(
        'monthly_payment_cat'
        , f.when(
            f.col('monthly_payment_tot') > 100
            , f.lit('100+')
        )
        .when(
            (f.col('monthly_payment_tot') <= 100) & (f.col('monthly_payment_tot') > 80)
            , f.lit('80-100')
        )
        .when(
            (f.col('monthly_payment_tot') <= 80) & (f.col('monthly_payment_tot') > 60)
            , f.lit('60-80')
        )
        .when(
            (f.col('monthly_payment_tot') <= 60) & (f.col('monthly_payment_tot') > 40)
            , f.lit('40-60')
        )
        .when(
            (f.col('monthly_payment_tot') <= 40) & (f.col('monthly_payment_tot') > 20)
            , f.lit('20-40')
        )
        .when(
            (f.col('monthly_payment_tot') <= 20) & (f.col('monthly_payment_tot') > 0)
            , f.lit('0-20')
        )
        .otherwise(f.lit('MISC'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'reporting_date'
                , 'ifp_cat'
            )
        )
    )
    .groupBy(
        'reporting_date'
        , 'ifp_acct_accs_flag'
    )
    .pivot('ifp_cat')
    .agg(
        f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
)

# COMMAND ----------

# DBTITLE 1,sales channel analysis
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

# DBTITLE 1,sales channel base
ls_top5 = ['Vodafone Retail', 'Online', 'Retailer', 'Telesales', 'Franchisee']

display(
    df_ifp_base
    .withColumn(
        'reporting_date'
        , f.last_day('ifp_prm_dvc_term_start_date')
    )
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

# DBTITLE 1,device rating
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

# COMMAND ----------

df_payment_arrange.columns

# COMMAND ----------

# DBTITLE 1,payment arrangement
display(
    df_wo_ifp_final
    # .filter(f.col('payment_cat') == 'one_plus')
    .join(
        df_payment_arrange
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
            (f.col('max_installment_date') >= f.col('wo_month_6p')) 
            & (f.col('max_installment_date') <= f.col('rec_created_dttm'))
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .groupBy('wo_month', 'has payment arrangement 6 month before write-off')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*')
        , f.mean('ifp_tenure(month)')
    )
    .filter(f.col('wo_month') != '2023-01')
)

# COMMAND ----------

display(
    df_wo_ifp_final
    .join(
        df_payment_arrange
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
            (f.col('max_installment_date') >= f.col('wo_month_6p')) 
            & (f.col('max_installment_date') <= f.col('rec_created_dttm'))
            , f.lit('Y')
        ).otherwise(f.lit('N'))
    )
    .withColumn(
        'tot_per_month'
        , f.count('*').over(
            Window.partitionBy(
                'wo_month'
            )
        )
    )
    .groupBy('wo_month', 'has payment arrangement 6 month before write-off')
    .agg(
        f.countDistinct('fs_acct_id')
        , f.count('*')
        , f.mean('ifp_tenure(month)')
        , f.round(f.count('*')/f.first('tot_per_month')*100, 2).alias('percentage')
    )
)

# COMMAND ----------

display(
    df_wo_ifp_final
    # .filter(f.col('payment_cat') == 'one_plus')
    .join(
        df_payment_arrange
        .groupBy('fs_acct_id')
        .agg(
            f.min('transactiondate').alias('min_trans_date')
            , f.max('transactiondate').alias('max_trans_date')
            , f.min('installmentduedate').alias('min_installment_date')
            , f.max('installmentduedate').alias('max_installment_date')
            , f.min('status').alias('min_status')
            , f.max('status').alias('max_status')
        )
        , ['fs_acct_id']
        , 'left'
    )
)
