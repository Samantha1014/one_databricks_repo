# Databricks notebook source
def check_missing_features(
    df: sql.DataFrame
    , cnt_field: str
    , ls_features: list
    , ls_group_keys: list
    , metric: str='cnt'
):
    
    df_agg_tot = (
        df
        .groupBy(ls_group_keys)
        .agg(
            f.count(cnt_field).alias("cnt")
        )
    )

    for i in ls_features:
        df = (
            df
            .withColumn(
                i
                , f.when(
                    f.col(i).isNull()
                    , f.lit(1)
                ).otherwise(f.lit(0))
            )
        )

    exprs_agg = [f.sum(x).alias(x) for x in ls_features]
    
    df_agg_group = (
        df
        .groupBy(ls_group_keys)
        .agg(*exprs_agg)
    )

    df_output = (
        df_agg_tot
        .join(df_agg_group, ls_group_keys, "left")
    )

    if metric == 'pct':
        for i in ls_features:

            df_output = (
                df_output
                .withColumn(
                    i
                    , f.round(f.col(i)/f.col("cnt") * 100, 2)
                )
            )
    
    display(df_output)

# COMMAND ----------

def check_distinct_values(
    df: sql.DataFrame
    , ls_features: list
    , ls_group_keys: list
):
    exprs = [f.countDistinct(x).alias(x) for x in ls_features]

    display(
        df
        .groupBy(ls_group_keys)
        .agg(*exprs)
    )

# COMMAND ----------

def trend_profile(
    df: sql.DataFrame
    , group_field: str
    , cnt_field: str
    , metric: str
    , form: str="wide"
) -> sql.DataFrame:
    
    df_output = (
        df
        .groupBy("reporting_date", group_field)
        .agg(
            f.countDistinct(cnt_field).alias("cnt")
        )
        .withColumn(
            "cnt_tot"
            , f.sum("cnt").over(
                Window
                .partitionBy("reporting_date")
            )
        )
        .withColumn(
            "pct"
            , f.round(f.col("cnt")/f.col("cnt_tot") * 100, 2)
        )
        .orderBy(f.desc("reporting_date"))
        
    )

    df_mapping = (
        df_output
        .groupBy(group_field)
        .agg(f.sum(metric).alias(metric))
        .orderBy(f.desc(metric))
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy(f.lit(1))
                .orderBy(f.desc(metric))
            )
        )
        .withColumn(
            "index"
            , f.lpad(f.col("index"), 2, "0")
        )
        .withColumn(
            "index"
            , f.concat_ws("", f.lit("f"), f.col("index"))
        )
        .withColumn(
            "group_field_label"
            , f.concat_ws("_", f.col("index"), f.col(group_field))
        )
        .select(group_field, "group_field_label")
    )

    df_output = (
        df_output
        .join(
            df_mapping
            , [group_field]
            , "left"
        )
    )

    if form == 'wide':
        df_output = (
            df_output
            .groupBy("reporting_date")
            .pivot("group_field_label")
            .agg(f.sum(metric).alias(metric))
            .orderBy(f.desc("reporting_date"))
        )


    return df_output
    

# COMMAND ----------

def trend_profile_num(
    df: sql.DataFrame
    , group_fields: list
    , calc_fields: list
    , metrics: list
) -> sql.DataFrame:
    
    exprs_agg = list()

    if "med" in metrics:
        exprs_agg.extend([f.median(x).alias(x+"_med") for x in calc_fields])

    if "avg" in metrics:
         exprs_agg.extend([f.avg(x).alias(x+"_avg") for x in calc_fields])

    if "sum" in metrics:
         exprs_agg.extend([f.sum(x).alias(x+"_sum") for x in calc_fields])
    
    
    if group_fields is None:
        df = (
            df
            .groupBy("reporting_date", "reporting_cycle_type")
        )
    else:
        df = (
            df
            .groupBy("reporting_date", "reporting_cycle_type", *group_fields)
        )


    df_output = (
        df
        .agg(*exprs_agg)
        .orderBy(f.desc("reporting_date"))
    )

    return df_output
