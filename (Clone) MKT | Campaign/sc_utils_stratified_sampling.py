# Databricks notebook source
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as f
from typing import Optional, List, Dict, Tuple

# COMMAND ----------

# DBTITLE 1,generate_sample
def generate_sample(
    df: DataFrame,
    size: Optional[int] = None,
    proportion: Optional[float] = None,
    strata: Optional[List[str]] = None,
    priority_field: Optional[str] = None,
    priority_groups: Optional[Dict[str, float]] = None,
    default_priority: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[DataFrame, DataFrame]:
    """
    Perform stratified or random sampling with optional priority boosting using PySpark.

    :param df: Input PySpark DataFrame.
    :param size: Number of samples (optional if proportion is provided).
    :param proportion: Proportion of samples from input DataFrame (optional if size is provided).
    :param strata: List of column names used for stratification (if None, random sampling is performed).
    :param priority_field: Column name used for prioritization.
    :param priority_groups: Dictionary of group names and boosting weights.
    :param default_priority: Default weight for records not specified in priority_group.
    :param seed: Seed for random functions to ensure reproducibility.
    :return: Tuple of (sampled DataFrame, remaining DataFrame).
    """
    
    if size is None and proportion is None:
        raise ValueError("Either 'size' or 'proportion' must be provided.")

    if size is None:
        total_count = df.count()
        size = int(total_count * proportion)
        
    if priority_field and priority_groups:
        df_priority = (
            df
            .select(f.col(priority_field))
            .distinct()
            .join(
                spark
                .createDataFrame(
                    priority_groups.items()
                    , [priority_field, "priority_weight"]
                )
                , on=priority_field
                , how="left"
            )
            .fillna({"priority_weight": default_priority})
        )
        
        df_priority = f.broadcast(df_priority)
        df_input = (
            df
            .join(df_priority, on=priority_field, how="left")
            .fillna({"priority_weight": default_priority})
        )
    else:
        df_input = df.withColumn("priority_weight", f.lit(default_priority))

    if strata is None:
        df_sampled = (
            df_input
            .withColumn(
                "weighted_random"
                , f.rand(seed) * f.col("priority_weight")
            )
            .orderBy(f.desc("weighted_random"))
            .limit(size)
        )
    else:
        df_stratum_sample_sizes = (
            df_input
            .groupBy(*strata)
            .agg(f.sum("priority_weight").alias("stratum_weight"))
            .withColumn(
                "total_weight"
                , f.sum("stratum_weight").over(Window.partitionBy(f.lit(1)))
            )
            .withColumn(
                "proportion"
                , f.col("stratum_weight") / f.col("total_weight")
            )
            .withColumn(
                "sample_size"
                , f.round(f.col("proportion") * size).cast("int")
            )
        )
        
        df_sampled = (
            df_input
            .withColumn(
                "weighted_random"
                , f.rand(seed) * f.col("priority_weight")
            )
            .withColumn(
                "rank"
                , f.row_number().over(
                    Window
                    .partitionBy(*strata)
                    .orderBy(f.desc("weighted_random"))
                )
            )
            .join(
                df_stratum_sample_sizes
                .select(*strata, "sample_size")
                , on=strata
                , how="left"
            )
            .filter(f.col("rank") <= f.col("sample_size"))
        )
        
    df_sampled = (
        df_sampled
        .select(df.columns)
    )
       
    df_remaining = (
        df
        .join(
            df_sampled
            , on=df.columns
            , how="left_anti"
        )
    )
    
    return df_sampled, df_remaining

# COMMAND ----------

def generate_sample_v2(
    df: DataFrame,
    size: Optional[int] = None,
    proportion: Optional[float] = None,
    strata: Optional[List[str]] = None,
    priority_field: Optional[str] = None,
    priority_groups: Optional[Dict[str, float]] = None,
    default_priority: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[DataFrame, DataFrame]:
    """
    Perform stratified or random sampling with optional priority boosting using PySpark.

    :param df: Input PySpark DataFrame.
    :param size: Number of samples (optional if proportion is provided).
    :param proportion: Proportion of samples from input DataFrame (optional if size is provided).
    :param strata: List of column names used for stratification (if None, random sampling is performed).
    :param priority_field: Column name used for prioritization.
    :param priority_groups: Dictionary of group names and boosting weights.
    :param default_priority: Default weight for records not specified in priority_group.
    :param seed: Seed for random functions to ensure reproducibility.
    :return: Tuple of (sampled DataFrame, remaining DataFrame).
    """
    
    if size is None and proportion is None:
        raise ValueError("Either 'size' or 'proportion' must be provided.")

    if size is None:
        total_count = df.count()
        size = int(total_count * proportion)
        
    if priority_field and priority_groups:
        df_priority = (
            df
            .select(f.col(priority_field))
            .distinct()
            .join(
                spark
                .createDataFrame(
                    priority_groups.items()
                    , [priority_field, "priority_weight"]
                )
                , on=priority_field
                , how="left"
            )
            .fillna({"priority_weight": default_priority})
        )
        
        df_priority = f.broadcast(df_priority)
        df_input = (
            df
            .join(df_priority, on=priority_field, how="left")
            .fillna({"priority_weight": default_priority})
        )
    else:
        df_input = df.withColumn("priority_weight", f.lit(default_priority))

    if strata is None:
        df_sampled = (
            df_input
            .withColumn(
                "weighted_random"
                , f.rand(seed) * f.col("priority_weight")
            )
            .orderBy(f.desc("weighted_random"))
            .limit(size)
        )
    else:
        df_stratum_sample_sizes = (
            df_input
            .groupBy(*strata)
            .agg(f.sum("priority_weight").alias("stratum_weight"))
            .withColumn(
                "total_weight"
                , f.sum("stratum_weight").over(Window.partitionBy(f.lit(1)))
            )
            .withColumn(
                "proportion"
                , f.col("stratum_weight") / f.col("total_weight")
            )
            .withColumn(
                "sample_size"
                , f.round(f.col("proportion") * size).cast("int")
            )
        )
        
        df_sampled = (
            df_input
            .withColumn(
                "weighted_random"
                , f.rand(seed) * f.col("priority_weight")
            )
            .withColumn(
                "rank"
                , f.row_number().over(
                    Window
                    .partitionBy(*strata)
                    .orderBy(f.desc("weighted_random"))
                )
            )
            .join(
                df_stratum_sample_sizes
                .select(*strata, "sample_size")
                , on=strata
                , how="left"
            )
            .filter(f.col("rank") <= f.col("sample_size"))
        )
        
   
    
    return df_sampled

# COMMAND ----------

def create_sample_target(df, strata):
    """
    Extract a sample from the dataframe based on multiple strata fields.
    
    :param df: Input DataFrame
    :param size: Total sample size required
    :param strata_fields: List of field names to use for stratification
    :return: DataFrame with sample targets for each combination of strata field values
    """
    df_sample_target = (
        df.groupBy(*strata)
        .agg(f.count("*").alias("cnt"))
        .withColumn(
            "cnt_tot"
            , f.sum("cnt").over(Window.partitionBy(f.lit(1)))
        )
        .withColumn(
            "pct"
            , f.col("cnt") / f.col("cnt_tot")
        )
    )
    return df_sample_target

# COMMAND ----------

def find_similar_sample(
    df
    , size
    , strata
    , df_target
    , priority_field: Optional[str] = None
    , priority_groups: Optional[Dict[str, float]] = None
    , default_priority: float = 1.0
    , seed: Optional[int] = None
):
    
    if priority_field and priority_groups:
        df_priority = (
            df
            .select(f.col(priority_field))
            .distinct()
            .join(
                spark
                .createDataFrame(
                    priority_groups.items()
                    , [priority_field, "priority_weight"]
                )
                , on=priority_field
                , how="left"
            )
            .fillna({"priority_weight": default_priority})
        )
        
        df_priority = f.broadcast(df_priority)
        df_input = (
            df
            .join(df_priority, on=priority_field, how="left")
            .fillna({"priority_weight": default_priority})
        )
    else:
        df_input = df.withColumn("priority_weight", f.lit(default_priority))

    df_stratum_sample_sizes = (
        df_target
        .withColumn(
            "sample_size"
            , f.ceil(f.col("pct") * size)
        )
    )
        
    df_sampled = (
        df_input
        .withColumn(
            "weighted_random"
            , f.rand(seed) * f.col("priority_weight")
        )
        .withColumn(
            "rank"
            , f.row_number().over(
                Window
                .partitionBy(*strata)
                .orderBy(f.desc("weighted_random"))
            )
        )
        .join(
            df_stratum_sample_sizes
            .select(*strata, "sample_size")
            , on=strata
            , how="left"
        )
        .filter(f.col("rank") <= f.col("sample_size"))
    )
        
    df_sampled = (
        df_sampled
        .select(df.columns)
    )

    return df_sampled

# COMMAND ----------

# DBTITLE 1,evaluate_sample
def evaluate_sample(df_sample, df_target, strata_fields):
    for field in strata_fields:
        is_categorical = df_sample.select(field).dtypes[0][1] in ['string', 'boolean']
        
        # Common aggregation function for both categorical and numerical fields
        def get_stats(df, suffix):
            if is_categorical:
                return (
                    df.groupBy(field)
                    .agg(f.count("*").alias("cnt"))
                    .withColumn("pct", f.round(f.col("cnt") / f.sum("cnt").over(Window.partitionBy(f.lit(1))) * 100, 2))
                    .select(
                        field
                        , f.col("cnt").alias(f"cnt_{suffix}")
                        , f.col("pct").alias(f"pct_{suffix}")
                    )
                )
            else:
                return (
                    df.agg(
                        f.mean(field).cast("double").alias("mean")
                        , f.median(field).cast("double").alias("median")
                        , f.percentile_approx(field, 0.25).cast("double").alias("q1")
                        , f.percentile_approx(field, 0.75).cast("double").alias("q3")
                    )
                    .select(f.expr("stack(4, 'mean', mean, 'median', median, 'q1', q1, 'q3', q3) as (metric, value)"))
                    .withColumn("value", f.round("value", 2))
                    .withColumnRenamed("value", suffix)
                )

        # Get stats for both sample and target groups
        df_sample_stats = get_stats(df_sample, "sample")
        df_target_stats = get_stats(df_target, "target")

        # Join and finalize the comparison dataframe
        if is_categorical:
            df_compare = (
                df_sample_stats
                .join(df_target_stats, field, "full")
                .withColumn("delta", f.round((f.col("pct_sample") / f.col("pct_target") - 1) * 100, 2))
                .select(field, "cnt_sample", "pct_sample", "cnt_target", "pct_target", "delta")
            )
        else:
            df_compare = (
                df_sample_stats
                .join(df_target_stats, "metric", "full")
                .withColumn("delta", f.round((f.col("sample") / f.col("target") - 1) * 100, 2))
                .select("metric", "sample", "target", "delta")
            )

        print(field)
        display(df_compare)
