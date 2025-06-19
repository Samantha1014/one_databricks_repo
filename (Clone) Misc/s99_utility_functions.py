# Databricks notebook source
def get_db_notebook_dir(notebook_path) -> str:
    dir_curr = notebook_path
    ls_dir_elements = dir_curr.split("/")
    ls_dir_elements_proc = ls_dir_elements[0:(len(ls_dir_elements) - 1)]
    dir_out = "/".join([str(x) for x in ls_dir_elements_proc])
    return dir_out

# COMMAND ----------

from typing import Union
def interpolation_linear(
    start_col: Union[str, pyspark.sql.Column],
    end_col: Union[str, pyspark.sql.Column],
    step_size: Union[int, str] = None,
) -> pyspark.sql.Column:
 
    if isinstance(start_col, str):
        start_col = f.col(start_col)

    if isinstance(end_col, str):
        end_col = f.col(end_col)

    if isinstance(step_size, str):
        step_size = f.expr(step_size)

    sequence = f.sequence(start=start_col, stop=end_col, step=step_size)

    return f.explode(sequence)


# COMMAND ----------

def lower_col_names(
    df: sql.DataFrame
) -> sql.DataFrame:

    df_out = (
        df
        .toDF(*[c.lower() for c in df.columns])
    )

    return df_out

# COMMAND ----------

def pull_col(
    df: sql.DataFrame
    , vt_col: str
) -> list:
    '''pull column of a spark df into a list
    Args:
        df: spark dataframe
        vt_col: column name
    '''

    ls_out = (
        df
        .select(vt_col)
        .rdd.map(lambda x: x[0])
        .collect()
    )

    return ls_out

# COMMAND ----------

def get_spkdf_schema(
    df: sql.DataFrame
) -> sql.DataFrame:
    
    import pandas as pd

    ls_df_field = []
    ls_df_field_type = []

    for i in df.dtypes:
        ls_df_field.append(i[0])
        ls_df_field_type.append(i[1])

    df_output = pd.DataFrame(
        data = {
            "field": ls_df_field
            , "type": ls_df_field_type
        }
    )

    df_output = spark.createDataFrame(df_output)

    return df_output

# COMMAND ----------

def check_missing_features(
    df: sql.DataFrame
    , ls_features: list
    , ls_group_keys: list
):
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

    exprs = {x: "sum" for x in ls_features}

    display(
        df
        .groupBy(ls_group_keys)
        .agg(exprs)
    )

# COMMAND ----------

def check_distinct_values(
    df: sql.DataFrame
    , ls_features: list
    , ls_group_keys: list
):
    exprs = [f.countDistinct(x) for x in ls_features]

    display(
        df
        .groupBy(ls_group_keys)
        .agg(*exprs)
    )

# COMMAND ----------

def spkdf_lump_factor(
    df
    , vt_target_col
    , ls_group_cols
    , vt_n_levels
    , vt_other_level
    #, vt_date_min str=None
) -> sql.dataframe:

    if ls_group_cols is None:
        df = (
            df
            .withColumn(
                "pseudo_group"
                , f.lit(1)
            )
        )

        ls_group_cols = ["pseudo_group"]

    df_input = (
        df
        .fillna(value = "unknown", subset = [vt_target_col])
    )

    #if vt_param_date_min is not None:
    #    df_fct_summary_input = (
    #        df_input
    #        .filter(f.col("reporting_date") >= vt_date_min)
    #    )
    #else:
    #    df_fct_summary_input = df_input

    df_fct_summary = (
        df_input
        .withColumn("lump_index", f.lit(1))
        .groupBy(ls_group_cols + [vt_target_col])
        .agg(f.sum("lump_index").alias("weight"))
        .withColumn(
            "weight"
            , f.when(
                f.col(vt_target_col) == "unknown"
                , 0
            ).otherwise(f.col("weight"))
        )
        .withColumn(
            "lump_rank"
            , f.row_number().over(
                Window
                .partitionBy(ls_group_cols)
                .orderBy(f.desc("weight"))
            )
        )
        .withColumn(
            "lump_col"
            , f.when(
                f.col("lump_rank") > vt_n_levels
                , f.lit(vt_other_level)
            )
            .otherwise(f.col(vt_target_col))
        )
        .select(*ls_group_cols, vt_target_col, "lump_col")
    ) 

    df_output = (
        df_input
        .join(df_fct_summary, ls_group_cols + [vt_target_col], "left")
        .withColumn(vt_target_col, f.col("lump_col"))
        .drop("lump_col")
    )

    return df_output


# COMMAND ----------

import pyspark
from pyspark import sql 
from pyspark.sql import Window
from pyspark.sql import functions as f

def spkdf_initial_split(
    df: sql.dataframe
    , vt_valid_pct: int
    , ls_strata: list
    , seed: int
) -> dict:
    
    df_model_sampling_stats = (
        df
        .withColumn("cnt", f.lit(1))
        .groupBy(ls_strata)
        .agg(f.sum("cnt").alias("cnt_total"))
        .withColumn("cnt_valid", f.round(vt_valid_pct * f.col("cnt_total")))
    )
    
    df_model_split = (
        df
        .withColumn("rand", f.rand(seed = seed))
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy(ls_strata)
                .orderBy("rand")
            )
        )
        .join(df_model_sampling_stats, ls_strata, "left")
        .withColumn(
            "split_label"
            , f.when(
                f.col('index') <= f.col("cnt_valid")
                , f.lit("valid")
            ).otherwise(f.lit("train"))
        )
        .drop("rand", "index", "cnt_total", "cnt_valid")
    )
    
    return df_model_split

# COMMAND ----------

def export_model_data(
    df: sql.DataFrame
    , vt_path: str
    , ls_partition_keys: list
):
    (
        df
        .write
        .format("delta")
        .partitionBy(ls_partition_keys)
        .mode("overwrite")
        .option("overwriteSchema", "True")
        .save(vt_path)
    )
    

# COMMAND ----------

def h2o_get_predict(
    h2o_df
    , h2o_model
    , vt_pred_col: str="Y"
) -> sql.dataframe:
    
    h2odf_base_predict = h2o_model.predict(h2o_df)
    h2odf_base_predict = h2o_df.cbind(h2odf_base_predict)
    h2odf_base_predict["p1"] = h2odf_base_predict[vt_pred_col]

    df_base_predict = h2o.as_list(h2odf_base_predict)
    df_base_predict = spark.createDataFrame(df_base_predict)
    
    return df_base_predict

# COMMAND ----------

def h2o_get_gainlift(
    df: sql.dataframe
    , vt_breakdown: int
) -> sql.dataframe:
    df_out = (
        df
        .withColumn("target_label", f.when(f.col("target_label") == 'Y', f.lit(1)).otherwise(f.lit(0)))
        .withColumn("rank", f.col("p1"))
        .withColumn("ntile", f.ntile(vt_breakdown).over(
            Window
            .partitionBy("group")
            .orderBy(f.desc("rank"))
        ))
        .groupBy("group", "ntile")
        .agg(
            f.countDistinct("id").alias("target")
            , f.sum("target_label").alias("response")
        )
        .withColumn("target_cs", f.sum('target').over(
            Window
            .partitionBy("group")
            .orderBy("ntile")

        ))
        .withColumn("target_total", f.sum('target').over(
            Window
            .partitionBy("group")

        ))
        .withColumn("target_cs_pct", f.col('target_cs')/f.col("target_total"))
        .withColumn("response_cs", f.sum('response').over(
            Window
            .partitionBy("group")
            .orderBy("ntile")

        ))
        .withColumn("response_total", f.sum('response').over(
            Window
            .partitionBy("group")

        ))
        .withColumn("response_cs_pct", f.col('response_cs')/f.col("response_total"))
        .withColumn("data_fraction_cs", f.round((100/vt_breakdown/100) * f.col("ntile"), 3))
        .select(
            "group"
            , "ntile"
            , "data_fraction_cs"
            , "target"
            , "target_cs"
            , "target_cs_pct"

            , "response"
            , "response_cs"
            , "response_cs_pct"

            , f.col("response_cs_pct").alias("gain")
            , f.col("target_cs_pct").alias("gain_base_line")

        )
        .withColumn(
            "lift"
            , f.round(f.col("gain")/f.col("data_fraction_cs"), 3)
        )
        .withColumn(
            "lift_baseline"
            , f.lit(1)
        )
        #.select("group", "ntile", )
        .orderBy(f.asc("group"), f.asc("ntile"))
    )
    
    return df_out
    

# COMMAND ----------

def h2o_get_calibration_summary(
    df: sql.dataframe
    , vt_breakdown: int
) -> sql.dataframe:
    df_out = (
        df
        .withColumn("target_label", f.when(f.col("target_label") == 'Y', f.lit(1)).otherwise(f.lit(0)))
        .withColumn("rank", f.col("p1"))
        .withColumn("ntile", f.ntile(vt_breakdown).over(
            Window
            .partitionBy("group")
            #.orderBy(f.desc("rank"))
            .orderBy("rank")
        ))
        .groupBy("group", "ntile")
        .agg(
            f.countDistinct("id").alias("target")
            , f.sum("target_label").alias("response")
            , f.mean("p1").alias("avg_prob")
        )

        .withColumn(
            "positive_fraction"
            , f.col("response")/f.col('target')
        )
        
        .withColumn("data_fraction_cs", f.round((100/vt_breakdown/100) * f.col("ntile"), 3))
        
        .select(
            "group"
            , "ntile"
            , "data_fraction_cs"
            , "target"
            , "response"
            , "positive_fraction"
            , "avg_prob"
        )
        .orderBy(f.asc("group"), f.asc("ntile"))
    )
    
    return df_out
    
