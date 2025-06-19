# Databricks notebook source
def get_sample_extractor(
    df
    , size
):
    df_out = (
        df
        .groupBy("ifp_top_ntile")
        .agg(f.countDistinct("fs_srvc_id").alias("cnt"))
        .withColumn(
            "cnt_tot"
            , f.sum("cnt").over(
                Window
                .partitionBy(f.lit(1))
            )
        )
        .withColumn(
            "pct"
            , f.col("cnt")/f.col("cnt_tot")
        )
        .withColumn(
            "sample_req"
            , f.ceil(f.col("pct") * size)
        )
    )

    return df_out


# COMMAND ----------

def get_local_control(
    df
    , df_extractor
    , seed
):
    df_out = (
        df
        .withColumn("rand", f.rand(seed = seed))
        .withColumn(
            "index"
            , f.row_number().over(
                Window
                .partitionBy("ifp_top_ntile")
                .orderBy("rand")
            )
        )
        .join(df_extractor, ["ifp_top_ntile"], "inner")
        .withColumn(
            "sample_flag"
            , f.when(
                f.col('index') <= f.col("sample_target")
                , f.lit('Y')
            ).otherwise(f.lit("N"))
        )
        .filter(f.col("sample_flag") == 'Y')
    )

    return df_out
    
