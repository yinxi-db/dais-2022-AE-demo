# Databricks notebook source
# MAGIC %md ### Instantiate DLT Input Parameters

# COMMAND ----------

server = spark.conf.get("eventhub.server")
shared_access_key_name = spark.conf.get("eventhub.shared_access_key_name")
shared_access_key = spark.conf.get("eventhub.shared_access_key")
entity_path = spark.conf.get("eventhub.entity_path")
consumer_group = spark.conf.get("eventhub.consumer_group")

# COMMAND ----------

# MAGIC %md ### Generate Kafka Details

# COMMAND ----------

topic = entity_path
bootstrap_server = server
eh_sql = f"kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required username=\"$ConnectionString\"password=\"Endpoint=sb://{server}/;SharedAccessKeyName={shared_access_key_name};SharedAccessKey={shared_access_key};EntityPath={entity_path}\";"

# COMMAND ----------

# MAGIC %md ### Ingestion From Kafka (Bronze)

# COMMAND ----------

# Import dependencies

import dlt
from pyspark.sql.functions import col, from_json, to_date, window, avg as spark_avg

# COMMAND ----------

@dlt.table(
  name="streaming_sensor_raw_table",
  comment="Raw Wind Turbine data from IoT Hub",
  table_properties={
    "pipeline.quality": "raw",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "pipelines.autoOptimize.zOrderCols": "timestamp"
  }
)
def streaming_sensor_raw_table():
  input_stream = (spark.readStream.format("kafka")
          .option("subscribe", topic)
          .option("kafka.bootstrap.servers", bootstrap_server)
          .option("kafka.sasl.mechanism", "PLAIN")
          .option("kafka.security.protocol", "SASL_SSL")
          .option("kafka.sasl.jaas.config",eh_sql)
          .option("kafka.group.id",consumer_group)
          .option("startingOffsets", "earliest")
          .option("maxOffsetsPerTrigger", "500")
          .option("failOnDataLoss","false")
          .load()
       )

  return (input_stream)

# COMMAND ----------

@dlt.table(
  name="streaming_sensor_bronze_table",
  comment="Normalized Wind Turbine from IoT Hub",
  table_properties={
    "pipeline.quality": "bronze",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "pipelines.autoOptimize.zOrderCols": "timestamp, date"
  }
)
def streaming_sensor_normalized_bronze_table():
  
  # Schema
  schema = "timestamp timestamp, deviceId string, temperature double, humidity double, windspeed double, winddirection string, rpm double, angle double"
  
  bronze_df = dlt.read_stream("streaming_sensor_raw_table")
  output_df = (
    bronze_df.withColumn('reading', from_json(col('value').cast('string'), schema))
      .select('reading.*', to_date('reading.timestamp').alias('date'))  
              )
  return (output_df)

# COMMAND ----------

# MAGIC %md ### Split & Normalize (Silver)

# COMMAND ----------

@dlt.table(
  comment="Weather telemetry only",
  table_properties={
    "pipeline.quality": "silver",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "pipelines.autoOptimize.zOrderCols": "timestamp, date, deviceid"
  }
)
def streaming_weather_silver_table():
  bronze_df = dlt.read_stream("streaming_sensor_bronze_table")
  output_df = (
  bronze_df.filter("temperature is not null") # Filter out weather telemetry only
    .select('date','deviceid','timestamp','temperature','humidity','windspeed','winddirection') 
              )
  return (output_df)

# COMMAND ----------

@dlt.table(
  comment="Turbine telemetry only",
  table_properties={
    "pipeline.quality": "silver",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "pipelines.autoOptimize.zOrderCols": "timestamp, date, deviceid"
  }
)
def streaming_turbine_silver_table():
  bronze_df = dlt.read_stream("streaming_sensor_bronze_table")
  output_df = (
  bronze_df.filter("temperature is null")
    .select('date','timestamp','deviceId','rpm','angle')  
  )
  return (output_df)

# COMMAND ----------

# MAGIC %md ### Aggregation (Silver)

# COMMAND ----------

# MAGIC %md The next step of our processing pipeline will clean and aggregate the measurements to 1 hour intervals.
# MAGIC 
# MAGIC We will use the following schema for Silver and Gold data sets:
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iot_delta_bronze_to_gold.png" width=800>

# COMMAND ----------

# MAGIC %md Turbine Telemetry Aggregation

# COMMAND ----------

@dlt.view(
  comment="Turbine 5 minute aggregate view",
)
def agg_turbine_silver_view():
  input_df = dlt.read_stream("streaming_turbine_silver_table")

  return (input_df
          .withWatermark("timestamp", "10 minutes")
          .groupBy('deviceId','date', window('timestamp','5 minutes'))
          .agg(spark_avg('rpm').alias('rpm'), spark_avg("angle").alias("angle"))
         )

# COMMAND ----------

dlt.create_target_table(
  name = "streaming_agg_turbine_silver_table",
  comment = "Turbine telemetry aggregated",
  table_properties={
    "pipeline.quality": "silver",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "pipelines.autoOptimize.zOrderCols": "date, deviceid"
  }
)

dlt.apply_changes(
  target = "streaming_agg_turbine_silver_table",
  source = "agg_turbine_silver_view",
  keys = ["date","window","deviceId"],
  sequence_by = col("date"),
  ignore_null_updates = False,
  apply_as_deletes = None,
  column_list = None,
  except_column_list = None
)

# COMMAND ----------

# MAGIC %md Weather Telemetry Aggregation

# COMMAND ----------

@dlt.view(
  comment="Weather 5 minute aggregate view",
)
def agg_weather_silver_view():
  input_df = dlt.read_stream("streaming_weather_silver_table")

  return (input_df
          .withWatermark("timestamp", "10 minutes")
          .groupBy('deviceid','date',window('timestamp','5 minutes'))            
          .agg({"temperature":"avg","humidity":"avg","windspeed":"avg","winddirection":"last"})
          .selectExpr('date','window','deviceid','`avg(temperature)` as temperature','`avg(humidity)` as humidity',
                '`avg(windspeed)` as windspeed','`last(winddirection)` as winddirection')
         )

# COMMAND ----------

dlt.create_target_table(
  name = "streaming_agg_weather_silver_table",
  comment = "Weather telemetry aggregated",
  table_properties={
    "pipeline.quality": "silver",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "pipelines.autoOptimize.zOrderCols": "date, deviceid"
  }
)

dlt.apply_changes(
  target = "streaming_agg_weather_silver_table",
  source = "agg_weather_silver_view",
  keys = ["date","window","deviceId"],
  sequence_by = col("date"),
  ignore_null_updates = False,
  apply_as_deletes = None,
  column_list = None,
  except_column_list = None
)

# COMMAND ----------

# MAGIC %md ### Enrichment (Gold)

# COMMAND ----------

@dlt.view(
  comment="Enriched Turbine view",
)
def agg_turbine_gold_view():
  turbine_agg = dlt.read_stream("streaming_agg_turbine_silver_table")
  weather_agg = dlt.read_stream("streaming_agg_weather_silver_table").drop("deviceId")
  turbine_enriched = (
    turbine_agg.join(weather_agg, ['date','window'])
    .selectExpr('date','deviceId','window.start as   window','rpm','angle','temperature','humidity','windspeed','winddirection')
  )
  return turbine_enriched

# COMMAND ----------

dlt.create_target_table(
  name = "turbine_enriched",
  comment = "Enriched Turbine telemetry",
  table_properties={
    "pipeline.quality": "gold",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
    "pipelines.autoOptimize.zOrderCols": "date, deviceid"
  }
)

dlt.apply_changes(
  target = "turbine_enriched",
  source = "agg_turbine_gold_view",
  keys = ["date","window","deviceId"],
  sequence_by = col("date"),
  ignore_null_updates = False,
  apply_as_deletes = None,
  column_list = None,
  except_column_list = None
)
