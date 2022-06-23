# Databricks notebook source
# MAGIC %md ### Backfill Historical Data
# MAGIC In order to train a model, we will need to backfill our streaming data with historical data. The cell below generates 1 year of historical hourly turbine and weather data and inserts it into our Gold Delta table.

# COMMAND ----------

db_name = dbutils.widgets.get("database")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

# COMMAND ----------

devices = ["WindTurbine-"+str(i) for i in range(int(dbutils.widgets.get("n_device")))]

# COMMAND ----------

dates = {"start":dbutils.widgets.get("start_date"), "end": dbutils.widgets.get("end_date")}

# COMMAND ----------

baselines = {"rpm"          : 6.859119,
              "angle"       : 5.001729,
              "temperature" : 25.261187,
              "humidity"    : 65.941392,
              "windspeed"   : 6.594139
            }

# COMMAND ----------

import pandas as pd
import numpy as np

# Function to simulate generating time-series data given a baseline, slope, and some seasonality
def generate_series(time_index, baseline, slope=0.01, period=365*24*12):
  rnd = np.random.RandomState(time_index)
  season_time = (time_index % period) / period
  seasonal_pattern = np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))
  return baseline * (1 + 0.1 * seasonal_pattern + 0.1 * rnd.randn(len(time_index)))

# COMMAND ----------

import pyspark.sql.functions as f
# Iterate through each device to generate historical data for that device
print("---Generating Historical Enriched Turbine Readings---")
for deviceid in devices:
  print(f'Backfilling device {deviceid}')
  windows = pd.date_range(start=dates['start'], end=dates['end'], freq='10T') # Generate a list of hourly timestamps from start to end date, 10 mins interval
  historical_values = pd.DataFrame({
    'date': windows.date,
    'window': windows, 
    'deviceId': deviceid
  })
  time_index = historical_values.index.to_numpy()                                 # Generate a time index

  for sensor in baselines.keys():
    historical_values[sensor] = generate_series(time_index, baselines[sensor])    # Generate time-series data from this sensor

  # Write dataframe to enriched_readings Delta table
  spark.createDataFrame(historical_values).write.format("delta").mode("append").saveAsTable(f"{db_name}.turbine_enriched")
  
# spark.sql(f'CREATE TABLE {db_name}.turbine_power USING DELTA PARTITIONED BY (date) AS SELECT date, window, deviceId, 0.1 * (temperature/humidity) * (3.1416 * 25) * windspeed * rpm AS power FROM {db_name}.turbine_enriched')

# COMMAND ----------

# Create power readings based on weather and operating conditions
print("---Generating Historical Turbine Power Readings---")
df_power = spark.table(f"{db_name}.turbine_enriched").select("date", "window", "deviceId", 
                                                             (0.1 * (f.col("temperature")/f.col("humidity")) * (3.1416 * 25) * f.col("windspeed") * f.col("rpm")).alias("power"))
df_power.write.partitionBy("date").mode("overWrite").saveAsTable(f"{db_name}.turbine_power")

# COMMAND ----------

# Create a maintenance records based on peak power usage
print("---Generating Historical Turbine Maintenance Records---")
spark.sql(f'CREATE TABLE {db_name}.turbine_maintenance USING DELTA  AS SELECT DISTINCT deviceid, FIRST(date) OVER (PARTITION BY deviceid, year(date), month(date) ORDER BY power) AS date, True AS maintenance FROM {db_name}.turbine_power')

# COMMAND ----------

## Optimize all 3 tables for querying and model training performance
spark.sql(f"OPTIMIZE {db_name}.turbine_power ZORDER BY deviceid, window")
spark.sql(f"OPTIMIZE {db_name}.turbine_maintenance ZORDER BY deviceid")

# COMMAND ----------


