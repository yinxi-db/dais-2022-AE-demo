# Databricks notebook source
# MAGIC %md ### Backfill Historical Data
# MAGIC In order to train a model, we will need to backfill our streaming data with historical data. The cell below generates 1 year of historical hourly turbine and weather data and inserts it into our Gold Delta table.

# COMMAND ----------

import pandas as pd
import numpy as np

# Function to simulate generating time-series data given a baseline, slope, and some seasonality
def generate_series(time_index, baseline, slope=0.01, period=365*24*12):
  rnd = np.random.RandomState(time_index)
  season_time = (time_index % period) / period
  seasonal_pattern = np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))
  return baseline * (1 + 0.1 * seasonal_pattern + 0.1 * rnd.randn(len(time_index)))
  
# Get start and end dates for our historical data
dates = spark.sql('select max(date)-interval 365 days as start, max(date) as end from pasa_demo.turbine_enriched').toPandas()
  
# Get the baseline readings for each sensor for backfilling data
turbine_enriched_pd = spark.table('pasa_demo.turbine_enriched').toPandas()
baselines = turbine_enriched_pd.min()[3:8]
devices = turbine_enriched_pd['deviceId'].unique()

# Iterate through each device to generate historical data for that device
print("---Generating Historical Enriched Turbine Readings---")
for deviceid in devices:
  print(f'Backfilling device {deviceid}')
  windows = pd.date_range(start=dates['start'][0], end=dates['end'][0], freq='5T') # Generate a list of hourly timestamps from start to end date
  historical_values = pd.DataFrame({
    'date': windows.date,
    'window': windows, 
    'winddirection': np.random.choice(['N','NW','W','SW','S','SE','E','NE'], size=len(windows)),
    'deviceId': deviceid
  })
  time_index = historical_values.index.to_numpy()                                 # Generate a time index

  for sensor in baselines.keys():
    historical_values[sensor] = generate_series(time_index, baselines[sensor])    # Generate time-series data from this sensor

  # Write dataframe to enriched_readings Delta table
  spark.createDataFrame(historical_values).write.format("delta").mode("append").saveAsTable("pasa_demo.__apply_changes_storage_turbine_enriched")
  
# Create power readings based on weather and operating conditions
print("---Generating Historical Turbine Power Readings---")
spark.sql(f'CREATE TABLE pasa_demo.turbine_power USING DELTA PARTITIONED BY (date) AS SELECT date, window, deviceId, 0.1 * (temperature/humidity) * (3.1416 * 25) * windspeed * rpm AS power FROM pasa_demo.turbine_enriched')

# Create a maintenance records based on peak power usage
print("---Generating Historical Turbine Maintenance Records---")
spark.sql(f'CREATE TABLE pasa_demo.turbine_maintenance USING DELTA  AS SELECT DISTINCT deviceid, FIRST(date) OVER (PARTITION BY deviceid, year(date), month(date) ORDER BY power) AS date, True AS maintenance FROM pasa_demo.turbine_power')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Optimize all 3 tables for querying and model training performance
# MAGIC OPTIMIZE pasa_demo.turbine_power ZORDER BY deviceid, window;
# MAGIC OPTIMIZE pasa_demo.turbine_maintenance ZORDER BY deviceid;
