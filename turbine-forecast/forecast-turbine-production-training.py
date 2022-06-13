# Databricks notebook source
try:
    df = spark.table("pasa_demo.turbine_enriched")
except:
    raise "turbine data table (pasa_demo.turbine_enriched) is not available"

# COMMAND ----------

from pyspark.sql.functions import col, window, lead, pandas_udf, lit
from pyspark.sql.window import Window

w = Window().partitionBy("deviceId").orderBy("window")
df = (df
      .withColumn("Power" ,0.1*col("temperature") / col("humidity")*3.1416*25*col("windspeed")*col("rpm") ) ## compute power
      .withColumn("power_forecast_target", lead("Power",12).over(w)) # compute forecast target
      .withColumn("forecast_window", lead("window",12).over(w)) # compute forecast window
     )
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Build model to predict power forecast

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
import pyspark.sql.types as t
import pandas as pd
from sklearn.metrics import mean_squared_error

train_return_schema = t.StructType([
    t.StructField("deviceId", t.StringType()), # unique device ID
    t.StructField("model_path", t.StringType()), # path to the model for a given device
    t.StructField("mse", t.FloatType())          # metric for model performance
])

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an sklearn model on grouped instances
    """
    features=["rpm","angle","temperature","humidity","windspeed"]
    target="power_forecast_target"
    
    # Pull metadata
    device_id = df_pandas["deviceId"].iloc[0]
    run_id = df_pandas["run_id"].iloc[0] # Pulls run ID to do a nested run

    # Train the model
    X = df_pandas[features]
    y = df_pandas[target]
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Evaluate the model
    predictions = rf.predict(X)
    mse = mean_squared_error(y, predictions) # Note we could add a train/test split

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id) as outer_run:
        # Small hack for for running as a job
        experiment_id = outer_run.info.experiment_id

        # Create a nested run for the specific device
        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            # Create a return pandas DataFrame that matches the schema above
            return_df = pd.DataFrame([[device_id, artifact_uri, mse]], 
                                    columns=["deviceId",  "model_path", "mse"])

    return return_df 

# COMMAND ----------

import mlflow
with mlflow.start_run(run_name="Training session for all devices") as run:
    run_id = run.info.run_id

    model_directories_df = (df.dropna()
        .withColumn("run_id", lit(run_id)) # Add run_id
        .groupby("deviceId")
        .applyInPandas(train_model, schema=train_return_schema)
    )

## materialize the dataframe
model_directories_df.cache().count()
display(model_directories_df)

# COMMAND ----------

model_directories_df.write.saveAsTable("pasa_demo.turbine_forecast_model")

# COMMAND ----------

# MAGIC %md
# MAGIC **Batch Inference**

# COMMAND ----------

from datetime import timedelta

apply_return_schema = t.StructType([
    t.StructField("deviceId", t.StringType()),
    t.StructField("forecast_window", t.TimestampType()),
    t.StructField("power_forecast_prediction", t.FloatType())
])

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Applies model to data for a particular device, represented as a pandas DataFrame
    """
    model_path = df_pandas["model_path"].iloc[0]

    features=["rpm","angle","temperature","humidity","windspeed"]
    X = df_pandas[features]

    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "forecast_window": df_pandas["window"]+timedelta(hours=1),
        "deviceId": df_pandas["deviceId"],
        "power_forecast_prediction": prediction
    })
    return return_df

# COMMAND ----------

prediction_df = (df.join(model_directories_df.select("deviceId", "model_path"), how="left", on="deviceId")
                 .groupby("deviceId")
                 .applyInPandas(apply_model, schema=apply_return_schema)
                )
display(prediction_df)

# COMMAND ----------

## write to tabel
prediction_df.write.mode("overWrite").option("overwriteSchema", "true").saveAsTable("pasa_demo.turbine_forecast")

# COMMAND ----------

# MAGIC %md
# MAGIC **Notes** 
# MAGIC We can do multi-step forecasting as well by engineering multi-step looking-forward power targets. Models suitable here can be a LSTM network

# COMMAND ----------

# MAGIC %md
# MAGIC **Online Inference**
# MAGIC 
# MAGIC Register an individual turbine model to model registry and enable serving from the UI

# COMMAND ----------

model_uri = model_directories_df.filter("deviceId == 'WindTurbine-7'").select("model_path").first()[0]
model_uri

# COMMAND ----------

mlflow.register_model(
   model_uri,
   "wind-turbine-7-power-forecast"
)

# COMMAND ----------


