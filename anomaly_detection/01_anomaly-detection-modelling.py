# Databricks notebook source
# MAGIC %md
# MAGIC #### Notebook Setup

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("sensor_table_name", "yz.turbine_enriched")
dbutils.widgets.text("maintenance_table_name", "yz.turbine_maintenance")
dbutils.widgets.text("mlflow_exp_root_path","/Users/yinxi.zhang@databricks.com")
dbutils.widgets.text("result_output_table","yz.trained_ae_models")

# COMMAND ----------

# Pyspark and ML Imports
import os, json, requests
from pyspark.sql import functions as F
import pyspark.sql.types as t
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

import mlflow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(42)

print(mlflow.__version__)
print(tf.__version__)

mlflow.tensorflow.autolog()
## mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True) # mlflow 1.25 release

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge wind turbine sensor data and maitenance records

# COMMAND ----------

sensor_table = dbutils.widgets.get("sensor_table_name")
maintenance_table = dbutils.widgets.get("maintenance_table_name")
df = spark.table(sensor_table).join(spark.table(maintenance_table), 
                                    how="left",
                                    on=["deviceId", "date"]
                                   ).fillna(False)
display(df)

# COMMAND ----------

## for the simplicity of demo, use the same dataset as validation data
val_sensor_table = sensor_table
val_maintenance_table = maitenance_table
df_val = spark.table(val_sensor_table).join(spark.table(val_maintenance_table), 
                                    how="left",
                                    on=["deviceId", "date"]
                                   ).fillna(False)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC #### Distributed Model Training with Pandas Function API
# MAGIC 
# MAGIC [Pandas UDFs](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html) allow us to vectorize Pandas code across multiple nodes in a cluster. Here we create a UDF to train an AE model against all the historic data for a particular turbine. We use a Grouped Map UDF as we perform this model training on the pump group level.
# MAGIC 
# MAGIC Steps of training a single AutoEncoder are:
# MAGIC 1. build AE neural network
# MAGIC 2. train the AE with normal training data only
# MAGIC 3. use a validation dataset (includes both normal and amomaly data) to select reconstructed error threshold:
# MAGIC     - generate predictions of the validation data
# MAGIC     - compute reconstructed error from raw AE model predictions
# MAGIC     - build reconstructed error theshold metrics and choose the best threshold
# MAGIC     
# MAGIC We will build [grouped map pandas UDFs](https://spark.apache.org/docs/2.4.4/sql-pyspark-pandas-with-arrow.html#grouped-map) for each module so we can train and compute threshold for each turbine model in parallel.

# COMMAND ----------

# MAGIC %md
# MAGIC Create a MLflow experiment if run from a repo.

# COMMAND ----------

from mlflow.tracking import MlflowClient
mlflow_exp_root_path = dbutils.widgets.get("mlflow_exp_root_path")
# Create an experiment with a name that is unique and case sensitive.
client = MlflowClient()
experiment_id = client.create_experiment(f"{mlflow_exp_root_path}/anomaly_detection_wind_turbines")

# COMMAND ----------

experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC Training Function

# COMMAND ----------

def train(features, df_train):        
    
    def build_auto_encoder_decoder(features, df_input):
        input_dim = len(features)
        inputs = Input(shape=input_dim)
        ## normalization layer
        scale_layer = Normalization()
        scale_layer.adapt(df_input)

        processed_input = scale_layer(inputs)
        ## AE model
        ae_model = Sequential([
        Dense(8, input_dim=input_dim, activation="relu"),
        Dense(2, activation="relu"), ## encoder
        Dense(2, activation="relu"), ## decoder
        Dense(8, activation="relu"),
        Dense(input_dim, activation="linear")
      ])
        ## final model
        outputs = ae_model(processed_input)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mse"])
        return model
      
    def train_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
        '''
        Trains an sklearn model on grouped instances
        '''
        import mlflow
        mlflow.tensorflow.autolog()
        # Pull metadata
        device_id = df_pandas['deviceId'].iloc[0]
        parent_run_id = df_pandas['parent_run_id'].iloc[0]
        # Train the model
        X = df_pandas[features]
        model = build_auto_encoder_decoder(features, X)
        
        with mlflow.start_run(experiment_id=experiment_id, run_id=parent_run_id):
            with mlflow.start_run(run_name=device_id, nested=True, experiment_id=experiment_id) as child_run:
                mlflow.log_param("deviceId", device_id)
                model.fit(X, X, validation_split=0.2, epochs=20,
                          callbacks=[EarlyStopping(patience=2,restore_best_weights=True)]
                         )

        returnDF = pd.DataFrame([[device_id, parent_run_id, child_run.info.run_id]], 
                    columns=["deviceId", "parent_run_id", "mlflow_run_id"])
        return returnDF
    

    ## training udf return schema
    train_schema = t.StructType([t.StructField('deviceId', t.StringType()), # unique pump ID
                           t.StructField('parent_run_id', t.StringType()),         # run id of parent mlflow run
                           t.StructField('mlflow_run_id', t.StringType()),         # run id of child mlflow run
    ])
    
    ## train with mlflow logging
    with mlflow.start_run(experiment_id=experiment_id) as run:
        df_train_results = (df_train
                            .withColumn("parent_run_id", F.lit(run.info.run_id))
                            .groupby("deviceId").applyInPandas(train_udf, train_schema) ) 
    
    return df_train_results

# COMMAND ----------

features = ["rpm","angle","temperature","humidity","windspeed"]
df_train_results = train(features, df)

# COMMAND ----------

## materialize dataframe
df_train_results.cache().count()
display(df_train_results)

# COMMAND ----------

# MAGIC %md
# MAGIC **Evaluation**

# COMMAND ----------

def predict(df_eval):
    def predict_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
        '''
        load the trained model and get prediction
        '''
        import mlflow
        import numpy as np
        
        # Pull metadata
        device_id = df_pandas['deviceId'].iloc[0]
        mlflow_run_id = df_pandas["mlflow_run_id"].iloc[0]
        model_uri = f"runs:/{mlflow_run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        X = df_pandas[features]
        raw_predictions = pd.DataFrame(data=model.predict(X).values, columns=[f"{c}_pred" for c in features]) ## raw predictions from tf models
        raw_predictions = pd.concat([df_pandas, raw_predictions], axis=1)                               
        ## add additional error cols
        error_cols = []
        for c in features:
            raw_predictions[c+"_log_error"] = np.log((raw_predictions[f"{c}_pred"] - X[c]).abs())
            error_cols.append(c+"_log_error")
        ## reconstructed error
        raw_predictions["reconstructed_error"] = raw_predictions[error_cols].mean(axis=1)
        return raw_predictions
      
    ## raw prediction output schema
    predicted_cols = ["reconstructed_error"]
    #predicted_cols = []
    for c in features:
        predicted_cols.extend([c+"_pred", c+"_log_error"])
    pred_schema = (df_eval
                   .select(*df_eval.columns, *[F.lit(0).cast("double").alias(c) for c in predicted_cols])
                   .schema
                      )
    df_eval_predicted = df_eval.groupby("deviceId").applyInPandas(predict_udf, pred_schema)
    
    return df_eval_predicted

# COMMAND ----------

## get raw predictions of validation data
df_eval = df_val.join(df_train_results, 
                       on = "deviceId",
                       how = "left")
df_eval_predicted = predict(df_eval)
df_eval_predicted.cache().count()
display(df_eval_predicted)

# COMMAND ----------

def evaluate(df):  

    def select_threshold(error_h, error_f, beta=1):
        """
        select threshold based on fbeta of all samples in data duration
        """
        from sklearn.metrics import fbeta_score
        range_min, range_max = min(min(error_h), min(error_f)),max(max(error_h),max(error_f))
        sample_points = np.linspace(range_min, range_max, 100)
        metric_max = 0
        p_best = range_min
        for p in sample_points:
            precision = (error_h>p).sum()/((error_h>p).sum()+(error_f>p).sum())
            recall = (error_f>p).sum()/len(error_f)
            fbeta = (1. + beta**2)*precision*recall/(beta**2*precision+recall)
            if fbeta > metric_max:
                p_best = p
                metric_max = fbeta

        return p, metric_max

    def eval_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
        device_id = df_pandas['deviceId'].iloc[0]
        error_h = df_pandas.loc[df_pandas.maintenance==False, "reconstructed_error"].values
        error_f = df_pandas.loc[df_pandas.maintenance==True, "reconstructed_error"].values
        p, metric_max = select_threshold(error_h,error_f,beta=1)
        returnDF = pd.DataFrame([[device_id, p, metric_max]], 
                    columns=["deviceId", "threshold", "eval_fbeta"])
        return returnDF

    ## distribute threshold computation
    schema = t.StructType([t.StructField('deviceId', t.StringType()), # unique pump ID
                           t.StructField('threshold', t.FloatType()),         # anomaly threshold
                           t.StructField('eval_fbeta', t.FloatType()),         # evaluation metrics
    ])
    return df.groupby("deviceId").applyInPandas(eval_udf, schema)

# COMMAND ----------

df_eval_results = evaluate(df_eval_predicted)

# COMMAND ----------

## inspect training threshold, metrics and mlflow run artifacts
df_results = df_eval_results.join(df_train_results, 
                       on = "deviceId",
                       how = "left").drop("parent_run_id")
display(df_results)
df_results.write.mode("overWrite").saveAsTable(dbutils.widgets.get("result_output_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distributed inference with the groupby predict Pandas Function API
# MAGIC 
# MAGIC To generate final anomaly prediction, we need both the AE model and computed reconstructed error threshold. The `predict` function we built in earlier cells applies a grouped mpa pandas UDF `predict_udf` to the entire spark dataframe, which contains data from all turbines, in parallel. It picks up the mlflow run id to each `deviceId` and load the corresponding model and generate predictions. This is the optimized way to perform batch inference.
# MAGIC 
# MAGIC **Note** this can feed back to the DLT pipeline for data streaming pipelines

# COMMAND ----------

df_predicted = (predict(df.join(df_results,
                               on = "deviceId",
                               how = "left")
                      )
               .withColumn("Anomaly_Prediction", F.col("reconstructed_error")>F.col("threshold")))
df_predicted.display()

