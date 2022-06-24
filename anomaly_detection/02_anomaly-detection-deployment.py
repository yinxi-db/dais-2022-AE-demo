# Databricks notebook source
# MAGIC %md
# MAGIC ### Notebook Setup

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("score_table_name", "yz_dais_2022.turbine_enriched")
dbutils.widgets.text("model_table_name", "yz_dais_2022.trained_ae_models")
dbutils.widgets.text("registered_model_name", "wind-farm-anomaly-detection")

# COMMAND ----------

import mlflow
import tensorflow as tf
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd

tf.random.set_seed(42)

print(mlflow.__version__)
print(tf.__version__)

mlflow.tensorflow.autolog()
## mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True) # mlflow 1.25 release

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load wind turbine sensor data for scoring

# COMMAND ----------

sensor_table = dbutils.widgets.get("score_table_name")
model_table = dbutils.widgets.get("model_table_name")
df_model = spark.table(model_table)
df = spark.table(sensor_table).join(df_model, 
                                    how="left",
                                    on="deviceId"
                                   ).fillna(False)
display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Build Custom MLflow Pyfunc Model
# MAGIC 
# MAGIC A **`pyfunc`** is a generic python model that can define any arbitrary logic, regardless of the libraries used to train it. **This object interoperates with any MLflow functionality, especially downstream scoring tools.**  As such, it's defined as a class with a related directory structure with all of the dependencies.  It is then "just an object" with a various methods such as a predict method.  Since it makes very few assumptions, it can be deployed using MLflow, SageMaker, a Spark UDF, or in any other environment.
# MAGIC 
# MAGIC Check out <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom" target="_blank">the **`pyfunc`** documentation for details</a><br>
# MAGIC 
# MAGIC This section of the notebook, we will build a custom MLflow model with two initialization arguments
# MAGIC  - `features`: features of the raw AE model per turbine
# MAGIC  - `device_model_map`: a dictionary maps individual turbines to its model info (loaded model and threshold)
# MAGIC 
# MAGIC We will later enable online serving for the ensemble model.

# COMMAND ----------

client = MlflowClient()

features = ["rpm","angle","temperature","humidity","windspeed"]
## create model map, load model outside of pyfunc class
device_model_map= df_model.select("deviceId","mlflow_run_id","threshold").toPandas().set_index("deviceId").to_dict("index")
device_model_map

# COMMAND ----------

## save to artifact
artifacts = {}
for device in device_model_map:
    artifacts["model-"+device] = client.download_artifacts(run_id=device_model_map[device]["mlflow_run_id"],
                                                           path="model"
                                                          )
artifacts

# COMMAND ----------

mlflow.pyfunc.load_model('/tmp/tmprv8bijsy/model').predict(df.select(*features).limit(5).toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build model class

# COMMAND ----------

class GroupByAEWrapperModel(mlflow.pyfunc.PythonModel):
  
    def __init__(self, features, device_model_map):
        self.features = features
        self.device_model_map = device_model_map
        
    def featurize(self, df):
        """
        implement featurization logic here
        """
        return df
    
    def load_context(self, context):
        self.models = {}
        for device in self.device_model_map:
            self.models[device] = mlflow.pyfunc.load_model(context.artifacts["model-"+device])
        
        
    def predict(self, context, model_input: pd.DataFrame):
        def predict_per_device(df_pandas):
            import mlflow
            import numpy as np
            from mlflow.tracking.client import MlflowClient
            # Pull metadata
            device_id = df_pandas["deviceId"]
            model = self.models[device_id]

            threshold = self.device_model_map[device_id]["threshold"]
            X = pd.DataFrame(data=[[df_pandas[c] for c in self.features]], columns=self.features)
            
            raw_predictions = pd.DataFrame(data=model.predict(X), columns=[f"{c}_pred" for c in self.features]) ## raw predictions from tf models
            ## add additional error cols
            error_cols = []
            for c in self.features:
                raw_predictions[c+"_log_error"] = np.log((raw_predictions[f"{c}_pred"] - X[c]).abs())
                error_cols.append(c+"_log_error")
            ## reconstructed error
            raw_predictions["Prediction_is_anomaly"] = raw_predictions[error_cols].mean(axis=1)>threshold
            return raw_predictions["Prediction_is_anomaly"].iloc[0]

        ## featurization step 
        processed_input = self.featurize(model_input)
        return processed_input.apply(predict_per_device, result_type="expand", axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Log the custom model

# COMMAND ----------

from sys import version_info

conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        "pip",
        {"pip": [
                "mlflow",
                "boto3==1.16.7",
                "cffi==1.14.5",
                "cloudpickle==1.6.0",
                "configparser==5.0.1",
                "dill==0.3.2",
                "fsspec==0.9.0",
                "pandas==1.2.4",
                "pillow==8.2.0",
                "psutil==5.8.0",
                "pyspark==3.2.1",
                "scipy==1.6.2",
                "tensorflow-cpu==2.8.0",
                "protobuf<4.0.0"
                ]
        },
    ],
    "name": "tf_env"
}

conda_env

# COMMAND ----------

df_inference = df.select("deviceId", *features)## at inference, we only need turbine id and feature cols
signature = infer_signature(df_inference.limit(3).toPandas(), ## input data
                            pd.DataFrame(data=[False],
                                         columns=["Prediction_is_anomaly"]
                                        )    ## prediction data
                           )
model_name = dbutils.widgets.get("registered_model_name")
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
      python_model=GroupByAEWrapperModel(features,device_model_map), 
      artifact_path="groupby_models", 
      input_example = df_inference.limit(3).toPandas(),
      signature=signature,
      artifacts = artifacts,
      registered_model_name=model_name,
      conda_env=conda_env
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load model locally and generate prediction

# COMMAND ----------

imported_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/groupby_models")
imported_model.predict(df_inference.limit(10).toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Enable model serving 
# MAGIC 
# MAGIC From [the model page](https://db-sme-demo-ml-practice.cloud.databricks.com/?o=2524178410398960#mlflow/models/multimodel-serving/serving), click on enable serving to create an Rest Endpoint of the model
# MAGIC 
# MAGIC We can sent queries to the endpoint

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://db-sme-demo-ml-practice.cloud.databricks.com/model-endpoint/wind-farm-anomaly-detection/1/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

score_model(df_inference.limit(3).toPandas())

# COMMAND ----------


