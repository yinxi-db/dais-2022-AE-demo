# dais-2022-AE-demo
Build auto-encoder for anomaly detection, trained and inference distributedly

This repo is forked from an [IoT streaming example using the LakeHouse Architecture](guanjieshen/databricks-iot-demo) and focus on machine learning modeling. 

/data_prep_dlt: generates simulated sensor data for multiple wind turbines in a wind farm and build [Delta-Live-Table pipelines](https://docs.microsoft.com/en-us/azure/databricks/data-engineering/delta-live-tables/) for data preprocessing.

/anomaly_detection: contains notebooks that build, train and deploy AutoEncoder anomaly detection models.
