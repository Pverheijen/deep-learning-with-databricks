# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Lab
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Add MLflow to your experiments
# MAGIC  - Create an EarlyStopping Callback
# MAGIC  - Create a UDF to apply your Keras model to a Spark DataFrame
# MAGIC   
# MAGIC **Bonus:**
# MAGIC * Modify your model (and track the parameters) to get the lowest MSE!

# COMMAND ----------

# MAGIC %pip install tensorflow-datasets

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load & Prepare Data

# COMMAND ----------

import tensorflow_datasets as tfds
import pandas as pd

# Import Dataset
wine_quality_tfds = tfds.load("wine_quality", split="train", shuffle_files=False)
wine_quality_pdf = tfds.as_dataframe(wine_quality_tfds)
wine_quality_pdf.columns = wine_quality_pdf.columns.str.replace("features/","")
wine_quality_pdf.head(5)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(wine_quality_pdf.drop("quality", axis=1),
                                                    wine_quality_pdf["quality"],
                                                    test_size=0.2,
                                                    random_state=1)
# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train-Validation Split
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.25,
                                                              random_state=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build_model
# MAGIC Create a `build_model()` function. Because Keras models are stateful, we want to get a fresh model every time we are trying out a new experiment.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
tf.random.set_seed(42)

def build_model():
  return Sequential([Dense(50, input_dim=11, activation="relu"),
                     Dense(20, activation="relu"),
                     Dense(1, activation="linear")])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Early Stopping
# MAGIC 
# MAGIC Let's add [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) to our network to we stop the training when a monitored metric has stopped improving.

# COMMAND ----------

# ANSWER
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

filepath = f"{working_dir}/keras_mlflow.ckpt"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=2, mode="auto", restore_best_weights=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Track Experiments!
# MAGIC 
# MAGIC Now let's use MLflow to automatically track experiments with [mlflow.tensorflow.autolog()](https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#mlflow.tensorflow.autolog). Try changing your hyperparameters, such as `epochs` or `batch_size` and compare what gives you the best result.
# MAGIC 
# MAGIC :NOTE: You can always add manual MLflow logging statements to log things in addition to the autologged values.

# COMMAND ----------

# ANSWER
import mlflow

mlflow.tensorflow.autolog()

with mlflow.start_run() as run:
  model = build_model()
  model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

  model.fit(X_train_split, 
            y_train_split, 
            validation_data=(X_val, y_val), 
            epochs=30, 
            batch_size=32, 
            callbacks=[checkpointer, early_stopping], 
            verbose=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## User Defined Function
# MAGIC 
# MAGIC Let's now register our Keras model as a Spark UDF to apply to rows in parallel.

# COMMAND ----------

# ANSWER
import pandas as pd

predict = mlflow.pyfunc.spark_udf(spark, f"runs:/{run.info.run_id}/model")

X_test_df = spark.createDataFrame(pd.concat([pd.DataFrame(X_test, columns=wine_quality_pdf.drop("quality", axis=1).columns), 
                                             pd.DataFrame(y_test.values, columns=["label"])], axis=1))

display(X_test_df.withColumn("prediction", predict(*wine_quality_pdf.drop("quality", axis=1).columns)))


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
