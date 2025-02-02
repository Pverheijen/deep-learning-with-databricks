# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Horovod Lab
# MAGIC 
# MAGIC In this lab we are going to build upon our previous lab model trained on the Wine Quality dataset and distribute the deep learning training process using both HorovodRunner and Petastorm.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Prepare your data for use with Horovod
# MAGIC  - Distribute the training of our model using HorovodRunner
# MAGIC  - Use Parquet files as input data for our distributed deep learning model with Petastorm + Horovod

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ## 1. Load and process data
# MAGIC 
# MAGIC We again load the Wine Quality data. However, as we saw in the demo, for Horovod we want to shard the data before passing into HorovodRunner. 
# MAGIC 
# MAGIC For the **`get_dataset`** function below, load the data, split into 80/20 train-test, standardize the features and return train and test sets.

# COMMAND ----------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_dataset(rank=0, size=1):
    scaler = StandardScaler()

    wine_quality = pd.read_parquet(f"{datasets_dir}/winequality/red.parquet".replace("dbfs:/", "/dbfs/"))
    X = wine_quality.drop("quality", axis=1)
    y = wine_quality["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train[rank::size])
    y_train = y_train[rank::size]
    X_test = scaler.transform(X_test[rank::size])
    y_test = y_test[rank::size]

    return (X_train, y_train), (X_test, y_test)

# COMMAND ----------

# MAGIC %md ##2. Build Model
# MAGIC 
# MAGIC Using the same model from earlier, let's define our model architecture

# COMMAND ----------

import tensorflow as tf
tf.random.set_seed(42)

def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    return Sequential([Dense(50, input_dim=11, activation="relu"),
                       Dense(20, activation="relu"),
                       Dense(1, activation="linear")])

# COMMAND ----------

# MAGIC %md ## 3. Horovod
# MAGIC 
# MAGIC In order to distribute the training of our Keras model with Horovod, we must define our **`run_training_horovod`** training function

# COMMAND ----------

# ANSWER
import horovod.tensorflow.keras as hvd
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback

BATCH_SIZE = 16
NUM_EPOCHS = 10
WARMUP_EPOCHS = 5
INITIAL_LR = 0.001

def run_training_horovod():
    # Horovod: initialize Horovod.
    hvd.init()
    print(f"Rank is: {hvd.rank()}")
    print(f"Size is: {hvd.size()}")

    (X_train, y_train), (X_test, y_test) = get_dataset(hvd.rank(), hvd.size())

    model = build_model()
    optimizer = optimizers.Adam(learning_rate=INITIAL_LR)
    optimizer = hvd.DistributedOptimizer(optimizer)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    checkpoint_dir = f"{working_dir}/horovod_checkpoint_weights_lab.ckpt".replace("dbfs:/", "/dbfs/")

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=INITIAL_LR*hvd.size(), warmup_epochs=WARMUP_EPOCHS, verbose=1),

        # Reduce the learning rate if training plateaus.
        ReduceLROnPlateau(patience=10, verbose=1, monitor="loss"),
      
        # Print out the learning rate for each epoch
        LambdaCallback(on_epoch_begin=lambda epoch,logs: print(f"current epoch id = {epoch}, learning rate = {model.optimizer.learning_rate.numpy()}"))
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint(checkpoint_dir, save_best_only=True, monitor="loss"))

    history = model.fit(X_train, y_train, callbacks=callbacks, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# COMMAND ----------

# MAGIC %md Let's now run our model on all workers.

# COMMAND ----------

# ANSWER
from sparkdl import HorovodRunner

hr = HorovodRunner(np=spark.sparkContext.defaultParallelism, driver_log_verbosity="all")
hr.run(run_training_horovod)

# COMMAND ----------

# MAGIC %md ## 4. Horovod with Petastorm
# MAGIC 
# MAGIC We're now going to build a distributed deep learning model capable of handling data in Apache Parquet format. To do so, we can use Horovod along with Petastorm. 
# MAGIC 
# MAGIC First let's load the Wine Quality data, and create a Spark DataFrame from the training data.

# COMMAND ----------

# Load dataset
wine_quality = pd.read_parquet(f"{datasets_dir}/winequality/red.parquet".replace("dbfs:/", "/dbfs/"))
X = wine_quality.drop("quality", axis=1)
y = wine_quality["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Concatenate our features and label, then create a Spark DataFrame from our Pandas DataFrame.
data = pd.concat([pd.DataFrame(X_train, columns=X.columns), 
                  pd.DataFrame(y_train.values, columns=["label"])], axis=1)
train_df = spark.createDataFrame(data)
display(train_df)

# COMMAND ----------

# MAGIC %md ### Create Vectors
# MAGIC 
# MAGIC Use the VectorAssembler to combine all the features (not including the label) into a single column called **`features`**.

# COMMAND ----------

# ANSWER
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=list(X.columns), outputCol="features")
vec_train_df = vec_assembler.transform(train_df).select("features", "label")
display(vec_train_df)

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

file_path = f"file:///{working_dir}/training_data"
dbutils.fs.rm(file_path, recurse=True)
dbutils.fs.mkdirs(file_path)
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, file_path)
converter_train = make_spark_converter(vec_train_df)

# COMMAND ----------

# ANSWER
import horovod.tensorflow.keras as hvd

def run_training_horovod():
    # Horovod: initialize Horovod.
    hvd.init()
    with converter_train.make_tf_dataset(batch_size = BATCH_SIZE,
                                         num_epochs=None, 
                                         cur_shard=hvd.rank(), 
                                         shard_count=hvd.size()) as train_dataset:

        dataset = train_dataset.map(lambda x: (x.features, x.label))
        model = build_model()
        steps_per_epoch = len(converter_train) // (BATCH_SIZE * hvd.size())
        optimizer = optimizers.Adam(learning_rate=INITIAL_LR)
        optimizer = hvd.DistributedOptimizer(optimizer)
        model.compile(optimizer=optimizer, loss="mse")

        checkpoint_dir = f"{working_dir}/petastorm_checkpoint_weights_lab.ckpt".replace("dbfs:/", "/dbfs/")

        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(initial_lr=INITIAL_LR*hvd.size(), warmup_epochs=5, verbose=1),
            ReduceLROnPlateau(monitor="loss", patience=10, verbose=1),
            LambdaCallback(on_epoch_begin=lambda epoch,logs: print(f"current epoch id = {epoch}, learning rate = {model.optimizer.learning_rate.numpy()}"))
        ]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(checkpoint_dir, save_best_only=True, monitor="loss"))

        history = model.fit(dataset, callbacks=callbacks, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Finally, let's run our newly define Horovod training function with Petastorm to run across all workers.

# COMMAND ----------

# ANSWER
from sparkdl import HorovodRunner

hr = HorovodRunner(np=spark.sparkContext.defaultParallelism, driver_log_verbosity="all")
hr.run(run_training_horovod)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
