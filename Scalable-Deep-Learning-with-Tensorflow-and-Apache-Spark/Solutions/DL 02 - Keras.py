# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Keras
# MAGIC 
# MAGIC In this notebook, we will build upon the concepts introduced in the previous lab to build a neural network that is more powerful than a simple linear regression model!
# MAGIC 
# MAGIC We will use the California Housing Dataset.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC   - Will modify these parameters for increased model performance:
# MAGIC     - Activation functions
# MAGIC     - Loss functions
# MAGIC     - Optimizer
# MAGIC     - Batch Size
# MAGIC   -  Save and load models

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2,
                                                    random_state=1)

print(cal_housing.DESCR)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Recall from Last Lab
# MAGIC 
# MAGIC ##### Steps to build a Keras model
# MAGIC <img style="width:20%" src="https://files.training.databricks.com/images/5_cycle.jpg" >

# COMMAND ----------

# MAGIC %md ## Define a Network
# MAGIC 
# MAGIC Let's not just reinvent linear regression. Let's build a model, but with multiple layers using the [Sequential model](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) from Keras.
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/Neural_network.svg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Activation Function
# MAGIC 
# MAGIC If we keep the activation as linear, then we aren't utilizing the power of neural networks!! The power of neural networks derives from the non-linear combinations of linear functions.
# MAGIC 
# MAGIC **RECAP:** So what are our options for [activation functions](http://cs231n.github.io/neural-networks-1/#actfun)? 

# COMMAND ----------

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
tf.random.set_seed(42)

model = Sequential()

# Input layer
model.add(Dense(20, input_dim=8, activation="relu")) 

# Automatically infers the input_dim based on the layer before it
model.add(Dense(20, activation="relu")) 

# Output layer
model.add(Dense(1, activation="linear")) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Alternative Keras Model Syntax

# COMMAND ----------

def build_model():
    return Sequential([Dense(20, input_dim=8, activation="relu"),
                       Dense(20, activation="relu"),
                       Dense(1, activation="linear")]) # Keep the last layer as linear because this is a regression problem

# COMMAND ----------

# MAGIC %md
# MAGIC We can check the model definition by calling `.summary()`

# COMMAND ----------

model = build_model()
model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Loss Functions + Metrics
# MAGIC 
# MAGIC In Keras, the *loss function* is the function for our optimizer to minimize. *[Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)* are similar to a loss function, except that the results from evaluating a metric are not used when training the model.
# MAGIC 
# MAGIC **Recap:** Which loss functions should we use for regression? Classification?

# COMMAND ----------

from tensorflow.keras import metrics
from tensorflow.keras import losses

loss = "mse" # Or loss = losses.mse
metrics = ["mae", "mse"] # Or metrics = [metrics.mae, metrics.mse]

model.compile(optimizer="sgd", loss=loss, metrics=metrics)
model.fit(X_train, y_train, epochs=10)

# COMMAND ----------

# MAGIC %md
# MAGIC The learning rate is too high for SGD - try decreasing the learning rate to a really small value and it won't explode.

# COMMAND ----------

from tensorflow.keras import optimizers

model = build_model()
optimizer = optimizers.SGD(learning_rate=0.000001)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(X_train, y_train, epochs=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Optimizer
# MAGIC 
# MAGIC Let's try this again, but using the Adam optimizer. There are a lot of optimizers out there, and here is a [great blog post](http://ruder.io/optimizing-gradient-descent/) illustrating the various optimizers.
# MAGIC 
# MAGIC When in doubt, the Adam optimizer does a very good job. If you want to adjust any of the hyperparameters, you will need to import the optimizer from `optimizers` instead of passing in the name as a string.

# COMMAND ----------

from tensorflow.keras import optimizers

model = build_model()
optimizer = optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = model.fit(X_train, y_train, epochs=20)

# COMMAND ----------

import matplotlib.pyplot as plt

def view_model_loss():
    plt.clf()
    plt.semilogy(history.history["loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()

view_model_loss()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Batch Size
# MAGIC 
# MAGIC Let's set our `batch_size` (how much data to be processed simultaneously by the model) to 64, and increase our `epochs` to 20. Mini-batches are often a power of 2, to facilitate memory allocation on GPU (typically between 16 and 512).
# MAGIC 
# MAGIC 
# MAGIC Also, if you don't want to see all of the intermediate values print out, you can set the `verbose` parameter: 0 = silent, 1 = progress bar, 2 = one line per epoch (defaults to 1)

# COMMAND ----------

model = build_model()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=2)

# COMMAND ----------

# MAGIC %md ## 5. Evaluate

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Model, Load Model and Train More
# MAGIC 
# MAGIC Whenever you train neural networks, you want to save them. This way, you can reuse them later! 
# MAGIC 
# MAGIC In our case, we want to save  need to save both the architecture and the weights, so we will use `model.save`. If you only want to save the weights, you can use `model.save_weights`.

# COMMAND ----------

filepath = f"{working_dir}/keras_checkpoint_weights.ckpt".replace("dbfs:/", "/dbfs/")

model.save(filepath)

# COMMAND ----------

# MAGIC %md You can load both the model and architecture together using `load_model()`

# COMMAND ----------

from tensorflow.keras.models import load_model

new_model = load_model(filepath)

# COMMAND ----------

# MAGIC %md
# MAGIC Check that the model architecture is the same.

# COMMAND ----------

new_model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's train it for one more epoch (we need to recompile), and then save those weights.  This is a *warm start.*

# COMMAND ----------

new_model.compile(optimizer="adam", loss="mse")
new_model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=2)
new_model.save_weights(filepath)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
