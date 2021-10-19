# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Agenda
# MAGIC ## Scalable Deep Learning with TensorFlow and Apache Spark™
# MAGIC 
# MAGIC **Cluster Requirements:**
# MAGIC * DBR 9.1 ML

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 1 AM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 30m  | **Introductions & Setup**                               | *Registration, Courseware & Q&As* |
# MAGIC | 25m    | **[Spark Review]($./DL 00 - Spark Review) (optional)**    | Review core concepts of Apache Spark|
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 35m  | **[Linear Regression]($./DL 01 - Linear Regression)** | Build a linear regression model using Sklearn and reimplement it in Keras </br> Modify # of epochs </br> Visualize loss | 
# MAGIC | 30m  | **[Keras]($./DL 02 - Keras)**  | Modify these parameters for increased model performance: activation functions, loss functions, optimizer, batch size </br> Save and load models |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 25m    | **[Keras Lab]($./Labs/DL 02L - Keras Lab)**    | Build and evaluate your first Keras model! |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 1 PM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 65m  | **[Advanced Keras]($./DL 03 - Advanced Keras)** & **[Lab]($./Labs/DL 03L - Advanced Keras Lab)**      | Perform data standardization for better model convergence </br> Add validation data </br> Generate model checkpointing/callbacks </br> Use TensorBoard </br> Apply dropout regularization |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 50m |**[MLflow]($./DL 04 - MLflow)** & **[Lab]($./Labs/DL 04L - MLflow Lab)**| Log experiments with MLflow</br> View MLflow UI</br> Generate a UDF with MLflow and apply to a Spark DataFrame |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 40m  | **[HyperOpt]($./DL 05 - Hyperopt)** & **[Lab]($./Labs/DL 05L - Hyperopt Lab)** | Use HyperOpt with SparkTrials to perform distributed hyperparameter search |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 35m  | **[Horovod]($./DL 06 - Horovod)** | Use Horovod to train a distributed neural network </br> Distributed Deep Learning best practices |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 2 AM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 20m  | **Review**                               | *Review of Day 1* |
# MAGIC | 30m  | **[Horovod Petastorm]($./DL 06a - Horovod Petastorm)** | Use Horovod to train a distributed neural network using Parquet files + Petastorm|
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 35m  | **[Horovod Lab]($./Labs/DL 06L - Horovod Lab)** | Prepare your data for use with Horovod</br> Distribute the training of our model using HorovodRunner</br> Use Parquet files as input data for our distributed deep learning model with Petastorm + Horovod | 
# MAGIC | 35m  | **[Model Interpretability]($./DL 07 - Model Interpretability)**  | Use LIME and SHAP to understand which features are most important in the model's prediction for that data point |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 40m    | **[CNNs]($./DL 08 - CNNs)**    | Analyze popular CNN architectures </br> Apply pre-trained CNNs to images using Pandas Scalar Iterator UDF |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 2 PM
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 20m  | **[SHAP for CNNs]($./Labs/DL 08L - SHAP for CNNs Lab)** | Use SHAP to visualize how the CNN makes predictions | 
# MAGIC | 30m  | **[Transfer Learning]($./DL 09 - Transfer Learning)**  | Perform transfer learning to create a cat vs dog classifier |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 25m  | **[Data Augmentation]($./DL 09a - Data Augmentation)**  | Apply data augmentation to improve transfer learning performance |
# MAGIC | 30m  | **[Transfer Learning Lab]($./Labs/DL 09L - Transfer Learning Lab)**  | Build a model to predict if a patient has pneumonia using transfer learning on chest X-rays |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 30m  | **[Model Serving]($./DL 09b - Model Serving)**  | Real time deployment of a convolutional neural network using REST and Databricks MLflow Model Serving |
# MAGIC | 25m  | **[Generative Adversarial Networks]($./DL 10 - Generative Adversarial Networks)**  | Understand Generative and discriminative models </br> Build GANs |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 25m  | **[PyTorch 101 (Optional)]($./DL 11 - PyTorch 101)**  | Fundamentals of PyTorch  |
# MAGIC | 25m  | **[Best Practices]($./DL 12 - Best Practices)**  | Discuss DL best practices, state of the art, and new research areas  |

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
