# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Spark Review
# MAGIC 
# MAGIC Before we get started with Machine Learning and Deep Learning, let's make sure we all understand how to use Databricks and Spark.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Create a Spark DataFrame
# MAGIC  - Analyze the Spark UI
# MAGIC  - Cache data
# MAGIC  - Go between Pandas and Spark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://files.training.databricks.com/images/spark_cluster_tasks.png)

# COMMAND ----------

# MAGIC %md Let's start off with running some code on our driver, such as `x = 1`. Insert a new cell below.

# COMMAND ----------

# MAGIC %md ## Spark DataFrame
# MAGIC 
# MAGIC Great! Now let's start with a distributed Spark DataFrame.

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 1000000)
      .withColumn('id', (col('id') / 1000).cast('integer'))
      .withColumn('v', rand(seed=1)))

# COMMAND ----------

# MAGIC %md Why were no Spark jobs kicked off above? Well, we didn't have to actually "touch" our data, so Spark didn't need to execute anything across the cluster.

# COMMAND ----------

display(df.sample(.001))

# COMMAND ----------

# MAGIC %md ## Count
# MAGIC 
# MAGIC Let's see how many records we have.

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md ## Spark UI
# MAGIC 
# MAGIC Open up the Spark UI - what are the shuffle read and shuffle write fields? The command below should give you a clue.

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md ## Cache
# MAGIC 
# MAGIC For repeated access, it will be much faster if we cache our data.

# COMMAND ----------

df.cache().count()

# COMMAND ----------

# MAGIC %md ## Re-run Count
# MAGIC 
# MAGIC Wow! Look at how much faster it is now!

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md ## Pandas
# MAGIC 
# MAGIC Let's convert our Spark DataFrame to a Pandas DataFrame.

# COMMAND ----------

df.toPandas()

# COMMAND ----------

# MAGIC %md ## Wrap-up
# MAGIC 
# MAGIC Alright! Now that you know the basics of Spark, let's get started!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
