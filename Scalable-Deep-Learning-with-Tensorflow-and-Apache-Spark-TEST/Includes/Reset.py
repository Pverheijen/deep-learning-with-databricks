# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

def get_cloud():
  with open("/databricks/common/conf/deploy.conf") as f:
    for line in f:
      if "databricks.instance.metadata.cloudProvider" in line and "\"GCP\"" in line: return "GCP"
      elif "databricks.instance.metadata.cloudProvider" in line and "\"AWS\"" in line: return "AWS"
      elif "databricks.instance.metadata.cloudProvider" in line and "\"Azure\"" in line: return "MSA"
              
# Does any work to reset the environment prior to testing.
username = spark.sql("SELECT current_user()").first()[0]

cloud = get_cloud()
print(f"Running on {cloud}")

if cloud != "GCP":
  course_dir = f"file:///dbfs/Users/{username}/dbacademy/deep_learning"
else:
  course_dir = f"file:///dbacademy/{username}/deep_learning"

print(f"Removing course directory: {course_dir}")
dbutils.fs.rm(course_dir, True)

# COMMAND ----------

# MAGIC %run "./Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
