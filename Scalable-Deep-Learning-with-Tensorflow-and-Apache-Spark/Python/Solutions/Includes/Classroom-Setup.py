# Databricks notebook source

import sys

# Suggested fix from Jason Kim & Ka-Hing Cheung
# https://databricks.atlassian.net/browse/ES-176458
wsfsPaths = list(filter(lambda p : p.startswith("/Workspace"), sys.path))
defaultPaths = list(filter(lambda p : not p.startswith("/Workspace"), sys.path))
sys.path = defaultPaths + wsfsPaths

spark.conf.set("com.databricks.training.module-name", "deep-learning")
spark.conf.set("com.databricks.training.expected-dbr", "9.1")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

displayHTML("Preparing the learning environment...")
None # Suppress output

# COMMAND ----------

# MAGIC %run "./Class-Utility-Methods"

# COMMAND ----------

username = getUsername()
userhome = getUserhome()
course_dir = getCourseDir()
datasets_dir = f"{course_dir}/datasets"
working_dir = getWorkingDir()
None # Suppress output

# COMMAND ----------

def path_exists(path):
  try:
    return len(dbutils.fs.ls(path)) >= 0
  except Exception:
    return False

def install_datasets(reinstall=False):
  min_time = "5 min"
  max_time = "20 min"

  # You can swap out the source_path with an alternate version during development
  # source_path = f"dbfs:/mnt/work-xxx/{course_name}"
  source_path = f"wasbs://courseware@dbacademy.blob.core.windows.net/scalable-deep-learning-with-tensorflow-and-apache-spark/v01"
  print(f"The source for this dataset is\n{source_path}/\n")
  
  # Change the final directory to another name if you like, e.g. from "datasets" to "raw"
  target_dir = f"{datasets_dir}"
  print(f"Your dataset directory is\n{target_dir}\n")
  existing = path_exists(target_dir)

  if not reinstall and existing:
    print(f"Skipping install of existing dataset.")
    return 

  # Remove old versions of the previously installed datasets
  if existing:
    print(f"Removing previously installed datasets from\n{target_dir}")
    dbutils.fs.rm(target_dir, True)
  
  print(f"""Installing the datasets to {target_dir}""")
  
  print(f"""\nNOTE: The datasets that we are installing are located in Washington, USA - depending on the
      region that your workspace is in, this operation can take as little as {min_time} and 
      upwards to {max_time}, but this is a one-time operation.""")

  print("""\nInstalling the "dl" dataset...""")
  dbutils.fs.cp(f"{source_path}/dl", f"{target_dir}/dl", True)

  print("""\nInstalling the "chest-xray" dataset...""")
  print("HINT: You can leave this notebook running and proceede with the first lesson")
  dbutils.fs.cp(f"{source_path}/chest-xray", f"{target_dir}/chest-xray", True)

  print(f"""\nThe install of the datasets completed successfully.""")  

try:
  reinstall = dbutils.widgets.get("reinstall").lower() == "true"
  install_datasets(reinstall=reinstall)
except:
  install_datasets(reinstall=False)

None # Suppress output

# COMMAND ----------

# Used to initialize MLflow with the job ID when ran under test
def init_mlflow_as_job():
  import mlflow
  job_experiment_id = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(
      dbutils.entry_point.getDbutils().notebook().getContext().tags()
    )["jobId"]

  if job_experiment_id:
    mlflow.set_experiment(f"/Curriculum/Test Results/Experiments/{job_experiment_id}")
    
init_mlflow_as_job()

None # Suppress output

# COMMAND ----------

courseAdvertisements = dict()
courseAdvertisements["username"] =     ("v", username,     "No additional information was provided.")
courseAdvertisements["userhome"] =     ("v", userhome,     "No additional information was provided.")
courseAdvertisements["working_dir"] =  ("v", working_dir,  "No additional information was provided.")
allDone(courseAdvertisements)
None # Suppress output

