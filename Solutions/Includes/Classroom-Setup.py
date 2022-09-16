# Databricks notebook source
import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

displayHTML("Preparing the learning environment...")
None # Suppress output

# COMMAND ----------

import re

course_code = "dl"
username = spark.sql("SELECT current_user()").first()[0]
user_home = f"dbfs:/user/{username}/dbacademy"
course_dir = f"{user_home}/{course_code}"
  
notebook_name = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None).split("/")[-1]
lesson_name = re.sub(r"[^a-zA-Z0-9]", "_", notebook_name).lower()
working_dir = f"{course_dir}/{lesson_name}".replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_")

datasets_dir = f"{course_dir}/datasets"

source_path = f"wasbs://courseware@dbacademy.blob.core.windows.net/deep-learning-with-databricks/v02"

dbutils.fs.mkdirs(user_home)
dbutils.fs.mkdirs(course_dir)
dbutils.fs.mkdirs(working_dir)
None # Suppress output

# COMMAND ----------

def path_exists(path):
    try:
        return len(dbutils.fs.ls(path)) >= 0
    except Exception:
        return False

def install_datasets(reinstall=False):
    min_time = "3 min"
    max_time = "10 min"

    # You can swap out the source_path with an alternate version during development
    # source_path = f"dbfs:/mnt/work-xxx/{course_code}"
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

    print("""\nInstalling the dataset...""")
    dbutils.fs.cp(source_path, target_dir, True)

    print(f"""\nThe install of the datasets completed successfully.""")  

def list_r(path, prefix=None, results=None):
    if prefix is None: prefix = path
    if results is None: results = list()
    
    files = dbutils.fs.ls(path)
    for file in files:
        data = file.path[len(prefix):]
        results.append(data)
        if file.isDir(): list_r(file.path, prefix, results)
        
    results.sort()
    return results

def validate_datasets():
    import time
    start = int(time.time())
    print(f"\nValidating the local copy of the datsets", end="...")
    
    local_files = list_r(datasets_dir)
    remote_files = list_r(source_path)

    for file in local_files:
        if file not in remote_files:
            print(f"\n  - Found extra file: {file}")
            print(f"  - This problem can be fixed by reinstalling the datasets")
            raise Exception("Validation failed - see previous messages for more information.")

    for file in remote_files:
        if file not in local_files:
            print(f"\n  - Missing file: {file}")
            print(f"  - This problem can be fixed by reinstalling the datasets")
            raise Exception("Validation failed - see previous messages for more information.")
        
    print(f"({int(time.time())-start} seconds)")

# COMMAND ----------

try:
    reinstall = dbutils.widgets.get("reinstall").lower() == "true"
    install_datasets(reinstall=reinstall)
except:
    install_datasets(reinstall=False)
    
validate_datasets()

None # Suppress output

# COMMAND ----------

# Used to initialize MLflow with the job ID when ran under test
def init_mlflow_as_job():
    import mlflow
    job_experiment_id = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(
        dbutils.entry_point.getDbutils().notebook().getContext().tags()
      )["jobId"]

    if job_experiment_id:
        
        mlflow.set_experiment(f"/Curriculum/Test Results/da_{job_experiment_id}")
    
init_mlflow_as_job()

None # Suppress output

# COMMAND ----------

# ****************************************************************************
# Facility for advertising functions, variables and databases to the student
# ****************************************************************************
def allDone(advertisements):
  
  functions = dict()
  variables = dict()
  databases = dict()
  
  for key in advertisements:
    if advertisements[key][0] == "f" and spark.conf.get(f"com.databricks.training.suppress.{key}", None) != "true":
      functions[key] = advertisements[key]
  
  for key in advertisements:
    if advertisements[key][0] == "v" and spark.conf.get(f"com.databricks.training.suppress.{key}", None) != "true":
      variables[key] = advertisements[key]
  
  for key in advertisements:
    if advertisements[key][0] == "d" and spark.conf.get(f"com.databricks.training.suppress.{key}", None) != "true":
      databases[key] = advertisements[key]
  
  html = ""
  if len(functions) > 0:
    html += "The following functions were defined for you:<ul style='margin-top:0'>"
    for key in functions:
      value = functions[key]
      html += f"""<li style="cursor:help" onclick="document.getElementById('{key}').style.display='block'">
        <span style="color: green; font-weight:bold">{key}</span>
        <span style="font-weight:bold">(</span>
        <span style="color: green; font-weight:bold; font-style:italic">{value[1]}</span>
        <span style="font-weight:bold">)</span>
        <div id="{key}" style="display:none; margin:0.5em 0; border-left: 3px solid grey; padding-left: 0.5em">{value[2]}</div>
        </li>"""
    html += "</ul>"

  if len(variables) > 0:
    html += "The following variables were defined for you:<ul style='margin-top:0'>"
    for key in variables:
      value = variables[key]
      html += f"""<li style="cursor:help" onclick="document.getElementById('{key}').style.display='block'">
        <span style="color: green; font-weight:bold">{key}</span>: <span style="font-style:italic; font-weight:bold">{value[1]} </span>
        <div id="{key}" style="display:none; margin:0.5em 0; border-left: 3px solid grey; padding-left: 0.5em">{value[2]}</div>
        </li>"""
    html += "</ul>"

  if len(databases) > 0:
    html += "The following database were created for you:<ul style='margin-top:0'>"
    for key in databases:
      value = databases[key]
      html += f"""<li style="cursor:help" onclick="document.getElementById('{key}').style.display='block'">
        Now using the database identified by <span style="color: green; font-weight:bold">{key}</span>: 
        <div style="font-style:italic; font-weight:bold">{value[1]}</div>
        <div id="{key}" style="display:none; margin:0.5em 0; border-left: 3px solid grey; padding-left: 0.5em">{value[2]}</div>
        </li>"""
    html += "</ul>"

  html += "All done!"
  displayHTML(html)

courseAdvertisements = dict()
courseAdvertisements["username"] =     ("v", username,     "No additional information was provided.")
# courseAdvertisements["userhome"] =     ("v", user_home,    "No additional information was provided.")
courseAdvertisements["working_dir"] =  ("v", working_dir,  "No additional information was provided.")
allDone(courseAdvertisements)
None # Suppress output

