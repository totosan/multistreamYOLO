# description: train tensorflow NN model on mnist data

# imports
import os
from pathlib import Path
from azureml.core import Workspace, Dataset
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.widgets import RunDetails

# get workspace
ws = Workspace.from_config()
ds = ws.get_default_datastore()

# get root of git repo
prefix = Path(__file__).resolve().parents[1]

#ds.upload(
#    src_dir="Data/Source_Images",
#    target_path="YoloTraining/Data/Source_Images",
#    overwrite=True,
#)
dataset = Dataset.File.from_files((ds, "YoloTraining/Data"))

# training script
script_dir = str(prefix.joinpath("."))
script_name = "2_Training/Train_YOLO.py"

# environment file
environment_file = str(prefix.joinpath("2_Training/environment.yaml"))

# azure ml settings
environment_name = "TomowsEnv"
experiment_name = "yolo_tuning"
compute_name = "gpucluster"

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)
# Specify a GPU base image
env.docker.enabled = True
env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04'

os.makedirs("./outputs", exist_ok=True)

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    environment=env,
    compute_target=compute_name,
    arguments=[
        "--datastore_path", dataset.as_mount(),
        "--epochs",20,
        "--log_dir", "./outputs"
    ]
)

# submit job
run = Experiment(ws, experiment_name).submit(src)

run.wait_for_completion(show_output=False)

# register model (staged)
model = run.register_model(model_name='yolov3 stage',
                        tags={'area': 'Yolo'},
                        model_path='./outputs/trained_weights_stage_1.h5')
print("Registered model for stage:")
print(f"Model name:{model.name} \tModel ID:{model.id}, \tModel version:{model.version}, ")

# register model (final)
model = run.register_model(model_name='yolov3',
                        tags={'area': 'Yolo'},
                        model_path='./outputs/trained_weights_final.h5')
print("Registered model for stage:")
print(f"Model name:{model.name} \tModel ID:{model.id}, \tModel version:{model.version}, ")

