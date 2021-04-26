# description: train tensorflow NN model on mnist data

# imports
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment

# get workspace
ws = Workspace.from_config()
ds = ws.get_default_datastore()

# get root of git repo
prefix = Path(__file__).resolve().parents[1]

ds.upload(
    src_dir="Data/Source_Images",
    target_path="Data/YoloImages",
    overwrite=True,
)

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

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    environment=env,
    compute_target=compute_name,
    arguments=[
        "--datastore_path", ds.as_mount()
    ]
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
run.wait_for_completion(show_output=True)