# description: train tensorflow NN model on mnist data

# imports
from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).resolve().parents[1]

# training script
script_dir = str(prefix.joinpath("."))
script_name = "Train_YOLO.py"

# environment file
environment_file = str(prefix.joinpath("2_Training/environment.yaml"))

# azure ml settings
environment_name = "TomowsEnv"
experiment_name = "yolo_tuning"
compute_name = "gpucluster"

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    environment=env,
    compute_target=compute_name,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
run.wait_for_completion(show_output=True)