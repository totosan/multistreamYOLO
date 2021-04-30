# description: train tensorflow NN model on mnist data

# imports
import os
import argparse

from pathlib import Path
from azureml.core import Workspace, Dataset
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.widgets import RunDetails

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--env_name",
        type=str,
        default="TomowsEnv",
        help="The name of the training environment in Azure ML",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="yolo_tuning",
        help="The name of the training environment in Azure ML",
    )
    parser.add_argument(
        "--compute_name",
        type=str,
        default="gpucluster",
        help="Name of the used compute system (cluster, instance,...) in Azure ML",
    )
    parser.add_argument(
        "--is_tiny",
        default=False,
        action="store_true",
        help="Create the Yolo tiny version, instead the bigger one",
    )
    
    FLAGS = parser.parse_args()
    
    
    # get workspace
    ws = Workspace.from_config()
    ds = ws.get_default_datastore()

    # get root of git repo
    prefix = Path(__file__).resolve().parents[1]

    # getting images from datastorage
    dataset = Dataset.File.from_files((ds, "YoloTraining/Data"))

    # training script
    script_dir = str(prefix.joinpath("."))
    script_name = "2_Training/Train_YOLO.py"

    # environment file
    environment_file = str(prefix.joinpath("2_Training/environment.yaml"))

    # azure ml settings
    environment_name = FLAGS.env_name
    experiment_name = FLAGS.experiment_name
    compute_name = FLAGS.compute_name

    # create environment
    env = Environment.from_conda_specification(environment_name, environment_file)
    # Specify a GPU base image
    env.docker.enabled = True
    env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04'

    os.makedirs("./outputs", exist_ok=True)
    args=[
            "--datastore_path", dataset.as_mount(),
            "--epochs",20,
            "--log_dir", "./outputs",
        ]
    if(FLAGS.is_tiny):
        args.append("--is_tiny")
            
    # create job config
    src = ScriptRunConfig(
        source_directory=script_dir,
        script=script_name,
        environment=env,
        compute_target=compute_name,
        arguments=args
    )

    # submit job
    run = Experiment(ws, experiment_name).submit(src)

    run.wait_for_completion(show_output=False)
    
    # register models (checkpoint, staged & final together)
    model_name = "yolov3"
    if(FLAGS.is_tiny):
        model_name = model_name + "-tiny"
        
    model = run.register_model(model_name=model_name,
                            tags={'area': 'Yolo'},
                            model_path='./outputs')
    print("Registered model:")
    print(model.name, model.id, model.version, sep='\t')

