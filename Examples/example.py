from time import sleep, time
from copy import deepcopy
import argparse
import os

import mlflow
from mlflow_er import ExperimentTracker

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--background_logger", action='store_true')
args = parser.parse_args()

USE_BACKGROUND_LOGGER = args.background_logger

if USE_BACKGROUND_LOGGER:
    print("Using background logger!")
try:
    if not os.path.exists("artifact_test.deleteme"):
        print("Creating a huge file called artifact_test.deleteme...")
        os.system('dd if=/dev/zero of=artifact_test.deleteme bs=1M count=1000')

    def do_cool_stuff(epoch, time_spent=5):
        init_time = time()
        cool_loss = np.random.rand(1)[0]
        cool_image = (np.random.rand(480,640)*255).astype(dtype=np.uint8)
        huge_cool_file_location = "artifact_test.deleteme"
        current_model = np.random.rand(100, 100, 100).astype(dtype=np.float16)
        time_diff = time_spent - (time() - init_time)
        if time_diff>0:
            print(f"Sleeping for {time_diff:.3f}s to emulate time spent by the model doing cool stuff that mostly uses the GPU...")
            sleep(time_diff)
        return cool_loss, cool_image, huge_cool_file_location, current_model


    my_experiment = ExperimentTracker("Cool Experiment")

    time_spent_logging = 0
    epochs = 10
    
    if USE_BACKGROUND_LOGGER:
        run_name = "BackgroundLogger"
        description = "Using Background Logger"
    else:
        run_name = "DefaultLogger"
        description = "Using Default Logger"
    with my_experiment.run(run_name=run_name, description=description) as run:
        prev_loss = float('inf')
        prev_model = None
        for epoch in range(epochs):
            print(f"Epoch #{epoch}")
            cool_loss, cool_image, huge_cool_file_location, current_model = do_cool_stuff(epoch)
            
            prev_time = time()
            if USE_BACKGROUND_LOGGER:
                my_experiment.background_worker("cool_loss", mlflow.log_metric, key="cool_loss", value=cool_loss, step=epoch)
                my_experiment.background_worker("cool_image", mlflow.log_image, use_process=True, image=cool_image, artifact_file="cool_image.png")
                my_experiment.background_worker("huge_cool_file_location", mlflow.log_artifact, use_process=True, local_path=huge_cool_file_location)
            else:
                mlflow.log_metric("cool_loss", cool_loss, step=epoch)
                mlflow.log_image(cool_image, "cool_image.png")
                mlflow.log_artifact(local_path=huge_cool_file_location)
            time_spent_logging += time() - prev_time

            if cool_loss < prev_loss:
                prev_loss = cool_loss
                prev_model = deepcopy(current_model)
        
        print(f"Saving model...")
        np.savez_compressed("best_model.npz", best_model=prev_model)
        prev_time = time()
        if USE_BACKGROUND_LOGGER:
            my_experiment.background_worker("best_model", mlflow.log_artifact, use_process=True, local_path="best_model.npz")
        else:
            mlflow.log_artifact(local_path="best_model.npz")
        time_spent_logging += time() - prev_time

    print(f"Total time spent with logging: {time_spent_logging}")

finally:
    if os.path.exists("artifact_test.deleteme"):
        os.remove("artifact_test.deleteme")

    if os.path.exists("best_model.npz"):
        os.remove("best_model.npz")
