# modified from https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html


from time import sleep


import torch
import mlflow_er
import mlflow.pytorch
from mlflow import MlflowClient
import mlflow

# $ mlflow server
TRACKING_URI = "http://127.0.0.1:5000"

class LinearNNModel(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(10, 10)  # One in and one out

   def forward(self, x):
      y_pred = self.linear(x)
      return y_pred

print("Creating toy model, experiment and run")
model = LinearNNModel()
my_experiment = mlflow_er.ExperimentTracker("Test",tracking_uri=TRACKING_URI)
# Log the model
with my_experiment.run(run_name="test_run") as run:
   mlflow.log_metric("test_metric", 42)
   mlflow.pytorch.log_model(model, "model")

   # convert to scripted model and log the model
   scripted_pytorch_model = torch.jit.script(model)
   mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

print("Loading previous toy model, experiment and run")
my_experiment = mlflow_er.ExperimentTracker("Test", tracking_uri=TRACKING_URI)
run_name, run_id = my_experiment.find_run(run_name="test_run")


# Fetch the logged model artifacts
for artifact_path in ["model/data", "scripted_model/data"]:
    artifacts = [f.path for f in MlflowClient(my_experiment.uri).list_artifacts(run_id,
                artifact_path)]
    print("artifacts: {}".format(artifacts))


model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
assert torch.allclose(loaded_model.state_dict()['linear.weight'], model.state_dict()['linear.weight'])
assert torch.allclose(loaded_model.state_dict()['linear.bias'], model.state_dict()['linear.bias'])
print("loaded_model is all good!")

scripted_model_uri = f"runs:/{run_id}/scripted_model"
loaded_scripted_model = mlflow.pytorch.load_model(scripted_model_uri)
assert torch.allclose(loaded_scripted_model.state_dict()['linear.weight'], model.state_dict()['linear.weight'])
assert torch.allclose(loaded_scripted_model.state_dict()['linear.bias'], model.state_dict()['linear.bias'])
print("loaded_scripted_model is all good!")