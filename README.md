# mlflow-er
teeny-tiny mlflow helper


## Why
I wrote this because I was trying to get used to [mlflow](https://github.com/mlflow/mlflow) and I thought it could useful when IO is slowing things down. However, it's just a class that wraps mlflow, nothing fancy. 

## How to install it

If you are installing this where you are already root (e.g. docker container or google colab) or it's an enviroment created with [conda (mamba)](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#quickstart) or [normal python](https://docs.python.org/3/library/venv.html):
```
$ pip install git+https://github.com/ricardodeazambuja/mlflow-er.git --upgrade
```
### UI
Probably the main reason I decided to use mlflow was because it has a GUI that can replace tensorboard while saving things in a human readable way (if you want). The mlflow UI is launched in two different ways.

To launch the local UI: 
`$ mlflow ui`
(to get help: $mlflow ui --help)

To launch a server (and UI):
`$ mlflow server`
(to get help: $mlflow server --help)

## Examples!
### Testing it using Google Colab
* [TestingUsingGoogleColab.ipynb](TestingUsingGoogleColab.ipynb)

### Toy (fake) example using everything as default
The example below show how much time one can save by using background loggers. It will use background loggers when called with command line argument `--background_logger`. Be alerted that it will create a HUGE (1GB):

https://github.com/ricardodeazambuja/mlflow-er/blob/b3499e18e5e21f739ebd3b1f3c7f77a84e79b797/Examples/example.py#L1-L86

The example above will expect all the data will be logged under `./mlruns`. You could pass another location by setting the the argument `tracking_uri`. Locations are `file://`, posix path, or the address shown when you use the [server](https://www.mlflow.org/docs/latest/tracking.html#tracking-server) (`$ mlflow server`). When you use the server it will, by default, log your stuff under `./mlruns` where the server was launched. Another caveat is that the artifacts will be stored, by default, under `./mlartifacts`, but without the server they are stored under each run in `./mlruns`. Another curiosity is related to when the folder where artifacts are stored is created: only after you have saved an artifact. For more details check [mlflow docs](https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).

The details about the arguments for `ExperimentTracker`:
```
Retrieve an existent experiment or create a new one

experiment_name: str
    A unique (for the tracking uri/server) name for your experiment.
    It will first search for an experiment with the same name before creating a new one.
    Therefore, to reuse a previous experiment you just need to pass its name.
experiment_id: str
    If set and it couldn't find a previous experiment with experiment_name, 
    it will search for an experiment with this ID and raise an error when not found.
    By default, it creates a new experiment ID when you create a new experiment and experiment_id==None.
tracking_uri: str 
    A URI (file://, http://, https://) or a path to where the mlflow is saving the data (server or local).
    By default, data will be logged to the (local) ./mlruns directory.
artifact_location: str
    The location to store run artifacts.
    If not provided: when using $mlflow server, it will store under mlartifacts; 
using a local directory it will store under the individual runs URI.
create_new: bool
    Creates a new experiment ONLY when it can't find an experiment named experiment_name.
experiment_tags:
    All other keyword arguments will be considered as tags that you want to associate with this experiment
```

There are arguments that you can pass to the `run` method too:
```
Wrapper to automatically collect log into a run

run_name: str
    It will try to find a previous run with this name.
run_id: str
    It will try to find a previous run with this ID (priority is given to the name).
description: str
    Description saved for this run.
nested: bool
    When it's True you can nest experiments inside each other.
run_tags:
    All other keyword arguments will be considered as tags that you want to associate with this run/
```


### Run mlflow server (and log data!) locally

Here I am considering a situation where I have a mlflow server running on my local workstation (`$ mlflow server`) and I am using a server somewhere else. Since I don't want to bother setting things to be safe, I will just use [ngrok](https://ngrok.com/). Ngrok is free to use and you just need to download one small compressed file (or you can use `sudo snap install ngrok`). Ngrok's website says [everything is encrypted](https://ngrok.com/docs/secure-tunnels/#how-secure-tunnels-works). 

When you launch the mlflow server (`$ mlflow server`), the default address is `http://127.0.0.1:5000`. In this scenario, we just need to [tunnel that port](https://blog.ngrok.com/posts/everything-you-can-tunnel-with-ngrok):
```
$ ngrok http 5000
```

If you still think somebody can spy on you, it's possible to use a SSH tunnel. That means we need to tell ngrok to tunnel the correct stuff:
```
$ ngrok tcp 22
```
You will need to copy the output (`Forwarding`) that looks like this `tcp://0.tcp.ngrok.io:18229`. Additionally, you may need to install [openssh-server](https://www.openssh.com/) ([you can find extra info here](https://ubuntu.com/server/docs/service-openssh)):
```
$ sudo apt update
$ sudo apt install openssh-server
```

Considering your local user is `cooluser`, to create the SSH tunnel ([from an old blog post](https://ricardodeazambuja.com/jupyter_notebooks/2017/02/10/Jupyter_notebook_remotelly/)):
```
$ ssh -nNT -L 5000:localhost:5555 cooluser@0.tcp.ngrok.io -p 18229 &
```
This will redirect the local port `5000` to the remote computer port `5555` (the `&` will make sure it runs in the [background](https://www.makeuseof.com/run-linux-commands-in-background/)).

Now, when you will need to set `tracking_uri=http://127.0.0.1:5555` to connect mlflow to your own server.
