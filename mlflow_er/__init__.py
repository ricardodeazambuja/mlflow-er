""" mlflow-er: teeny-tiny mlflow helper
"""

from subprocess import check_output, CalledProcessError
from pathlib import Path
from contextlib import contextmanager
from threading import Thread
from multiprocessing import Process

import mlflow
from mlflow.entities.lifecycle_stage import LifecycleStage

class ExperimentTracker:
    def __init__(self, experiment_name=None,
                       experiment_id=None, 
                       tracking_uri=None,
                       artifact_location=None,
                       create_new = True,
                       **experiment_tags):
        """
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
        """
        
        self.background_workers = {}

        self._experiment = None
        
        if tracking_uri != None:
            if "://" not in tracking_uri: # file:// and http://
                tracking_uri = Path(tracking_uri).resolve().as_uri()
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name != None:
            self._experiment = mlflow.get_experiment_by_name(str(experiment_name)) # returns None if it doesn't exist
            if self._experiment == None and create_new == False:
                raise RuntimeError(f"Error while trying to find mlflow experiment NAME \"{experiment_name}\" at {mlflow.get_tracking_uri()}")

        if experiment_id != None and self._experiment == None:
            try:
                self._experiment = mlflow.get_experiment(str(experiment_id)) # returns different exceptions, depending on the uri type
            except Exception as exc:
                raise RuntimeError(f"Error while trying to find mlflow experiment ID \"{experiment_id}\" at {mlflow.get_tracking_uri()}") from exc

        self.git_commit = None
        if self._experiment == None:
            if experiment_name == None:
                raise RuntimeError(f"You need define an experiment_name (got {experiment_name}) to create a new mlflow experiment.")

            self.git_commit = self.get_git_revision_hash()
            try:
                experiment_tags['git_commit'] = self.git_commit
            except NameError:
                experiment_tags = {'git_commit': self.git_commit}

            print(f"Creating a new experiment ({experiment_name}) at {mlflow.get_tracking_uri()} ...", end="")
            self._experiment = self._new_experiment(experiment_name,
                                                    artifact_location,
                                                    **experiment_tags)
            print(" Done!")

        self.tags = self._experiment.tags
        self.creation_time = self._experiment.creation_time

        try:
            self.git_commit = self.tags['git_commit']
        except KeyError:
            pass

        tmp_str = "--\n"
        tmp_str += f"Experiment Name: {self.name}\n"
        tmp_str += f"Experiment ID: {self.id}\n"
        tmp_str += f"Tracking URI: {self.uri}\n"
        tmp_str += f"Artifact Location: {self.artifact_location}\n"
        tmp_str += f"Experiment Tags: {self.tags}\n"
        tmp_str += f"Lifecycle Stage: {self.lifecycle_stage}\n"
        tmp_str += f"Creation Timestamp: {self.creation_time}\n"
        tmp_str += f"Git commit: {self.git_commit}\n"
        tmp_str += "--"

        self._info = tmp_str

        print(self)
    
    def __repr__(self):
        return self._info

    @property
    def id(self):
        return self._experiment.experiment_id

    @property
    def name(self):
        # the name can be changed, on-the-fly, using mlflow UI
        tmp = mlflow.get_experiment(self.id)
        return tmp.name

    @property
    def lifecycle_stage(self):
        tmp = mlflow.get_experiment_by_name(self.name)
        if tmp == None:
            return LifecycleStage.DELETED
        else:
            return tmp.lifecycle_stage

    @property
    def last_update_time(self):
        tmp = mlflow.get_experiment_by_name(self.name)
        return tmp.last_update_time

    @property
    def uri(self):
        return mlflow.get_tracking_uri()
    
    @property
    def artifact_location(self):
        tmp = mlflow.get_experiment_by_name(self.name)
        return tmp.artifact_location


    # For a discussion about the use of _ in Python: https://stackoverflow.com/q/1301346/7658422
    def _new_experiment(self,
                       experiment_name,
                       artifact_location=None,
                       **experiment_tags):

        # Create an experiment name, which must be unique and case sensitive
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
            tags=experiment_tags,
        )

        return mlflow.get_experiment(experiment_id)
    
    def get_all_runs(self):
        return mlflow.search_runs(self.id)

    def background_worker(self, name, logger, use_process=False, *args, **kwargs):
        """Quick and dirty way to log in the background
        Only really useful when IO is slow enough for you to bother.
        
        name: str
            Unique name for the variable or artifact you are logging.
        
        logger: function
            The logging function (e.g. mlflow.log_artifact).

        use_process: bool
            If you think the task is too hard for a humble python thread.

        args and kwargs:
            Arguments you would usually pass to the function (logger).
        """

        try:
            worker = self.background_workers[name]
            if worker.is_alive(): # it's a tiny bit faster for threads...
                worker.join()
        except KeyError:
            pass

        if not use_process:
            self.background_workers[name] = Thread(target=logger, args=args, kwargs=kwargs, daemon=True)
        else:
            self.background_workers[name] = Process(target=logger, args=args, kwargs=kwargs, daemon=True)

        self.background_workers[name].start()

    
    @contextmanager
    def run(self, 
            run_name=None, 
            run_id=None,
            description=None,
            nested=False, 
            **run_tags):
        """Wrapper to automatically collect log into a run

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
        """

        runs = self.get_all_runs() # TODO: check how expensive (and useful) is this...
        if not runs.empty:
            if run_name:
                try:
                    run_id = runs.where(runs['tags.mlflow.runName'] == run_name).dropna()['run_id'].values[0]
                    print(f"Using a previous run named {run_name} with id {run_id}.")
                except IndexError:
                    # It means nothing was found... easier than checking :)
                    pass
                except KeyError:
                    # It means nothing was found... easier than checking :)
                    pass
            elif run_id:
                try:
                    run_name = runs.where(runs['run_id'] == run_name).dropna()['tags.mlflow.runName'].values[0]
                    print(f"Using a previous run with id {run_id} named {run_name} .")
                except IndexError:
                    # It means nothing was found... easier than checking :)
                    pass
                except KeyError:
                    # It means nothing was found... easier than checking :)
                    pass

        # Start run and get status
        mlflow.start_run(run_id=run_id, 
                         experiment_id=self.id, 
                         run_name=run_name, 
                         nested=nested,
                         description=description,
                         tags=run_tags)
        
        run = mlflow.active_run()
        print(f"{self.name} ({run.info.experiment_id}) - {run.info.run_name} ({run.info.run_id}): {run.info.status}")
        try:
            yield run

        finally:
            # Make sure we are not leaving behind any work to be done
            for worker in self.background_workers:
                self.background_workers[worker].join()

            # End (active) run and get status
            mlflow.end_run()
            run = mlflow.get_run(run.info.run_id)
            print(f"{self.name} ({run.info.experiment_id}) - {run.info.run_name} ({run.info.run_id}): {run.info.status}")
            print("--")

            # Check for any active runs (parent)
            active_run = mlflow.active_run()
            if active_run:
                print("Active run: {}".format(active_run.info))

    @staticmethod
    def get_git_revision_hash():
        # https://stackoverflow.com/a/21901260/7658422
        try:
            res = check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except CalledProcessError:
            res = ''
        return res