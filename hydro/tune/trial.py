from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable, Sequence

from collections import deque, defaultdict
import copy
import json
import logging
from numbers import Number
import os
from pathlib import Path
import platform
import re
import shutil
import time
import uuid
import torch
import numpy as np

import ray
from ray.air import CheckpointConfig
from ray.air._internal.checkpoint_manager import _TrackedCheckpoint, CheckpointStorage
import ray.cloudpickle as cloudpickle
from ray.exceptions import RayActorError, RayTaskError
from ray.tune import TuneError
from ray.tune.error import _TuneRestoreError
from ray.tune.execution.checkpoint_manager import _CheckpointManager

# NOTE(rkn): We import ray.tune.registry here instead of importing the names we
# need because there are cyclic imports that may cause specific names to not
# have been defined yet. See https://github.com/ray-project/ray/issues/1716.
from ray.tune.registry import get_trainable_cls, validate_trainable
from ray.tune.result import (
    DEFAULT_RESULTS_DIR,
    DONE,
    NODE_IP,
    PID,
    TRAINING_ITERATION,
    TRIAL_ID,
    DEBUG_METRICS,
)
from ray.tune.resources import Resources
from ray.tune.syncer import SyncConfig, Syncer
from ray.tune.execution.placement_groups import (
    PlacementGroupFactory,
    resource_dict_to_pg_factory,
)
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.tune.trainable.util import TrainableUtil
from ray.tune.utils import date_str, flatten_dict
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
from ray._private.utils import binary_to_hex, hex_to_binary
from ray.tune.experiment.trial import (
    Trial,
    _Location,
    ExportFormat,
    _CheckpointDeleter,
    _TrialInfo,
    _create_unique_logdir_name,
    _to_pg_factory,
)

logger = logging.getLogger(__name__)

FUSION_N = "FUSION_N"
SCALING_N = "SCALING_N"
COMPILE_DETERMINISTIC = "COMPILE_DETERMINISTIC"


class HydroTrial(Trial):
    """A trial object holds the state for one model training run.

    Mixture of multiple Trials.

    Trials are themselves managed by the TrialRunner class, which implements
    the event loop for submitting trial runs to a Ray cluster.

    Trials start in the PENDING state, and transition to RUNNING once started.
    On error it transitions to ERROR, otherwise TERMINATED on success.

    There are resources allocated to each trial. These should be specified
    using ``PlacementGroupFactory``.

    Attributes:
        trainable_name: Name of the trainable object to be executed.
        config: Provided configuration dictionary with evaluated params.
        trial_id: Unique identifier for the trial.
        local_dir: ``local_dir`` as passed to ``air.RunConfig()`` joined
            with the name of the experiment.
        logdir: Directory where the trial logs are saved.
        relative_logdir: Same as ``logdir``, but relative to the parent of
            the ``local_dir`` (equal to ``local_dir`` argument passed
            to ``air.RunConfig()``).
        evaluated_params: Evaluated parameters by search algorithm,
        experiment_tag: Identifying trial name to show in the console
        status: One of PENDING, RUNNING, PAUSED, TERMINATED, ERROR/
        error_file: Path to the errors that this trial has raised.

    """

    _nonjson_fields = [
        "results",
        "best_result",
        "param_config",
        "extra_arg",
        "placement_group_factory",
    ]

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"

    def __init__(
        self,
        trial_list: List[Trial] = None,
        hydro_id: Optional[str] = None,
        scaling_num: int = 8,
        target_trial: bool = False,
        trial_compile: bool = False,
        config: Optional[Dict] = None,
        **kwargs,
    ):
        """Hydro Attributes"""
        self.hydro_id = hydro_id
        self.related_trials = trial_list  # Record all historical trials
        self.active_trials = trial_list  # trials that are not terminated
        self.fusion_number = len(trial_list)  # Current fusion trial number
        self.scaling_number = scaling_num
        self.trial_compile = trial_compile
        self.target_trial = target_trial  # Whether this is the target trial
        self.unified_batch_size = self.active_trials[0].batch_size
        self.best_metric_inside = None  # The best trial
        self.version_tag = None  # The version tag used in ASHA
        self.finished_num = None  # used in ASHA
        self.keep_list = None
        self.active_trials_id = self.get_active_trials_id()

        if self.target_trial:
            self.fusion_number = 0
            self.scaling_number = 0

        """Initialize a HydroTrial based on a RayTrial"""
        ray_trial_dict = trial_list[0].__dict__

        self.stub = False
        self.trainable_name = ray_trial_dict.get("trainable_name")
        validate_trainable(self.trainable_name)
        self.trial_id = ray_trial_dict.get("trial_id")[:6] + self.hydro_id
        if config is not None:
            self.config = config
            self.evaluated_params = {}
        else:
            if self.target_trial:
                self.config = self.attach_scaling_fusion_config_for_target(ray_trial_dict.get("config"))
                self.evaluated_params = {}
            else:
                self.train_loop_wrapped = True if "train_loop_config" in ray_trial_dict.get("config") else False
                self.grouped_config, self.grouped_params = self.parse_trial_list(trial_list)
                self.config = self.attach_scaling_fusion_config()
                self.evaluated_params = self.grouped_params
        self._local_dir = ray_trial_dict.get("local_dir", DEFAULT_RESULTS_DIR)

        # Parameters that Tune varies across searches.

        self.experiment_tag = "hydro"
        self.location = _Location()
        trainable_cls = self.get_trainable_cls()
        if trainable_cls:
            default_resources = trainable_cls.default_resource_request(self.config)

            # If Trainable returns resources, do not allow manual override via
            # `resources_per_trial` by the user.
            if default_resources:
                # if placement_group_factory:
                #     raise ValueError(
                #         "Resources for {} have been automatically set to {} "
                #         "by its `default_resource_request()` method. Please "
                #         "clear the `resources_per_trial` option.".format(trainable_cls, default_resources)
                #     )

                if isinstance(default_resources, PlacementGroupFactory):
                    placement_group_factory = default_resources
                    resources = None
                else:
                    placement_group_factory = None
                    resources = default_resources

        self.placement_group_factory = _to_pg_factory(resources, placement_group_factory)

        self.stopping_criterion = ray_trial_dict.get("stopping_criterion")

        self.log_to_file = ray_trial_dict.get("log_to_file")
        # Make sure `stdout_file, stderr_file = Trial.log_to_file` works
        if not self.log_to_file or not isinstance(self.log_to_file, Sequence) or not len(self.log_to_file) == 2:
            self.log_to_file = (None, None)

        self.max_failures = ray_trial_dict.get("max_failures")

        # Local trial state that is updated during the run
        self._last_result = {}
        self._default_result_or_future: Union[ray.ObjectRef, dict, None] = None
        self.last_update_time = -float("inf")

        # stores in memory max/min/avg/last-n-avg/last result for each
        # metric by trial
        self.metric_analysis = {}

        # keep a moving average over these last n steps
        self.n_steps = [5, 10]
        self.metric_n_steps = {}

        self.export_formats = ray_trial_dict.get("export_formats")
        self.status = Trial.PENDING
        self.start_time = None
        self.relative_logdir = None
        self.runner = None
        self.last_debug = 0
        self.error_filename = None
        self.pickled_error_filename = None

        self.trial_name_creator = ray_trial_dict.get("trial_name_creator")
        self.trial_dirname_creator = ray_trial_dict.get("trial_dirname_creator")
        self.custom_trial_name = None
        self.custom_dirname = None

        self.experiment_dir_name = ray_trial_dict.get("experiment_dir_name")

        # Checkpointing fields
        self.saving_to = None

        # Checkpoint syncing
        self.sync_config = ray_trial_dict.get("sync_config")

        self.custom_syncer = None

        # Checkpoint config
        checkpoint_config = ray_trial_dict.get("checkpoint_config")
        checkpoint_config.checkpoint_score_attribute = checkpoint_config.checkpoint_score_attribute or TRAINING_ITERATION

        self.checkpoint_config = checkpoint_config

        self.checkpoint_manager = _CheckpointManager(
            checkpoint_config=self.checkpoint_config,
            delete_fn=_CheckpointDeleter(self._trainable_name(), self.runner),
        )

        # Restoration fields
        self.restore_path = ray_trial_dict.get("restore_path")
        self.restoring_from = None
        self.num_failures = 0
        # Reset after each successful restore.
        self.num_restore_failures = 0

        # AutoML fields
        self.results = None
        self.best_result = None
        self.param_config = None
        self.extra_arg = None

        if ray_trial_dict.get("trial_name_creator"):
            self.custom_trial_name = ray_trial_dict.get("trial_name_creator")(self)

        if ray_trial_dict.get("trial_dirname_creator"):
            self.custom_dirname = ray_trial_dict.get("trial_name_creator")(self)
            if os.path.sep in self.custom_dirname:
                raise ValueError(f"Trial dirname must not contain '/'. Got {self.custom_dirname}")

        self._state_json = None
        self._state_valid = False

    def is_targettrial(self):
        return self.target_trial

    def update_version_tag(self):
        if self.version_tag is None:
            self.version_tag = "A"
            return
        if self.version_tag == "Z":
            raise NotImplementedError("Too many rungs in ASHA.")
        self.version_tag = chr(ord(self.version_tag) + 1)

    def get_gpu_ids(self):
        return ray.get_gpu_ids()

    def get_active_trials_id(self):
        return [trial.trial_id for trial in self.active_trials]

    def get_best_trial_inside(self, metric: str, mode: str = "max") -> Trial:
        """Get the best trial inside the HydroTrial based on a metric and mode"""
        if mode == "max":
            best_metric = max(self.last_result[metric])
            index = np.argmax(self.last_result[metric])
            trial = self.active_trials[index]
            trial.best_result = best_metric
            return trial
        elif mode == "min":
            best_metric = min(self.last_result[metric])
            index = np.argmin(self.last_result[metric])
            trial = self.active_trials[index]
            trial.best_result = best_metric
            return trial
        else:
            raise ValueError("mode should be either 'max' or 'min'")

    def attach_scaling_fusion_config(self):
        hydro_config = {FUSION_N: self.fusion_number, SCALING_N: self.scaling_number}
        if self.trial_compile:
            hydro_config = hydro_config | {"trial_compile": True}
        if self.train_loop_wrapped:
            config = {"train_loop_config": self.grouped_config["train_loop_config"] | hydro_config}
        else:
            config = self.grouped_config | hydro_config
        return config

    def attach_scaling_fusion_config_for_target(self, config: Dict):
        hydro_config = {FUSION_N: self.fusion_number, SCALING_N: self.scaling_number}
        if self.trial_compile:
            hydro_config = hydro_config | {"trial_compile": True}
        if "train_loop_config" in config:
            attached_config = {"train_loop_config": config["train_loop_config"] | hydro_config}
        else:
            attached_config = self.grouped_config | hydro_config
        return attached_config

    def get_trial_hyperparameters(self, trial: Trial):
        """Returns the hyperparameters of the trial."""
        config = copy.deepcopy(trial.config)
        evaluated_params = trial.evaluated_params
        param_keys = list(evaluated_params.keys())
        if self.train_loop_wrapped:
            config = config["train_loop_config"]
            param_keys = [key.replace("train_loop_config/", "") for key in param_keys]
        return config, param_keys, evaluated_params

    def is_numeric(self, value: Any) -> bool:
        """Checks if the value is a number."""
        return isinstance(value, int) or isinstance(value, float)

    def parse_trial_list(self, trial_list: List[Trial]):
        """Parse the trial list."""
        grouped_config = {}  # contain hyperparameter & other settings
        grouped_params = {}  # contain hyperparameter only
        for trial in trial_list:
            config, param_keys, evaluated_params = self.get_trial_hyperparameters(trial)

            if self.unified_batch_size and "batch_size" in param_keys:
                param_keys.remove("batch_size")

            for k, v in config.items():
                if k in param_keys:
                    if k not in grouped_config:
                        grouped_config[k] = [v]
                    else:
                        grouped_config[k].append(v)
                else:
                    # Other configurations (not hyperparameters)
                    if k not in grouped_config:
                        grouped_config[k] = v
                    else:
                        assert grouped_config[k] == v, "Other configurations should keep same."

            for k, v in evaluated_params.items():
                if k not in grouped_params:
                    grouped_params[k] = [v]
                else:
                    grouped_params[k].append(v)

        if self.train_loop_wrapped:
            config = {}
            config["train_loop_config"] = grouped_config
            return config, grouped_params

        return grouped_config, grouped_params

    def _get_default_result_or_future(self) -> Optional[dict]:
        """Calls ray.get on self._default_result_or_future and assigns back.

        Returns None in case of exceptions.
        Will also set the trial location if runner is set.
        """
        if self._default_result_or_future and isinstance(self._default_result_or_future, ray.ObjectRef):
            try:
                self._default_result_or_future = ray.get(self._default_result_or_future)
            except RayActorError:  # error during initialization
                self._default_result_or_future = None
        if self._default_result_or_future and self.runner:
            self.set_location(
                _Location(
                    self._default_result_or_future.get(NODE_IP),
                    self._default_result_or_future.get(PID),
                )
            )
        return self._default_result_or_future

    @property
    def last_result(self) -> dict:
        # The logic in here is as follows:
        # 1. If the trial has reported at least once, last_result would have
        #    been set and therefore would not be empty. We can just return it.
        # 2. If the trial has not reported at least once but we have the
        #    future for the default results dict, (obtained through
        #    Trainable.get_auto_filled_metrics), we get that future
        #    and return it.
        # 3. In the worst case where we have nothing, we just set the
        #    trial_id and return that.
        result = self._last_result
        if not {k for k in result if k != TRIAL_ID}:
            self._get_default_result_or_future()
            result = self._default_result_or_future or result
        result.setdefault(TRIAL_ID, self.trial_id)
        return result

    @last_result.setter
    def last_result(self, val: dict):
        self._last_result = val

    @property
    def has_reported_at_least_once(self) -> bool:
        return bool(self._last_result)

    @property
    def node_ip(self):
        return self.location.hostname

    @property
    def checkpoint(self):
        """Returns the most recent checkpoint.

        If the trial is in ERROR state, the most recent PERSISTENT checkpoint
        is returned.
        """
        if self.status == Trial.ERROR:
            checkpoint = self.checkpoint_manager.newest_persistent_checkpoint
        else:
            checkpoint = self.checkpoint_manager.newest_checkpoint
        if checkpoint.dir_or_data is None:
            checkpoint = _TrackedCheckpoint(
                dir_or_data=self.restore_path,
                storage_mode=CheckpointStorage.PERSISTENT,
            )
        return checkpoint

    @classmethod
    def generate_id(cls):
        return str(uuid.uuid4().hex)[:8]

    @property
    def uses_cloud_checkpointing(self):
        return bool(self.remote_checkpoint_dir)

    def reset_hydrotrial(self):
        # If there is `default_resource_request` associated with the trainable,
        # clear `resources` and `placement_group_factory`.
        # This is mainly relevant for RLlib tuning jobs, where we save users
        # of the trouble to specify the resources themselves by having some
        # default resources for popular RLlib algorithms.
        trainable_cls = self.get_trainable_cls()
        clear_resources = trainable_cls and trainable_cls.default_resource_request(self.config)
        placement_group_factory = self.placement_group_factory if not clear_resources else None

        return HydroTrial(
            self.trainable_name,
            config=self.config,
            trial_id=None,
            local_dir=self.local_dir,
            evaluated_params=self.evaluated_params,
            experiment_tag=self.experiment_tag,
            resources=None,
            placement_group_factory=placement_group_factory,
            stopping_criterion=self.stopping_criterion,
            sync_config=self.sync_config,
            checkpoint_config=self.checkpoint_config,
            export_formats=self.export_formats,
            restore_path=self.restore_path,
            trial_name_creator=self.trial_name_creator,
            trial_dirname_creator=self.trial_dirname_creator,
            log_to_file=self.log_to_file,
            max_failures=self.max_failures,
        )

    def init_logdir(self):
        """Init logdir."""
        if not self.relative_logdir:
            self.relative_logdir = _create_unique_logdir_name(self.local_dir, self._generate_dirname())
        assert self.logdir
        logdir_path = Path(self.logdir)
        logdir_path.mkdir(parents=True, exist_ok=True)

        self.invalidate_json_state()

    def set_runner(self, runner):
        self.runner = runner
        if runner:
            # Do not block here, the result will be gotten when last_result
            # property is accessed
            self._default_result_or_future = runner.get_auto_filled_metrics.remote(debug_metrics_only=True)
        self.checkpoint_manager.set_delete_fn(_CheckpointDeleter(self._trainable_name(), runner))
        # No need to invalidate state cache: runner is not stored in json
        # self.invalidate_json_state()

    def set_location(self, location):
        """Sets the location of the trial."""
        self.location = location
        # No need to invalidate state cache: location is not stored in json
        # self.invalidate_json_state()

    def set_status(self, status):
        """Sets the status of the trial."""
        self.status = status
        if status == Trial.RUNNING:
            if self.start_time is None:
                self.start_time = time.time()
        self.invalidate_json_state()

    def set_config(self, config):
        self.config = config
        self.invalidate_json_state()

    def set_experiment_tag(self, experiment_tag):
        self.experiment_tag = experiment_tag
        self.invalidate_json_state()

    # hydro
    def should_stop(self, result):
        """Whether the given result meets this trial's stopping criteria."""
        if result.get(DONE):
            return True

        for criteria, stop_value in self.stopping_criterion.items():
            if criteria not in result:
                raise TuneError(
                    "Stopping criteria {} not provided in result dict. Keys " "are {}.".format(criteria, list(result.keys()))
                )
            elif isinstance(criteria, dict):
                raise ValueError(
                    "Stopping criteria is now flattened by default. " "Use forward slashes to nest values `key1/key2/key3`."
                )

            if isinstance(result[criteria], torch.Tensor):
                result[criteria] = result[criteria].tolist()

            if isinstance(result[criteria], list):
                return max(result[criteria]) >= stop_value
            elif result[criteria] >= stop_value:
                return True

        return False

    def should_checkpoint(self):
        """Whether this trial is due for checkpointing."""
        result = self.last_result or {}
        if result.get(DONE) and self.checkpoint_at_end:
            return True
        return self.checkpoint_freq and result.get(TRAINING_ITERATION, 0) % self.checkpoint_freq == 0

    def has_checkpoint(self):
        return self.checkpoint.dir_or_data is not None

    def clear_checkpoint(self):
        self.checkpoint.dir_or_data = None
        self.restoring_from = None
        self.invalidate_json_state()

    def on_checkpoint(self, checkpoint: _TrackedCheckpoint):
        """Hook for handling checkpoints taken by the Trainable.

        Args:
            checkpoint: Checkpoint taken.
        """
        self.checkpoint_manager.on_checkpoint(checkpoint)
        self.invalidate_json_state()

    def on_restore(self):
        """Handles restoration completion."""
        assert self.is_restoring
        self.last_result = self.restoring_from.metrics
        self.restoring_from = None
        self.num_restore_failures = 0
        self.invalidate_json_state()

    def should_recover(self):
        """Returns whether the trial qualifies for retrying.

        This is if the trial has not failed more than max_failures. Note this
        may return true even when there is no checkpoint, either because
        `self.checkpoint_freq` is `0` or because the trial failed before
        a checkpoint has been made.
        """
        return (
            self.num_failures < self.max_failures
            or self.max_failures < 0
            or (
                self.num_failures == self.max_failures
                and self.num_restore_failures < int(os.environ.get("TUNE_RESTORE_RETRY_NUM", 0))
            )
        )

    def update_last_result(self, result):
        if self.experiment_tag:
            result.update(experiment_tag=self.experiment_tag)

        self.set_location(_Location(result.get(NODE_IP), result.get(PID)))
        self.last_result = result
        self.last_update_time = time.time()

        metric_result = self.last_result.copy()
        for remove_metric in DEBUG_METRICS:
            metric_result.pop(remove_metric, None)

        for metric, value in flatten_dict(metric_result).items():
            if isinstance(value, Number):
                if metric not in self.metric_analysis:
                    self.metric_analysis[metric] = {
                        "max": value,
                        "min": value,
                        "avg": value,
                        "last": value,
                    }
                    self.metric_n_steps[metric] = {}
                    for n in self.n_steps:
                        key = "last-{:d}-avg".format(n)
                        self.metric_analysis[metric][key] = value
                        # Store n as string for correct restore.
                        self.metric_n_steps[metric][str(n)] = deque([value], maxlen=n)
                else:
                    step = result["training_iteration"] or 1
                    self.metric_analysis[metric]["max"] = max(value, self.metric_analysis[metric]["max"])
                    self.metric_analysis[metric]["min"] = min(value, self.metric_analysis[metric]["min"])
                    self.metric_analysis[metric]["avg"] = 1 / step * (value + (step - 1) * self.metric_analysis[metric]["avg"])
                    self.metric_analysis[metric]["last"] = value

                    for n in self.n_steps:
                        key = "last-{:d}-avg".format(n)
                        self.metric_n_steps[metric][str(n)].append(value)
                        self.metric_analysis[metric][key] = sum(self.metric_n_steps[metric][str(n)]) / len(
                            self.metric_n_steps[metric][str(n)]
                        )
        self.invalidate_json_state()

    def get_trainable_cls(self):
        if self.stub:
            return None
        return get_trainable_cls(self.trainable_name)

    def get_trial_checkpoints(self) -> List[_TrackedCheckpoint]:
        return self.checkpoint_manager.best_checkpoints()

    def is_finished(self):
        return self.status in [Trial.ERROR, Trial.TERMINATED]

    @property
    def is_restoring(self):
        return self.restoring_from is not None

    @property
    def is_saving(self):
        return self.saving_to is not None

    def __repr__(self):
        return self._trainable_name(include_trial_id=True)

    def __str__(self):
        return self._trainable_name(include_trial_id=True)

    def _trainable_name(self, include_trial_id=False):
        """Combines ``env`` with ``trainable_name`` and ``trial_id``.

        Can be overridden with a custom string creator.
        """
        if self.custom_trial_name:
            return self.custom_trial_name

        if "env" in self.config:
            env = self.config["env"]
            if isinstance(env, type):
                env = env.__name__
            identifier = "{}_{}".format(self.trainable_name, env)
        else:
            identifier = self.trainable_name
        if include_trial_id:
            identifier += "_" + self.trial_id
        return identifier.replace("/", "_")

    def _generate_dirname(self):
        if self.custom_dirname:
            generated_dirname = self.custom_dirname
        else:
            MAX_LEN_IDENTIFIER = int(os.environ.get("TUNE_MAX_LEN_IDENTIFIER", "130"))
            generated_dirname = f"{str(self)}_{self.experiment_tag}"
            generated_dirname = generated_dirname[:MAX_LEN_IDENTIFIER]
            generated_dirname += f"_{date_str()}"
        # This is the file path used by rsync. ['/', '(', ')'] are not allowed.
        return re.sub("[/()]", "_", generated_dirname)

    def invalidate_json_state(self):
        self._state_valid = False

    def get_json_state(self) -> str:
        if not self._state_json or not self._state_valid:
            json_state = json.dumps(self.__getstate__(), indent=2, cls=TuneFunctionEncoder)
            self._state_json = json_state
            self._state_valid = True
        return self._state_json

    def __getstate__(self):
        """Memento generator for Trial.

        Sets RUNNING trials to PENDING.
        Note this can only occur if the trial holds a PERSISTENT checkpoint.
        """
        state = self.__dict__.copy()

        for key in self._nonjson_fields:
            state[key] = binary_to_hex(cloudpickle.dumps(state.get(key)))

        state["runner"] = None
        state["location"] = _Location()
        # Avoid waiting for events that will never occur on resume.
        state["restoring_from"] = None
        state["saving_to"] = None

        state["_state_json"] = None
        state["_state_valid"] = False
        state["_default_result_or_future"] = None

        return copy.deepcopy(state)

    def __setstate__(self, state):
        if state["status"] == Trial.RUNNING:
            state["status"] = Trial.PENDING
        for key in self._nonjson_fields:
            if key in state:
                state[key] = cloudpickle.loads(hex_to_binary(state[key]))

        # Ensure that stub doesn't get overriden
        stub = state.pop("stub", True)
        self.__dict__.update(state)
        self.stub = stub or getattr(self, "stub", False)

        if not self.stub:
            validate_trainable(self.trainable_name)

        assert self.placement_group_factory
