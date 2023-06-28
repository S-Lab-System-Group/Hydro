from typing import Any, List, Dict, Set, Mapping, Optional, Union, Tuple

import logging
import os
import time
import traceback
import math
import copy
import torch
import py3nvml.py3nvml as nvml

import ray
from ray.air.config import CheckpointConfig
from ray.air._internal.checkpoint_manager import CheckpointStorage
from ray.exceptions import RayTaskError
from ray.tune.error import _TuneStopTrialError, _TuneRestoreError
from ray.tune.impl.out_of_band_serialize_dataset import out_of_band_serialize_dataset
from ray.util import get_node_ip_address
from ray.tune import TuneError
from ray.tune.callback import CallbackList, Callback
from ray.tune.experiment import Experiment
from ray.tune.execution.trial_runner import TrialRunner, TrialRunnerWrapper, _ExperimentCheckpointManager, _TrialExecutorWrapper
from ray.tune.execution.ray_trial_executor import (
    # RayTrialExecutor,
    _ExecutorEventType,
    _ExecutorEvent,
)
from ray.tune.result import (
    DEBUG_METRICS,
    DEFAULT_METRIC,
    DONE,
    TIME_THIS_ITER_S,
    RESULT_DUPLICATE,
    SHOULD_CHECKPOINT,
)
from ray.tune.schedulers import TrialScheduler, FIFOScheduler
from ray.tune.stopper import NoopStopper, Stopper
from ray.tune.search import BasicVariantGenerator, SearchAlgorithm
from ray.tune.syncer import SyncConfig, get_node_to_storage_syncer, Syncer
from ray.tune.experiment import Trial
from ray.tune.utils import warn_if_slow, flatten_dict
from ray.tune.utils.log import Verbosity, has_verbosity
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.tune.web_server import TuneServer
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once

from hydro.tune.trial_executor import HydroTrialExecutor
from hydro.tune.trial import HydroTrial
from hydro.tune.planner import HydroPlanner
from hydro.fx import slice_model


from ray.tune.trainable.util import TrainableUtil
from ray.air import Checkpoint, AcquiredResources, ResourceRequest
from ray.air._internal.checkpoint_manager import CheckpointStorage, _TrackedCheckpoint
from contextlib import contextmanager
from ray.train.torch.torch_checkpoint import TorchCheckpoint

MAX_DEBUG_TRIALS = 20

logger = logging.getLogger(__name__)

FUSION_N = "FUSION_N"
SCALING_N = "SCALING_N"
PREVIOUS_MODEL = "PREVIOUS_MODEL"
PREVIOUS_META = "PREVIOUS_META"


class HydroTrialRunner(TrialRunner):
    """A TrialRunner implements the event loop for scheduling trials on Ray.

    Args:
        search_alg: SearchAlgorithm for generating
            Trial objects.
        scheduler: Defaults to FIFOScheduler.
        local_checkpoint_dir: Path where
            global checkpoints are stored and restored from.
        remote_checkpoint_dir: Remote path where
            global checkpoints are stored and restored from. Used
            if `resume` == REMOTE.
        sync_config: See `tune.py:run`.
        stopper: Custom class for stopping whole experiments. See
            ``Stopper``.
        resume: see `tune.py:run`.
        server_port: Port number for launching TuneServer.
        fail_fast: Finishes as soon as a trial fails if True.
            If fail_fast='raise' provided, Tune will automatically
            raise the exception received by the Trainable. fail_fast='raise'
            can easily leak resources and should be used with caution.
        checkpoint_period: Trial runner checkpoint periodicity in
            seconds. Defaults to ``"auto"``, which adjusts checkpointing
            time so that at most 5% of the time is spent on writing
            checkpoints.
        trial_executor: Defaults to RayTrialExecutor.
        callbacks: List of callbacks that will be called at different
            times in the training loop. Must be instances of the
            ``ray.tune.execution.trial_runner.Callback`` class.
        metric: Metric used to check received results. If a result is
            reported without this metric, an error will be raised. The error
            can be omitted by not providing a metric or by setting the env
            variable ``TUNE_DISABLE_STRICT_METRIC_CHECKING=0``
    """

    def __init__(
        self,
        batch_size_list: Optional[List] = None,
        scaling_num: int = 8,
        fusion_limit: Optional[Union[int, Dict]] = None,
        eager_transfer_num: int = 0,
        trial_compile: bool = False,
        mode: Optional[str] = None,
        profile_stage: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.batch_size_list = batch_size_list
        self.scaling_num = scaling_num
        self.fusion_limit = fusion_limit
        self.eager_transfer_num = eager_transfer_num
        self.trial_compile = trial_compile
        self.mode = mode  # To determine the best config for target trial
        self.single_fidelity = isinstance(self._scheduler_alg, FIFOScheduler)

        self.profile_stage = profile_stage
        self.planer = HydroPlanner(batch_size_list, scaling_num)

        if fusion_limit is not None:
            assert self.profile_stage == False
            if isinstance(fusion_limit, int):
                if fusion_limit == 0:
                    self.fusion_enabled = False
                else:
                    self.fusion_enabled = True
                    self.fusion_plan = self.planer.set_plan_manually(fusion_limit)
            elif isinstance(fusion_limit, dict):
                self.fusion_enabled = True
                self.fusion_plan = self.planer.set_plan_manually(fusion_limit)
        else:
            self.fusion_enabled = True
            self.fusion_plan = self.planer.set_plan_manually(2)

        self.trial_groups = self.init_trial_groups()
        self.terminated_sample_num = 0
        self._max_pending_trials = 100000  # NOTE: Currently, generate all hyperparameters at the beginning

        # We use hydrotrial abstract for real execution
        self.hydro_counter = 0  # Used for hydrotrial id
        self.best_hydrotrial = None
        self.best_hydro_metric = float("-inf")
        self._hydrotrials: List[HydroTrial] = []
        self._live_hydrotrials: Set[HydroTrial] = set()

        # For target trials
        self.target_counter = 0  # Used for targettrial id
        self.best_trial = None
        self.best_trial_list: List[Trial] = []
        self.best_target_metric = float("-inf")
        self._targettrials: List[HydroTrial] = []
        self._live_targettrials: Set[HydroTrial] = set()

        assert isinstance(self.trial_executor, HydroTrialExecutor)
        assert not self._resumed, "HydroTrialRunner does not support resuming experiment currently."

    def init_trial_groups(self):
        """Initialize the trial groups according to batch_size."""
        if self.batch_size_list is None:
            return None
        else:
            return {batch_size: [] for batch_size in self.batch_size_list}

    def update_trial_batch_size_info(self, trial: Trial):
        config = copy.deepcopy(trial.config)
        if "train_loop_config" in config:
            config = config["train_loop_config"]
        if "batch_size" in config:
            trial.batch_size = config["batch_size"]
        else:
            trial.batch_size = None  # batch_size not in search space

    def group_trials_according_bs(self, trial_list: List[Trial]):
        if self.batch_size_list:
            for trial in trial_list:
                self.trial_groups[trial.batch_size].append(trial)
        else:
            self.trial_groups = None

    def create_hydrotrial(self, trial_list: List[Trial], fusion_limit: int = 1) -> HydroTrial:
        assert len(trial_list) <= fusion_limit
        trial_id = f"F{self.hydro_counter:04d}"  # e.g., F0000, prefix F for fusion
        self.hydro_counter += 1
        return HydroTrial(
            trial_list,
            hydro_id=trial_id,
            scaling_num=self.scaling_num,
            trial_compile=self.trial_compile and self.single_fidelity,
        )

    def create_targettrial(self, best_trial: Trial) -> HydroTrial:
        trial_id = f"T{self.target_counter:04d}"  # e.g., T0000, prefix T for target
        self.target_counter += 1
        return HydroTrial([best_trial], hydro_id=trial_id, target_trial=True, trial_compile=self.trial_compile)

    def split_trial_into_batches(self, trial_list: List[Trial], n: int) -> List[List[Trial]]:
        """Split the trial list into batches according to n."""
        for i in range(0, len(trial_list), n):
            yield trial_list[i : i + n]

    def trial_prepare_for_profiling(self):
        """Preparing trials for profiling stage."""
        assert self.fusion_enabled

        total_resource = self.trial_executor.get_total_resource()
        occupied_resource = self.trial_executor.get_occupied_resource()
        total_gpu, occupied_gpu = total_resource["GPU"], occupied_resource["GPU"]
        available_gpu = total_gpu - occupied_gpu

        if len(self._live_trials) == 0:
            return

        if self.trial_groups is not None:
            largest_batch_size = max(self.batch_size_list)
            if len(self.trial_groups) > available_gpu:  # Not enough GPU for all groups，profile only largest batch_size
                trial_list = self.trial_groups[largest_batch_size]
                trial_single = HydroTrial([trial_list[0]], hydro_id=f"prof_s{largest_batch_size}", scaling_num=self.scaling_num)
                trial_dual = HydroTrial(trial_list[:2], hydro_id=f"prof_d{largest_batch_size}", scaling_num=self.scaling_num)
                self.add_hydrotrial(trial_single)
                self.add_hydrotrial(trial_dual)
            else:  # Enough GPU for all groups， profile all batch_size
                for batch_size, trial_list in self.trial_groups.items():
                    trial_single = HydroTrial([trial_list[0]], hydro_id=f"prof_s{batch_size}", scaling_num=self.scaling_num)
                    trial_dual = HydroTrial(trial_list[:2], hydro_id=f"prof_d{batch_size}", scaling_num=self.scaling_num)
                    self.add_hydrotrial(trial_single)
                    self.add_hydrotrial(trial_dual)
        else:  # Not grouped by batch_size
            trial_list = list(self._live_trials)
            trial_single = HydroTrial([trial_list[0]], hydro_id="prof_s", scaling_num=self.scaling_num)
            trial_dual = HydroTrial(trial_list[:2], hydro_id="prof_d", scaling_num=self.scaling_num)
            self.add_hydrotrial(trial_single)
            self.add_hydrotrial(trial_dual)

        # Clean
        self.trial_groups = self.init_trial_groups()
        self._live_trials = set()

    def trial_fusion_according_plan_evenly(self):
        """Fuse trials according to the fusion plan."""
        assert self.fusion_enabled
        if self.trial_groups:
            for batch_size, trial_list in self.trial_groups.items():
                fusion_limit = self.fusion_plan[batch_size]
                if len(trial_list) == 0:
                    continue

                if len(trial_list) > fusion_limit:
                    # Try to split them evenly
                    fuse_trial_num = math.ceil(len(trial_list) / math.ceil(len(trial_list) / fusion_limit))
                    trial_batches = list(self.split_trial_into_batches(trial_list, fuse_trial_num))
                    for trial_batch in trial_batches:
                        hydrotrial = self.create_hydrotrial(trial_batch, fuse_trial_num)
                        self.add_hydrotrial(hydrotrial)
                else:
                    hydrotrial = self.create_hydrotrial(trial_list, fusion_limit)
                    self.add_hydrotrial(hydrotrial)
        else:  # Not grouped by batch_size
            fusion_limit = self.fusion_plan
            trial_list = self._live_trials
            if len(trial_list) > fusion_limit:
                trial_batches = list(self.split_trial_into_batches(trial_list, fusion_limit))
                for trial_batch in trial_batches:
                    hydrotrial = self.create_hydrotrial(trial_batch, fusion_limit)
                    self.add_hydrotrial(hydrotrial)
            else:
                hydrotrial = self.create_hydrotrial(trial_list, fusion_limit)
                self.add_hydrotrial(hydrotrial)

        # Clean
        self.trial_groups = self.init_trial_groups()
        self._live_trials = set()

    def trial_fusion_according_plan_and_resource(self):
        """Fuse trials according to the fusion plan."""
        assert self.fusion_enabled

        total_resource = self.trial_executor.get_total_resource()
        occupied_resource = self.trial_executor.get_occupied_resource()
        total_gpu, occupied_gpu = total_resource["GPU"], occupied_resource["GPU"]
        available_gpu = total_gpu - occupied_gpu

        if len(self._live_trials) == 0:
            return

        if self.trial_groups is not None:
            if len(self.trial_groups) > available_gpu:
                self.trial_fusion_according_plan_evenly()
                return
            sorted_batch_list = sorted(self.trial_groups, key=lambda x: len(self.trial_groups[x]), reverse=True)
            self.trial_groups = {i: self.trial_groups[i] for i in sorted_batch_list}

            base_gpus_per_group = available_gpu // len(self.trial_groups)
            mod_gpu = available_gpu % len(self.trial_groups)
            gpus_per_group = {i: base_gpus_per_group for i in sorted_batch_list}
            if mod_gpu > 0:
                for i in range(int(mod_gpu)):
                    gpus_per_group[sorted_batch_list[i]] += 1

            for batch_size, trial_list in self.trial_groups.items():
                fusion_limit = self.fusion_plan[batch_size]
                gpus = gpus_per_group[batch_size]
                trials_per_gpu = math.ceil(len(trial_list) / gpus)

                if len(trial_list) == 0:
                    continue

                if trials_per_gpu <= fusion_limit:
                    trial_batches = list(self.split_trial_into_batches(trial_list, trials_per_gpu))
                    for trial_batch in trial_batches:
                        hydrotrial = self.create_hydrotrial(trial_batch, trials_per_gpu)
                        self.add_hydrotrial(hydrotrial)
                else:  # len(trial_list) > fusion_limit:
                    # Try to split them evenly
                    fuse_trial_num = math.ceil(len(trial_list) / math.ceil(len(trial_list) / fusion_limit))
                    trial_batches = list(self.split_trial_into_batches(trial_list, fuse_trial_num))
                    for trial_batch in trial_batches:
                        hydrotrial = self.create_hydrotrial(trial_batch, fuse_trial_num)
                        self.add_hydrotrial(hydrotrial)
        else:  # Not grouped by batch_size
            fusion_limit = self.fusion_plan
            trial_list = list(self._live_trials)
            trials_per_gpu = math.ceil(len(trial_list) / available_gpu)

            if trials_per_gpu <= fusion_limit:
                trial_batches = list(self.split_trial_into_batches(trial_list, trials_per_gpu))
                for trial_batch in trial_batches:
                    hydrotrial = self.create_hydrotrial(trial_batch, trials_per_gpu)
                    self.add_hydrotrial(hydrotrial)
            else:  # len(trial_list) > fusion_limit:
                # Try to split them evenly
                fuse_trial_num = math.ceil(len(trial_list) / math.ceil(len(trial_list) / fusion_limit))
                trial_batches = list(self.split_trial_into_batches(trial_list, fuse_trial_num))
                for trial_batch in trial_batches:
                    hydrotrial = self.create_hydrotrial(trial_batch, fuse_trial_num)
                    self.add_hydrotrial(hydrotrial)

        # Clean
        self.trial_groups = self.init_trial_groups()
        self._live_trials = set()

    def get_best_hydrotrial(self):
        hydrotrials = self.get_hydrotrials()

        metric_op = 1.0 if self.mode == "max" else -1.0
        best_updated = False
        for t in hydrotrials:
            if not t.last_result:
                continue
            if self._metric not in t.last_result:
                continue
            if t.is_targettrial():  # Not compare with target trial
                continue
            if metric_op > 0:
                t.best_metric_inside = max(t.last_result[self._metric])
            else:
                t.best_metric_inside = min(t.last_result[self._metric])
            if not self.best_hydrotrial or metric_op * t.best_metric_inside > metric_op * self.best_hydro_metric:
                self.best_hydro_metric = t.best_metric_inside
                self.best_hydrotrial = t
                trial = self.best_hydrotrial.get_best_trial_inside(self._metric, self.mode)
                if self.best_trial is not trial:
                    self.best_trial = trial
                    best_updated = True
        return best_updated

    def check_idle_gpu(self):
        total_resource = self.trial_executor.get_total_resource()
        occupied_resource = self.trial_executor.get_occupied_resource()
        total_gpu, occupied_gpu = total_resource["GPU"], occupied_resource["GPU"]
        available_gpu = total_gpu - occupied_gpu
        return available_gpu

    def eager_transfer_target_trial_if_possible(self, idle_resource_check: bool = True):
        if self.terminated_sample_num < self.eager_transfer_num:
            return

        # If enable `idle_resource_check`, Hydro will schedule target trial only if there is idle resource.
        if idle_resource_check:
            if self.check_idle_gpu() == 0:
                return

        best_updated = self.get_best_hydrotrial()
        if best_updated and self.best_trial not in self.best_trial_list:  # avoid duplicate execute same trial
            self.best_trial_list.append(self.best_trial)
            targettrial = self.create_targettrial(self.best_trial)
            self.add_targettrial(targettrial)

    def step(self):
        """Runs one step of the trial event loop.

        Callers should typically run this method repeatedly in a loop. They
        may inspect or modify the runner's state in between calls to step().
        """
        if self.is_finished():
            raise TuneError("Called step when all trials finished?")
        with warn_if_slow("on_step_begin"):
            # self.trial_executor.on_step_begin(self.get_hydrotrials())
            self.trial_executor.on_step_begin()
        with warn_if_slow("callbacks.on_step_begin"):
            self._callbacks.on_step_begin(iteration=self._iteration, trials=self._hydrotrials)

        next_hydrotrial = self._update_trial_queue_and_get_next_trial()
        if next_hydrotrial:
            logger.debug(f"Got new trial to run: {next_hydrotrial}")

        self._wait_and_handle_event(next_hydrotrial=next_hydrotrial)

        self._stop_experiment_if_needed()

        try:
            self.checkpoint()
        except Exception as e:
            logger.warning(f"Trial Runner checkpointing failed: {str(e)}")
        self._iteration += 1

        if self._server:
            with warn_if_slow("server"):
                self._process_stop_requests()

            if self.is_finished():
                self._server.shutdown()

        self._reconcile_live_hydrotrials()

        with warn_if_slow("on_step_end"):
            self.trial_executor.on_step_end(self.get_hydrotrials())
        with warn_if_slow("callbacks.on_step_end"):
            self._callbacks.on_step_end(iteration=self._iteration, trials=self._hydrotrials)

    def profile_GPU_memory(self, index):
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = meminfo.total >> 20  # Byte to MB
        used_mem = meminfo.used >> 20
        # meminfo = nvml.nvmlDeviceGetBAR1MemoryInfo(handle)
        # total_mem = meminfo.bar1Total >> 20  # Byte to MB
        # used_mem = meminfo.bar1Used >> 20
        nvml.nvmlShutdown()
        return total_mem, used_mem

    def _process_trial_results(self, trial, results):
        logger.debug(f"Processing trial results for trial {trial}: {results}")
        with warn_if_slow(
            "process_trial_results",
            message="Processing trial results took {duration:.3f} s, "
            "which may be a performance bottleneck. Please consider "
            "reporting results less frequently to Ray Tune.",
        ):
            if self.profile_stage:
                gpu_id_list = results[0]["gpu_ids"]
                assert len(gpu_id_list) == 1, "Scaled model should fit in single GPU during profiling."
                total_mem, used_mem = self.profile_GPU_memory(gpu_id_list[0])
                self.planer.report_memory_usage(trial.hydro_id, used_mem, total_mem)
                logger.debug(f"Total GPU memory: {total_mem} MB, {trial.hydro_id} Used: {used_mem} MB")

            for i, result in enumerate(results):
                with warn_if_slow("process_trial_result"):
                    decision = self._process_trial_result(trial, result)
                if decision is None:
                    # If we didn't get a decision, this means a
                    # non-training future (e.g. a save) was scheduled.
                    # We do not allow processing more results then.
                    if i < len(results) - 1:
                        if log_once("trial_runner_buffer_checkpoint"):
                            logger.warning(
                                f"Trial {trial} has a non-training future "
                                f"scheduled but {len(results) - i} results "
                                f"left to process. This means that a "
                                f"checkpoint was requested, but buffered "
                                f"training was continued before it was "
                                f"saved. Consider using non-buffered "
                                f"training by setting the env variable "
                                f"`TUNE_RESULT_BUFFER_LENGTH=1`."
                            )
                elif decision == TrialScheduler.STOP:
                    # If the decision is to stop the trial,
                    # ignore all results that came after that.
                    break

    def setup_experiments(self, experiments: List[Experiment], total_num_samples: int) -> None:
        """Obtains any necessary information from experiments.

        Mainly used to setup callbacks.

        Args:
            experiments: List of Experiments
                to use.
            total_num_samples: Total number of samples
                factoring in grid search samplers.
        """
        experiment = experiments[0]
        spec = experiment.public_spec if experiment else {}
        spec["total_num_samples"] = total_num_samples
        self._callbacks.setup(**spec)

    def end_experiment_callbacks(self) -> None:
        """Calls ``on_experiment_end`` method in callbacks."""
        self._callbacks.on_experiment_end(trials=self._hydrotrials)

    def auto_transfer_target_trial(self):
        best_updated = self.get_best_hydrotrial()
        if best_updated and self.best_trial not in self.best_trial_list:  # avoid duplicate execute same trial
            self.best_trial_list.append(self.best_trial)
            targettrial = self.create_targettrial(self.best_trial)
            self.add_targettrial(targettrial)

    def is_finished(self):
        """Returns whether all trials have finished running."""
        # The checks here are partly redundant but optimized for quick
        # evaluation. Specifically, if there are live trials, we check
        # these live trials first. Only if none of the live trials is
        # live anymore do we loop over all trials for a final check.
        trials_done = (
            len(self._live_hydrotrials) == 0 or all(trial.is_finished() for trial in self._live_hydrotrials)
        ) and all(trial.is_finished() for trial in self._hydrotrials)

        # Add auto transfer if no targettrial is executed in advance
        if trials_done and len(self._targettrials) == 0 and not self.profile_stage:
            self.auto_transfer_target_trial()
            trials_done = False

        return trials_done and self._search_alg.is_finished()

    def update_pending_trial_resources(self, resources: Union[dict, PlacementGroupFactory]):
        """Update trial resources when resuming from checkpoint.

        Only updating the pending ones.
        """
        assert resources
        if isinstance(resources, dict) and "gpu" not in resources:
            resources["gpu"] = 0
        for trial in self._trials:
            if trial.status == Trial.PENDING:
                trial.update_resources(resources=resources)

    def update_trial_batch_size(self, trial: Trial, largest_batch_size: int):
        config = copy.deepcopy(trial.config)
        if "train_loop_config" in config:
            config = config["train_loop_config"]
        assert "batch_size" in config
        config["batch_size"] = largest_batch_size
        trial.config = {"train_loop_config": config}

    def _update_trial_queue_and_get_next_trial(self) -> Optional[Trial]:
        """Adding suggested trials to the live queue of trials (they start as PENDING trials).

        Returns:
            next_trial: Trial
        """
        wait_for_trial = True  # wait for new trials when all trials are finished
        num_pending_trials = 0
        for trial in self._live_trials:
            if not trial.is_finished():
                wait_for_trial = False
                if trial.status == Trial.PENDING:
                    num_pending_trials += 1

        if not self._search_alg.is_finished():
            # Create pending trials until it fails.
            while num_pending_trials < self._max_pending_trials:
                if not self._update_trial_queue(blocking=wait_for_trial):
                    break
                wait_for_trial = False  # wait at most one trial
                num_pending_trials += 1

        self.group_trials_according_bs(self._live_trials)

        # Profiling
        if self.profile_stage:
            assert len(self._trials) > 0
            self.trial_prepare_for_profiling()

            # self.fusion_plan = self.planer.set_plan_manually(10)
            # self.trial_executor.start_trial(trial_s)
            # self.trial_executor.stop_trial(trial_s)
        else:
            # self.group_trials_according_bs(self._live_trials)
            # self.trial_fusion_according_plan_evenly()
            self.trial_fusion_according_plan_and_resource()
            self.eager_transfer_target_trial_if_possible()

        with warn_if_slow("choose_trial_to_run"):
            return self._scheduler_alg.choose_trial_to_run(self)

    def adjust_hydrotrial_configuration(self, hydrotrial, original_model, metadata, keep_list):
        """Adjust hydrotrial configuration according to keep_list."""
        new_fusion_num = len(keep_list)
        cfg = copy.deepcopy(hydrotrial.config["train_loop_config"])
        model = slice_model(original_model, keep_list)

        for k, v in hydrotrial.grouped_params.items():
            k = k.replace("train_loop_config/", "")
            if isinstance(cfg[k], int):
                continue

            cfg[k] = [cfg[k][i] for i in keep_list]

        cfg[FUSION_N] = new_fusion_num
        cfg[PREVIOUS_MODEL] = model
        cfg[PREVIOUS_META] = metadata

        new_hydro_id = self.update_hydrotrial_id(hydrotrial.hydro_id)
        new_trial_list = [hydrotrial.active_trials[i] for i in keep_list]

        new_hydrotrial = HydroTrial(
            new_trial_list,
            config={"train_loop_config": cfg},
            hydro_id=new_hydro_id,
        )
        new_hydrotrial.version_tag = hydrotrial.version_tag
        new_hydrotrial.update_version_tag()
        new_hydrotrial.related_trials = hydrotrial.related_trials

        new_grouped_params = {}
        for k, v in hydrotrial.grouped_params.items():
            if isinstance(hydrotrial.grouped_params[k], int):
                continue
            new_grouped_params[k] = [hydrotrial.grouped_params[k][i] for i in keep_list]
        new_hydrotrial.grouped_params = new_grouped_params
        # new_hydrotrial.last_result = hydrotrial.last_result

        hydrotrial.finished_num = hydrotrial.fusion_number - new_fusion_num
        self.stop_trial(hydrotrial)
        self.add_hydrotrial(new_hydrotrial)

    def update_hydrotrial_id(self, previous_id):
        if previous_id[-1].isalpha():
            new_id = previous_id[:-1] + chr(ord(previous_id[-1]) + 1)
            return new_id
        else:
            return previous_id + "A"

    def _wait_and_handle_event(self, next_hydrotrial: Optional[HydroTrial]):
        try:
            # Single wait of entire tune loop.
            event = self.trial_executor.get_next_executor_event(self._live_hydrotrials, next_hydrotrial is not None)
            if event.type == _ExecutorEventType.PG_READY:
                self._on_pg_ready(next_hydrotrial)
            elif event.type == _ExecutorEventType.NO_RUNNING_TRIAL_TIMEOUT:
                self._insufficient_resources_manager.on_no_available_trials(self.get_hydrotrials())
            elif event.type == _ExecutorEventType.YIELD:
                pass
            else:
                trial = event.trial
                result = event.result
                if event.type == _ExecutorEventType.ERROR:
                    self._on_executor_error(trial, result[_ExecutorEvent.KEY_EXCEPTION])
                elif event.type == _ExecutorEventType.RESTORING_RESULT:
                    self._on_restoring_result(trial)
                else:
                    assert event.type in (
                        _ExecutorEventType.SAVING_RESULT,
                        _ExecutorEventType.TRAINING_RESULT,
                    ), f"Unexpected future type - {event.type}"
                    if event.type == _ExecutorEventType.TRAINING_RESULT:
                        self._on_training_result(trial, result[_ExecutorEvent.KEY_FUTURE_RESULT])
                    else:
                        self._on_saving_result(trial, result[_ExecutorEvent.KEY_FUTURE_RESULT])
                    self._post_process_on_training_saving_result(trial)
        except Exception as e:
            if e is TuneError or self._fail_fast == TrialRunner.RAISE:
                raise e
            else:
                raise TuneError(traceback.format_exc())

    def _on_pg_ready(self, next_hydrotrial: Optional[HydroTrial]):
        def _start_trial(trial: HydroTrial) -> bool:
            """Helper function to start trial and call callbacks"""
            with warn_if_slow("start_trial"):
                if self.trial_executor.start_trial(trial):
                    self._callbacks.on_trial_start(iteration=self._iteration, trials=self._hydrotrials, trial=trial)
                    return True
                return False

        assert next_hydrotrial is not None
        logger.debug(f"Trying to start trial: {next_hydrotrial}")
        if not _start_trial(next_hydrotrial) and next_hydrotrial.status != Trial.ERROR:
            # Only try to start another trial if previous trial startup
            # did not error (e.g. it just didn't start because its
            # placement group is not ready, yet).
            # Without this clause, this test fails:
            # test_trial_runner_pg.py::
            # TrialRunnerPlacementGroupHeterogeneousTest::
            # testResourceDeadlock
            next_hydrotrial = self.trial_executor.get_staged_trial()
            if next_hydrotrial is not None:
                # Must be able to start.
                assert _start_trial(next_hydrotrial)
            else:
                logger.debug(f"Reconciling resource requests: {self.get_hydrotrials()}")
                self.trial_executor._pg_manager.reconcile_placement_groups(self.get_hydrotrials())

    def _on_saving_result(self, trial, checkpoint_value: Union[ray.ObjectRef, str]):
        with warn_if_slow("process_trial_save") as _profile:
            self._process_trial_save(trial, checkpoint_value)
        with warn_if_slow("callbacks.on_trial_save"):
            self._callbacks.on_trial_save(iteration=self._iteration, trials=self._hydrotrials, trial=trial)
        if _profile.too_slow and trial.sync_on_checkpoint:
            # TODO(ujvl): Suggest using cloud checkpointing once
            #  API has converged.

            msg = (
                "Consider turning off forced head-worker trial "
                "checkpoint syncs by setting sync_on_checkpoint=False"
                ". Note that this may result in faulty trial "
                "restoration if a failure occurs while the checkpoint "
                "is being synced from the worker to the head node."
            )

            if trial.location.hostname and (trial.location.hostname != get_node_ip_address()):
                if log_once("tune_head_worker_checkpoint"):
                    logger.warning(msg)

    def _on_restoring_result(self, trial):
        with warn_if_slow("process_trial_restore"):
            self._process_trial_restore(trial)
        with warn_if_slow("callbacks.on_trial_restore"):
            self._callbacks.on_trial_restore(iteration=self._iteration, trials=self._hydrotrials, trial=trial)

    def get_hydrotrials(self):
        """Returns the list of trials managed by this TrialRunner.

        Note that the caller usually should not mutate trial state directly.
        """
        return self._hydrotrials

    def get_live_hydrotrials(self):
        """Returns the set of trials that are not in Trial.TERMINATED state."""
        return self._live_hydrotrials

    def add_trial(self, trial: Trial):
        """Adds a new trial to this TrialRunner.

        Trials may be added at any time.

        Args:
            trial: Trial to queue.
        """
        self._trials.append(trial)
        if trial.status != Trial.TERMINATED:
            self._live_trials.add(trial)
        with warn_if_slow("scheduler.on_trial_add"):
            self._scheduler_alg.on_trial_add(TrialRunnerWrapper(self, runner_whitelist_attr={"search_alg"}), trial)
        # self.trial_executor.mark_trial_to_checkpoint(trial)

    def add_hydrotrial(self, hydrotrial: HydroTrial):
        """Adds a new hydrotrial to this TrialRunner.

        Trials may be added at any time.

        Args:
            trial: Trial to queue.
        """
        if hydrotrial.version_tag is not None:
            self._hydrotrials = [hydrotrial] + self._hydrotrials  # Prioritize resuming trial
        else:
            self._hydrotrials.append(hydrotrial)
        if hydrotrial.status != Trial.TERMINATED:
            self._live_hydrotrials.add(hydrotrial)
        # with warn_if_slow("scheduler.on_trial_add"):
        #     self._scheduler_alg.on_trial_add(TrialRunnerWrapper(self, runner_whitelist_attr={"search_alg"}), hydrotrial)
        self.trial_executor.mark_trial_to_checkpoint(hydrotrial)

    def add_targettrial(self, targettrial: HydroTrial):
        """Adds a new targettrial to this TrialRunner.

        Trials may be added at any time.

        Args:
            trial: Trial to queue.
        """
        self._targettrials.append(targettrial)  # For record
        self._hydrotrials = [targettrial] + self._hydrotrials  # Prioritize target trial
        if targettrial.status != Trial.TERMINATED:
            self._live_targettrials.add(targettrial)
            self._live_hydrotrials.add(targettrial)
        # with warn_if_slow("scheduler.on_trial_add"):
        #     self._scheduler_alg.on_trial_add(TrialRunnerWrapper(self, runner_whitelist_attr={"search_alg"}), targettrial)
        self.trial_executor.mark_trial_to_checkpoint(targettrial)

    def debug_string(self, delim="\n"):
        from hydro.tune.progress_reporter import _trial_progress_str

        result_keys = [list(t.last_result) for t in self.get_hydrotrials() if t.last_result]
        metrics = set().union(*result_keys)
        messages = [
            self._scheduler_alg.debug_string(),
            self.trial_executor.debug_string(),
            _trial_progress_str(self.get_hydrotrials(), metrics, force_table=True),
        ]
        return delim.join(messages)

    def _stop_experiment_if_needed(self):
        """Stops all trials."""
        fail_fast = self._fail_fast and self._has_errored
        if self._stopper.stop_all() or fail_fast or self._should_stop_experiment:
            self._search_alg.set_finished()
            [self.trial_executor.stop_trial(t) for t in self._hydrotrials if t.status is not Trial.ERROR]

    def _process_trial_result(self, trial, result):
        result.update(trial_id=trial.trial_id)
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.tolist()

        is_duplicate = RESULT_DUPLICATE in result
        force_checkpoint = result.get(SHOULD_CHECKPOINT, False)
        # TrialScheduler and SearchAlgorithm still receive a
        # notification because there may be special handling for
        # the `on_trial_complete` hook.
        if is_duplicate:
            logger.debug("Trial finished without logging 'done'.")
            result = trial.last_result
            result.update(done=True)

        self._total_time += result.get(TIME_THIS_ITER_S, 0)

        flat_result = flatten_dict(result)
        self._validate_result_metrics(flat_result)

        if self._stopper(trial.trial_id, result) or trial.should_stop(flat_result):
            decision = TrialScheduler.STOP
        else:
            with warn_if_slow("scheduler.on_trial_result"):
                decision = self._scheduler_alg.on_trial_result(self, trial, flat_result)
        if decision == TrialScheduler.STOP:
            result.update(done=True)
        else:
            # Only updating search alg if the trial is not to be stopped.
            with warn_if_slow("search_alg.on_trial_result"):
                self._search_alg.on_trial_result(trial.trial_id, flat_result)

        if decision == TrialScheduler.NOOP:
            assert trial.keep_list is not None

        # If this is not a duplicate result, the callbacks should
        # be informed about the result.
        if not is_duplicate:
            with warn_if_slow("callbacks.on_trial_result"):
                self._callbacks.on_trial_result(
                    iteration=self._iteration,
                    trials=self._hydrotrials,
                    trial=trial,
                    result=result.copy(),
                )
            trial.update_last_result(result)
            # Include in next experiment checkpoint
            self.trial_executor.mark_trial_to_checkpoint(trial)

        # Checkpoints to disk. This should be checked even if
        # the scheduler decision is STOP or PAUSE. Note that
        # PAUSE only checkpoints to memory and does not update
        # the global checkpoint state.
        self._checkpoint_trial_if_needed(trial, force=force_checkpoint)

        if trial.is_saving:
            logger.debug(f"Caching trial decision for trial {trial}: {decision}")
            # Cache decision to execute on after the save is processed.
            # This prevents changing the trial's state or kicking off
            # another training step prematurely.
            self._cached_trial_decisions[trial.trial_id] = decision
            return None
        else:
            self._queue_decision(trial, decision)
            return decision

    def _validate_result_metrics(self, result):
        """
        Check if any of the required metrics was not reported
        in the last result. If the only items are ``done`` or any of
        DEBUG_METRICS, this means that no result was ever received and
        the trial just returned. This is also okay and will not raise
        an error.

        This will ignore checking for the DEFAULT_METRIC.
        """
        if int(os.environ.get("TUNE_DISABLE_STRICT_METRIC_CHECKING", 0)) != 1 and (
            len({k for k in result if k not in list(DEBUG_METRICS) + [DONE]}) > 1
        ):
            base_metric = self._metric if self._metric != DEFAULT_METRIC else None
            scheduler_metric = self._scheduler_alg.metric if self._scheduler_alg.metric != DEFAULT_METRIC else None
            search_metrics = self._search_alg.metric if self._search_alg.metric != DEFAULT_METRIC else None

            if isinstance(search_metrics, str):
                search_metrics = [search_metrics]

            if base_metric and base_metric not in result:
                report_metric = base_metric
                location = "tune.TuneConfig()"
            elif scheduler_metric and scheduler_metric not in result:
                report_metric = scheduler_metric
                location = type(self._scheduler_alg).__name__
            elif search_metrics and any(search_metric not in result for search_metric in search_metrics):
                report_metric = list(
                    filter(
                        lambda search_metric: search_metric not in result,
                        search_metrics,
                    )
                )
                if len(report_metric) == 1:
                    report_metric = report_metric[0]
                location = type(self._search_alg).__name__
            else:
                report_metric = None
                location = None

            if report_metric:
                raise ValueError(
                    "Trial returned a result which did not include the "
                    "specified metric(s) `{}` that `{}` expects. "
                    "Make sure your calls to `tune.report()` include the "
                    "metric, or set the "
                    "TUNE_DISABLE_STRICT_METRIC_CHECKING "
                    "environment variable to 1. Result: {}".format(report_metric, location, result)
                )

    def _process_trial_save(self, trial: Trial, checkpoint_value: Union[ray.ObjectRef, str]):
        """Processes a trial save.

        Acts on the decision cached during the last `_process_trial` call.

        Args:
            trial: Trial being saved.
        """
        logger.debug("Trial %s: Processing trial save.", trial)

        try:
            trial.saving_to.dir_or_data = checkpoint_value
            self._callbacks.on_checkpoint(
                iteration=self._iteration,
                trials=self._hydrotrials,
                trial=trial,
                checkpoint=trial.saving_to,
            )
            trial.on_checkpoint(trial.saving_to)
            if trial.checkpoint.storage_mode != CheckpointStorage.MEMORY:
                self.trial_executor.mark_trial_to_checkpoint(trial)
        except Exception:
            logger.exception("Trial %s: Error handling checkpoint %s", trial, checkpoint_value)
            if self._fail_fast == TrialRunner.RAISE:
                raise

        trial.saving_to = None
        decision = self._cached_trial_decisions.pop(trial.trial_id, None)
        if decision and checkpoint_value:
            self._queue_decision(trial, decision)

    def _process_trial_restore(self, trial: HydroTrial):
        """Processes a trial restore.

        Args:
            trial: Trial being restored.
        """
        logger.debug("Trial %s: Processing trial restore.", trial)
        trial.on_restore()
        logger.debug("Trial %s: Restore processed successfully", trial)
        self.trial_executor.set_status(trial, Trial.RUNNING)
        self.trial_executor.continue_training(trial)
        self._live_hydrotrials.add(trial)

    def _execute_action(self, trial: Trial, decision: str):
        """Executes action based on decision.

        Args:
            trial: Trial to act on.
            decision: Scheduling decision to undertake.
        """
        if decision == TrialScheduler.CONTINUE:
            self.trial_executor.continue_training(trial)
        elif decision == TrialScheduler.PAUSE:
            self.pause_trial(trial)
        elif decision == TrialScheduler.STOP:
            self.stop_trial(trial)
        elif decision == TrialScheduler.NOOP:
            assert trial.has_checkpoint()
            checkpoint = trial.checkpoint
            checkpoint_dir = checkpoint.dir_or_data
            metadata = TrainableUtil.load_metadata(checkpoint_dir)
            checkpoint = checkpoint.to_air_checkpoint()
            model = checkpoint.get_model()

            self.adjust_hydrotrial_configuration(trial, model, metadata, trial.keep_list)
        else:
            raise ValueError("Invalid decision: {}".format(decision))

    def _post_process_on_training_saving_result(self, trial):
        # `self._queued_trial_decisions` now contains a final decision
        # based on all results
        final_decision = self._queued_trial_decisions.pop(trial.trial_id, None)
        if final_decision:
            self._execute_action(trial, final_decision)

    def _process_trial_failure(self, trial: Trial, exc: Optional[Union[TuneError, RayTaskError]] = None):
        """Handle trial failure.

        Attempt trial recovery if possible, clean up state otherwise.

        Args:
            trial: Failed trial.
            exc: Exception prior to invoking this method.
        """
        self._has_errored = True
        if trial.status == Trial.RUNNING:
            if trial.should_recover():
                self._try_recover(trial, exc=exc)
            else:
                self._scheduler_alg.on_trial_error(self, trial)
                self._search_alg.on_trial_complete(trial.trial_id, error=True)
                self._callbacks.on_trial_error(iteration=self._iteration, trials=self._hydrotrials, trial=trial)
                self.trial_executor.stop_trial(trial, exc=exc)

    def _try_recover(self, trial: Trial, exc: Union[TuneError, RayTaskError]):
        """Tries to recover trial.

        Notifies SearchAlgorithm and Scheduler if failure to recover.

        Args:
            trial: Trial to recover.
            exc: Exception prior to invoking this method.
        """
        self._cached_trial_decisions.pop(trial.trial_id, None)
        # Resetting this, in case that the trial is in saving status when it crashes.
        if trial.is_saving:
            trial.saving_to = None
        if trial.is_restoring and exc:
            exc = _TuneRestoreError(exc)
        self.trial_executor.stop_trial(
            trial,
            error=exc is not None,
            exc=exc,
        )
        if self.trial_executor.has_resources_for_trial(trial):
            requeue_trial = False
            logger.info(
                "Trial %s: Attempting to restore trial state from last checkpoint.",
                trial,
            )
            # TODO(xwjiang): For better consistency, consider not starting
            #  trials here. Instead rely on requeuing the trial.
            started = self.trial_executor.start_trial(trial)
            if not started:
                requeue_trial = True
            elif trial.status == Trial.ERROR:
                logger.exception("Trial %s: Error restoring trial from checkpoint, abort.", trial)
                if started:
                    # Clean up again if an actor was launched
                    self.trial_executor.stop_trial(trial, error=True)
                self._scheduler_alg.on_trial_error(self, trial)
                self._search_alg.on_trial_complete(trial.trial_id, error=True)
                self._callbacks.on_trial_error(iteration=self._iteration, trials=self._hydrotrials, trial=trial)
            else:
                logger.debug("Trial %s: Restore dispatched correctly.", trial)
        else:
            requeue_trial = True

        if requeue_trial:
            logger.debug("Trial %s: Notifying Scheduler and requeueing.", trial)
            self._requeue_trial(trial)

    def _requeue_trial(self, trial):
        """Notification to TrialScheduler and requeue trial.

        This does not notify the SearchAlgorithm because the function
        evaluation is still in progress.

        """
        self._scheduler_alg.on_trial_error(self, trial)
        self.trial_executor.set_status(trial, Trial.PENDING)

        # TODO(rliaw): Right now, this pushes the trial to the end of queue
        # because restoration can be expensive. However, this is not
        # ideal since it just hides the issue - a better fix would
        # be to use an actor table to detect the IP of the Trainable
        # and rsync the files there.
        # See https://github.com/ray-project/ray/issues/5168
        self._trials.pop(self._trials.index(trial))
        self._trials.append(trial)
        self._live_trials.add(trial)

        # with warn_if_slow("scheduler.on_trial_add"):
        #     self._scheduler_alg.on_trial_add(TrialRunnerWrapper(self, runner_whitelist_attr={"search_alg"}), trial)

    def _update_trial_queue(self, blocking: bool = False, timeout: int = 600) -> bool:
        """Adds next trials to queue if possible.

        Note that the timeout is currently unexposed to the user.

        Args:
            blocking: Blocks until either a trial is available
                or is_finished (timeout or search algorithm finishes).
            timeout: Seconds before blocking times out.

        Returns:
            Boolean indicating if a new trial was created or not.
        """
        trial = self._search_alg.next_trial()
        if blocking and not trial:
            start = time.time()
            # Checking `is_finished` instead of _search_alg.is_finished
            # is fine because blocking only occurs if all trials are
            # finished and search_algorithm is not yet finished
            while not trial and not self.is_finished() and time.time() - start < timeout:
                logger.debug("Blocking for next trial...")
                trial = self._search_alg.next_trial()
                time.sleep(1)

        if trial:
            self.update_trial_batch_size_info(trial)
            self.add_trial(trial)
            return True

        return False

    def stop_trial(self, trial):
        """The canonical implementation of stopping a trial.

        Trials may be in any external status when this function is called.
        If trial is in state PENDING or PAUSED, calls `on_trial_remove` for
        scheduler and `on_trial_complete()` for search_alg.
        If trial is in state RUNNING, calls `on_trial_complete` for scheduler
        and search_alg if RUNNING. Caller to ensure that there is no
        outstanding future to be handled for the trial. If there is, the future
        would be discarded.
        """
        try:
            if trial.status in [Trial.ERROR, Trial.TERMINATED]:
                return
            elif trial.status in [Trial.PENDING, Trial.PAUSED]:
                self._scheduler_alg.on_trial_remove(self, trial)
                for t in trial.active_trials:
                    self._search_alg.on_trial_complete(t.trial_id)
            elif trial.status is Trial.RUNNING:
                # By this time trial.last_result should have been
                # updated already.
                self._scheduler_alg.on_trial_complete(self, trial, flatten_dict(trial.last_result))
                for t in trial.active_trials:
                    self._search_alg.on_trial_complete(t.trial_id, result=flatten_dict(trial.last_result))
            self._callbacks.on_trial_complete(iteration=self._iteration, trials=self._hydrotrials, trial=trial)
            self.trial_executor.export_trial_if_needed(trial)
            self.trial_executor.stop_trial(trial)
            self._live_hydrotrials.discard(trial)
            if trial.is_targettrial():
                self._live_targettrials.discard(trial)
            else:
                if trial.finished_num is None:
                    self.terminated_sample_num += trial.fusion_number
                else:
                    self.terminated_sample_num += trial.finished_num
        except Exception as e:
            logger.exception("Trial %s: Error stopping trial.", trial)
            if self._fail_fast == TrialRunner.RAISE:
                raise
            if isinstance(e, TuneError):
                self._process_trial_failure(trial, exc=e)
            else:
                self._process_trial_failure(trial, _TuneStopTrialError(traceback.format_exc()))

    # def cleanup_trials(self):
    #     self.trial_executor.cleanup(self.get_hydrotrials())

    def cleanup(self):
        """Cleanup trials and callbacks."""
        self.cleanup_trials()
        self.end_experiment_callbacks()

    def _reconcile_live_hydrotrials(self):
        """Loop through live trials and remove if terminated"""
        for trial in list(self._live_hydrotrials):
            # Only for TERMINATED trials. ERRORed trials might be retried.
            if trial.status == Trial.TERMINATED:
                self._live_hydrotrials.remove(trial)

                if trial.is_targettrial():
                    self._live_targettrials.discard(trial)

    def __getstate__(self):
        """Gets state for trial.

        Note that this is not used as a pickling override as
        does not have all fields.
        """
        state = self.__dict__.copy()
        for k in [
            "_hydrotrials",
            "_live_hydrotrials",
            "_stop_queue",
            "_server",
            "_search_alg",
            "_scheduler_alg",
            "_pending_trial_queue_times",
            "trial_executor",
            "_syncer",
            "_callbacks",
            "_checkpoint_manager",
            "_local_checkpoint_dir",
            "_sync_config",
            "_experiment_dir_name",
        ]:
            del state[k]
        state["launch_web_server"] = bool(self._server)
        return state
