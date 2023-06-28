from typing import Callable, Dict, Iterable, List, Optional, Set, Union, Tuple

import copy
import inspect
import logging
import os
import random
import time
import traceback
from collections import deque, defaultdict, Counter
from contextlib import contextmanager
from enum import Enum
from functools import partial


import ray
from ray.air import Checkpoint, AcquiredResources, ResourceRequest
from ray.air._internal.checkpoint_manager import CheckpointStorage, _TrackedCheckpoint
from ray.air.constants import COPY_DIRECTORY_CHECKPOINTS_INSTEAD_OF_MOVING_ENV
from ray.air.execution import ResourceManager
from ray.air.execution.resources.placement_group import (
    PlacementGroupResourceManager,
)
from ray.exceptions import GetTimeoutError, RayTaskError
from ray.tune.error import (
    TuneError,
    _AbortTrialExecution,
    _TuneNoNextExecutorEventError,
    _TuneStartTrialError,
)
from ray.tune.logger import NoopLogger
from ray.tune.result import STDERR_FILE, STDOUT_FILE, TRIAL_INFO
from ray.tune.experiment.trial import Trial, _Location, _TrialInfo
from ray.tune.utils import warn_if_slow
from ray.tune.utils.resource_updater import _ResourceUpdater
from ray.tune.trainable.util import TrainableUtil
from ray.util import log_once
from ray.util.annotations import DeveloperAPI

from ray.tune.execution.ray_trial_executor import (
    RayTrialExecutor,
    _TrialCleanup,
    _ExecutorEventType,
    _ExecutorEvent,
    _LocalWrapper,
    _class_cache,
    _noop_logger_creator,
)

logger = logging.getLogger(__name__)

DEFAULT_GET_TIMEOUT = 60.0  # seconds

DEFAULT_ENV_VARS = {
    # https://github.com/ray-project/ray/issues/28197
    "PL_DISABLE_FORK": "1"
}
ENV_VARS_TO_PROPAGATE = {
    COPY_DIRECTORY_CHECKPOINTS_INSTEAD_OF_MOVING_ENV,
    "TUNE_CHECKPOINT_CLOUD_RETRY_NUM",
    "TUNE_CHECKPOINT_CLOUD_RETRY_WAIT_TIME_S",
}
PREVIOUS_META = "PREVIOUS_META"


class HydroTrialExecutor(RayTrialExecutor):
    """An implementation of TrialExecutor based on Ray. Only support HydroTrail."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._cached_hydrotrial_state = {}
        self._hydrotrials_to_cache = set()

    # hydro
    def get_total_resource(self):
        cpu = self._resource_updater._avail_resources.cpu
        gpu = self._resource_updater._avail_resources.gpu
        return {"CPU": cpu, "GPU": gpu}

    # hydro
    def get_occupied_resource(self):
        return self._occupied_resources()

    # def on_step_begin(self, trials: List[Trial]) -> None:
    #     """Before step() is called, update the available resources."""
    #     self._resource_updater.update_avail_resources()
    #     # self._trial_just_finished_before = self._trial_just_finished
    #     # self._trial_just_finished = False

    def set_status(self, trial: Trial, status: str) -> None:
        """Sets status and checkpoints metadata if needed.

        Only checkpoints metadata if trial status is a terminal condition.
        PENDING, PAUSED, and RUNNING switches have checkpoints taken care of
        in the TrialRunner.

        Args:
            trial: Trial to checkpoint.
            status: Status to set trial to.
        """
        if trial.status == status:
            logger.debug("Trial %s: Status %s unchanged.", trial, trial.status)
        else:
            logger.debug("Trial %s: Changing status from %s to %s.", trial, trial.status, status)
        trial.set_status(status)
        if status in [Trial.TERMINATED, Trial.ERROR]:
            self._trials_to_cache.add(trial)

    def mark_trial_to_checkpoint(self, trial: Trial) -> None:
        self._trials_to_cache.add(trial)

    def get_checkpoints(self) -> Dict[str, str]:
        """Returns a copy of mapping of the trial ID to pickled metadata."""
        for trial in self._trials_to_cache:
            self._cached_trial_state[trial.trial_id] = trial.get_json_state()
        self._trials_to_cache.clear()
        return self._cached_trial_state

    def _train(self, trial):
        """Start one iteration of training and save remote id."""

        if self._find_future(trial):
            logging.debug(
                "Trial {} already has a queued future. Skipping this "
                "`train` call. This may occur if a trial has "
                "been unpaused within a scheduler callback.".format(str(trial))
            )
            return

        assert trial.status == Trial.RUNNING, trial.status
        buffer_time_s = max(
            self._buffer_min_time_s,
            min(self._buffer_max_time_s, len(self._futures) // 10),
        )
        with self._change_working_directory(trial):
            buffer_length = self._buffer_length
            if buffer_length > 1 and trial.checkpoint_at_end:
                # If a trial checkpoint can be triggered externally,
                # it is not safe to buffer results.
                if log_once("trial_executor_buffer_checkpoint"):
                    logger.warning(
                        "Disabling buffered training as you passed " "`checkpoint_at_end` to `air.CheckpointConfig()`."
                    )
                buffer_length = 1

            if buffer_length > 1:
                if trial.checkpoint_freq > 0:
                    buffer_length = min(buffer_length, trial.checkpoint_freq)
                remote = trial.runner.train_buffered.remote(buffer_time_s, buffer_length)
            else:
                remote = trial.runner.train.remote()

        # Local Mode
        if isinstance(remote, dict):
            remote = _LocalWrapper(remote)

        self._futures[remote] = (_ExecutorEventType.TRAINING_RESULT, trial)
        trial_item = self._find_future(trial)
        assert len(trial_item) < 2, trial_item

    def _start_trial(self, trial: Trial) -> bool:
        """Starts trial and restores last result if trial was paused.

        Args:
            trial: The trial to start.

        Returns:
            True if trial was started successfully, False otherwise.

        See `RayTrialExecutor.restore` for possible errors raised.
        """
        self.set_status(trial, Trial.PENDING)
        runner = self._setup_remote_runner(trial)
        if not runner:
            return False
        trial.set_runner(runner)
        self.restore(trial)
        self.set_status(trial, Trial.RUNNING)

        self._unstage_trial_with_resources(trial)

        if not trial.is_restoring:
            if trial.version_tag is not None:
                metadata = trial.config["train_loop_config"][PREVIOUS_META]
                with self._change_working_directory(trial):
                    remote = trial.runner.reset_meta_data.remote(metadata)

            self._train(trial)
        return True

    def start_trial(self, trial: Trial) -> bool:
        """Starts the trial.

        Will not return resources if trial repeatedly fails on start.

        Args:
            trial: Trial to be started.

        Returns:
            True if the remote runner has been started. False if trial was
                not started (e.g. because of lacking resources/pending PG).
        """
        try:
            return self._start_trial(trial)
        except _AbortTrialExecution as e:
            logger.exception("Trial %s: Error starting runner, aborting!", trial)
            time.sleep(2)
            self._stop_trial(trial, exc=e)
            return False
        except Exception as e:
            logger.exception("Trial %s: Unexpected error starting runner.", trial)
            time.sleep(2)
            if isinstance(e, TuneError):
                self._stop_trial(trial, exc=e)
            else:
                self._stop_trial(trial, exc=_TuneStartTrialError(traceback.format_exc()))
            # Note that we don't return the resources, since they may
            # have been lost. TODO(ujvl): is this the right thing to do?
            return False

    def _find_future(self, trial):
        out = [rid for rid, t in self._futures.items() if t[1] is trial]
        assert len(out) <= 1, "Expecting one future for any given trial at any given time."
        return out

    def export_trial_if_needed(self, trial: Trial) -> Dict:
        """Exports model of this trial based on trial.export_formats.

        Return:
            A dict that maps ExportFormats to successfully exported models.
        """
        if trial.export_formats and len(trial.export_formats) > 0:
            with self._change_working_directory(trial):
                return ray.get(
                    trial.runner.export_model.remote(trial.export_formats),
                    timeout=DEFAULT_GET_TIMEOUT,
                )
        return {}

    def has_gpus(self) -> bool:
        return self._resource_updater.get_num_gpus() > 0

    @contextmanager
    def _change_working_directory(self, trial):
        """Context manager changing working directory to trial logdir.
        Used in local mode.

        For non-local mode it is no-op.
        """
        if ray._private.worker._mode() == ray._private.worker.LOCAL_MODE:
            old_dir = os.getcwd()
            try:
                os.chdir(trial.logdir)
                yield
            finally:
                os.chdir(old_dir)
        else:
            yield

    # # Hydro
    # def get_next_executor_event(self, live_hydrotrials: Set[Trial], next_hydrotrial_exists: bool) -> _ExecutorEvent:
    #     """Get the next executor event to be processed in TrialRunner.

    #     In case there are multiple events available for handling, the next
    #     event is determined by the following priority:
    #     1. if there is `next_trial_exists`, and if there is cached resources
    #     to use, PG_READY is emitted.
    #     2. if there is `next_trial_exists` and there is no cached resources
    #     to use, wait on pg future and randomized other futures. If multiple
    #     futures are ready, pg future will take priority to be handled first.
    #     3. if there is no `next_trial_exists`, wait on just randomized other
    #     futures.

    #     An example of #3 would be synchronous hyperband. Although there are pgs
    #     ready, the scheduler is holding back scheduling new trials since the
    #     whole band of trials is waiting for the slowest trial to finish. In
    #     this case, we prioritize handling training result to avoid deadlock
    #     situation.

    #     This is a blocking wait with a timeout (specified with env var).
    #     The reason for the timeout is
    #     we still want to print status info periodically in TrialRunner for
    #     better user experience.

    #     The handle of `ExecutorEvent.STOP_RESULT` is purely internal to
    #     RayTrialExecutor itself. All the other future results are handled by
    #     TrialRunner.

    #     In the future we may want to do most of the handle of
    #     `ExecutorEvent.RESTORE_RESULT` and `SAVING_RESULT` in
    #     RayTrialExecutor itself and only notify TrialRunner to invoke
    #     corresponding callbacks. This view is more consistent with our goal
    #     of TrialRunner responsible for external facing Trial state transition,
    #     while RayTrialExecutor responsible for internal facing transitions,
    #     namely, `is_saving`, `is_restoring` etc.

    #     Also you may notice that the boundary between RayTrialExecutor and
    #     PlacementGroupManager right now is really blurry. This will be
    #     improved once we move to an ActorPool abstraction.

    #     `next_trial_exists` means that there is a trial to run - prioritize
    #     returning PG_READY in this case.
    #     """
    #     # First update status of staged placement groups
    #     self._stage_and_update_status(live_hydrotrials)
    #     while True:
    #         ###################################################################
    #         # when next_trial_exists and there are cached resources
    #         ###################################################################
    #         # There could be existing PGs from either `self._cached_actor_pg`
    #         # or from `self._pg_manager._ready`. If so and if there is indeed
    #         # a next trial to run, we return `PG_READY` future for trial
    #         # runner. The next trial can then be scheduled on this PG.
    #         if next_hydrotrial_exists:
    #             if len(self._cached_actor_pg) > 0:
    #                 return _ExecutorEvent(_ExecutorEventType.PG_READY)
    #             # TODO(xwjiang): Expose proper API when we decide to do
    #             #  ActorPool abstraction.
    #             if any(len(r) > 0 for r in self._pg_manager._ready.values()):
    #                 return _ExecutorEvent(_ExecutorEventType.PG_READY)

    #         ###################################################################
    #         # Prepare for futures to wait
    #         ###################################################################
    #         futures_to_wait = list(self._futures.keys())
    #         random.shuffle(futures_to_wait)
    #         if next_hydrotrial_exists:
    #             # Only wait for pg explicitly if there is next trial to run.
    #             # In which case, handling PG_READY triumphs handling other events.
    #             # Since we want to place pending trial ASAP.
    #             futures_to_wait = self._pg_manager.get_staging_future_list() + futures_to_wait
    #         logger.debug(
    #             f"get_next_executor_event before wait with futures "
    #             f"{futures_to_wait} and "
    #             f"next_trial_exists={next_hydrotrial_exists}"
    #         )

    #         ready_futures, _ = ray.wait(futures_to_wait, num_returns=1, timeout=self._get_next_event_wait)

    #         ###################################################################
    #         # Dealing with no future returned case.
    #         ###################################################################
    #         if len(ready_futures) == 0:
    #             if len(self._futures) == 0:
    #                 # No running trial and timing out with wait, could be we may
    #                 # have insufficient cluster resources that makes tune run
    #                 # infeasible.
    #                 # TODO: Move InsufficientResourceManager's logic
    #                 #  to TrialExecutor. It is not Runner's responsibility!
    #                 return _ExecutorEvent(_ExecutorEventType.NO_RUNNING_TRIAL_TIMEOUT)
    #             else:
    #                 # Training simply takes long time, yield the control back to main
    #                 # event loop to print progress info etc.
    #                 return _ExecutorEvent(_ExecutorEventType.YIELD)

    #         ###################################################################
    #         # If there is future returned.
    #         ###################################################################
    #         assert len(ready_futures) == 1
    #         ready_future = ready_futures[0]

    #         ###################################################################
    #         # If it is a PG_READY event.
    #         ###################################################################
    #         if ready_future not in self._futures.keys():
    #             self._pg_manager.handle_ready_future(ready_future)
    #             return _ExecutorEvent(_ExecutorEventType.PG_READY)

    #         ###################################################################
    #         # non PG_READY event
    #         ###################################################################
    #         result_type, trial_or_pg = self._futures.pop(ready_future)
    #         if result_type == _ExecutorEventType.STOP_RESULT:
    #             pg = trial_or_pg
    #             _post_stop_cleanup(ready_future, pg)
    #         else:
    #             trial = trial_or_pg
    #             assert isinstance(trial, Trial)
    #             try:
    #                 future_result = ray.get(ready_future)
    #                 # For local mode
    #                 if isinstance(future_result, _LocalWrapper):
    #                     future_result = future_result.unwrap()
    #                 if result_type in (
    #                     _ExecutorEventType.TRAINING_RESULT,
    #                     _ExecutorEventType.SAVING_RESULT,
    #                     _ExecutorEventType.RESTORING_RESULT,
    #                 ):
    #                     logger.debug(f"Returning [{result_type}] for trial {trial}")
    #                     return _ExecutorEvent(result_type, trial, result={_ExecutorEvent.KEY_FUTURE_RESULT: future_result},)
    #                 else:
    #                     raise TuneError(f"Unexpected future type - [{result_type}]")
    #             except RayTaskError as e:
    #                 return _ExecutorEvent(
    #                     _ExecutorEventType.ERROR, trial, result={_ExecutorEvent.KEY_EXCEPTION: e.as_instanceof_cause()},
    #                 )
    #             except Exception:
    #                 return _ExecutorEvent(
    #                     _ExecutorEventType.ERROR,
    #                     trial,
    #                     result={_ExecutorEvent.KEY_EXCEPTION: _TuneNoNextExecutorEventError(traceback.format_exc())},
    #                 )
