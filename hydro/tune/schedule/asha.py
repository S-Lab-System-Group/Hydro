import logging
from typing import Dict, List, Optional, Union

import numpy as np

from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.execution.trial_runner import TrialRunner
from ray.tune.experiment import Trial
from ray.util import PublicAPI

from hydro.tune import HydroTrialRunner, HydroTrial

logger = logging.getLogger(__name__)


@PublicAPI
class ASHAHydroScheduler(AsyncHyperBandScheduler):
    """Implements the Async Successive Halving.

    This should provide similar theoretical performance as HyperBand but
    avoid straggler issues that HyperBand faces. One implementation detail
    is when using multiple brackets, trial allocation to bracket is done
    randomly with over a softmax probability.

    See https://arxiv.org/abs/1810.05934

    Args:
        time_attr: A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute. If None but a mode was passed,
            the `ray.tune.result.DEFAULT_METRIC` will be used per default.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        max_t: max time units per trial. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period: Only stop trials at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor: Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets: Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
        stop_last_trials: Whether to terminate the trials after
            reaching max_t. Defaults to True.
    """

    def __init__(
        self,
        time_attr: str = "training_iteration",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factor: float = 4,
        brackets: int = 1,
        stop_last_trials: bool = True,
    ):
        assert max_t > 0, "Max (time_attr) not valid!"
        assert max_t >= grace_period, "grace_period must be <= max_t!"
        assert grace_period > 0, "grace_period must be positive!"
        assert reduction_factor > 1, "Reduction Factor not valid!"
        assert brackets == 1, "currently only support brackets=1."
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'!"

        self._reduction_factor = reduction_factor
        self._max_t = max_t

        self._trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self._brackets = [
            _Bracket(
                grace_period,
                max_t,
                reduction_factor,
                s,
                stop_last_trials=stop_last_trials,
            )
            for s in range(brackets)
        ]
        self._counter = 0  # for
        self._num_stopped = 0
        self._metric = metric
        self._mode = mode
        self._metric_op = None
        if self._mode == "max":
            self._metric_op = 1.0
        elif self._mode == "min":
            self._metric_op = -1.0
        self._time_attr = time_attr
        self._stop_last_trials = stop_last_trials

    def on_trial_add(self, trial_runner: TrialRunner, trial: Trial):
        if not self._metric or not self._metric_op:
            raise ValueError(
                "{} has been instantiated without a valid `metric` ({}) or "
                "`mode` ({}) parameter. Either pass these parameters when "
                "instantiating the scheduler, or pass them as parameters "
                "to `tune.TuneConfig()`".format(self.__class__.__name__, self._metric, self._mode)
            )

        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        self._trial_info[trial.trial_id] = self._brackets[idx]

    def parse_result(self, trial: HydroTrial, result: Dict):
        iteration = result[self._time_attr]
        id_list = trial.active_trials_id
        result_list = [self._metric_op * metric for metric in result[self._metric]]
        assert len(id_list) == len(result_list)
        return iteration, id_list, result_list

    def process_action_list(self, action_list: List, trial: HydroTrial):
        if all(act == TrialScheduler.CONTINUE for act in action_list):
            return TrialScheduler.CONTINUE
        elif all(act == TrialScheduler.STOP for act in action_list):
            self._num_stopped += len(action_list)
            return TrialScheduler.STOP
        else:
            keep_list = [idx for idx, act in enumerate(action_list) if act == TrialScheduler.CONTINUE]
            self._num_stopped += len(action_list) - len(keep_list)
            trial.keep_list = keep_list
            # NOTE: we use `NOOP` to indicate that we need to keep some trials in the hydrotrial
            return TrialScheduler.NOOP

    def on_trial_result(self, trial_runner: TrialRunner, trial: HydroTrial, result: Dict) -> str:
        action = TrialScheduler.CONTINUE
        if trial.is_targettrial():
            return action
        if self._time_attr not in result or self._metric not in result:
            return action
        if result[self._time_attr] >= self._max_t and self._stop_last_trials:
            action = TrialScheduler.STOP
        else:
            iteration, id_list, result_list = self.parse_result(trial, result)
            action_list = []
            for idx, trial_id in enumerate(id_list):
                res = result_list[idx]
                bracket = self._trial_info[trial_id]
                action_list.append(bracket.on_result(trial_id, iteration, res))
            action = self.process_action_list(action_list, trial)
        return action

    def on_trial_complete(self, trial_runner: TrialRunner, trial: Trial, result: Dict):
        if self._time_attr not in result or self._metric not in result or trial.is_targettrial():
            return
        iteration, id_list, result_list = self.parse_result(trial, result)
        keep_list = trial.keep_list
        if keep_list is None:
            for idx, trial_id in enumerate(id_list):
                res = result_list[idx]
                bracket = self._trial_info[trial_id]
                bracket.on_result(trial_id, iteration, res)
                del self._trial_info[trial_id]
        else:
            for idx, trial_id in enumerate(id_list):
                if idx in keep_list:
                    continue
                res = result_list[idx]
                bracket = self._trial_info[trial_id]
                bracket.on_result(trial_id, iteration, res)
                del self._trial_info[trial_id]

    def on_trial_remove(self, trial_runner: TrialRunner, trial: Trial):
        if trial.is_targettrial():
            return
        id_list = trial.active_trials_id
        for idx, trial_id in enumerate(id_list):
            del self._trial_info[trial_id]

    def debug_string(self) -> str:
        out = f"Using AsyncHyperBand: num_stopped={self._num_stopped}"
        out += "\n" + "\n".join([b.debug_str() for b in self._brackets])
        return out

    def choose_trial_to_run(self, trial_runner: HydroTrialRunner) -> Optional[HydroTrial]:
        for trial in trial_runner.get_hydrotrials():
            if trial.status == HydroTrial.PENDING and trial_runner.trial_executor.has_resources_for_trial(trial):
                return trial
        for trial in trial_runner.get_hydrotrials():
            if trial.status == HydroTrial.PAUSED and trial_runner.trial_executor.has_resources_for_trial(trial):
                return trial
        return None


class _Bracket:
    """Bookkeeping system to track the cutoffs.

    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.

    Example:
        >>> trial1, trial2, trial3 = ... # doctest: +SKIP
        >>> b = _Bracket(1, 10, 2, 0) # doctest: +SKIP
        >>> # CONTINUE
        >>> b.on_result(trial1, 1, 2) # doctest: +SKIP
        >>> # CONTINUE
        >>> b.on_result(trial2, 1, 4) # doctest: +SKIP
        >>> # rungs are reversed
        >>> b.cutoff(b._rungs[-1][1]) == 3.0 # doctest: +SKIP
         # STOP
        >>> b.on_result(trial3, 1, 1) # doctest: +SKIP
        >>> b.cutoff(b._rungs[3][1]) == 2.0 # doctest: +SKIP
    """

    def __init__(
        self,
        min_t: int,
        max_t: int,
        reduction_factor: float,
        s: int,
        stop_last_trials: bool = True,
    ):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [(min_t * self.rf ** (k + s), {}) for k in reversed(range(MAX_RUNGS))]
        self._stop_last_trials = stop_last_trials

    def cutoff(self, recorded) -> Optional[Union[int, float, complex, np.ndarray]]:
        if not recorded:
            return None
        return np.nanpercentile(list(recorded.values()), (1 - 1 / self.rf) * 100)

    def on_result(self, trial_id: str, cur_iter: int, cur_rew: Optional[float]) -> str:
        action = TrialScheduler.CONTINUE
        for milestone, recorded in self._rungs:
            if cur_iter >= milestone and trial_id in recorded and not self._stop_last_trials:
                # If our result has been recorded for this trial already, the
                # decision to continue training has already been made. Thus we can
                # skip new cutoff calculation and just continue training.
                # We can also break as milestones are descending.
                break
            if cur_iter < milestone or trial_id in recorded:
                continue
            else:
                cutoff = self.cutoff(recorded)
                if cutoff is not None and cur_rew < cutoff:
                    action = TrialScheduler.STOP
                if cur_rew is None:
                    logger.warning("Reward attribute is None! Consider" " reporting using a different field.")
                else:
                    recorded[trial_id] = cur_rew
                break
        return action

    def debug_str(self) -> str:
        # TODO: fix up the output for this
        iters = " | ".join(["Iter {:.3f}: {}".format(milestone, self.cutoff(recorded)) for milestone, recorded in self._rungs])
        return "Bracket: " + iters
