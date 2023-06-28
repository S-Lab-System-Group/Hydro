import datetime
from dataclasses import dataclass
from typing import Callable, Optional, Union, Dict

from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import TrialScheduler
from ray.tune.search import SearchAlgorithm, Searcher
from ray.util import PublicAPI


@dataclass
@PublicAPI(stability="beta")
class TuneConfig:
    """Tune specific configs.

    Args:
        ======================================================================
        Hydro args:
        scaling_num: The number of model width scaling. Hydro supports switch
            different tuning modes ``Hydro (Scaling+Fusion) | Hydro (Fusion Only)
            | Ray (Classic HPO)`` via setting this value. Specifically,
            0 = Using Ray Tune (disable both scaling and fusion),
            1 = Using Hydro (fusion only, disable scaling),
            Any integer > 1 (preferably powers of 2) enables both scaling and fusion
            Default value is 8.
        fusion_limit: User defined maximum model fusion number. Only work when
            `scaling_num` > 0. Default is None.
            0 = Disabling fusion (scaling only).
            1 = Similar with disabling fusion, but will still replace original model
            with Hydro modules.
            If set to None, Hydro will automatically profile and determine the actual
            fusion number according to GPU memory capacity.
            If set to a positive integer, Hydro will use this value as the fusion limit.
            If set to a dict, Hydro will use this dict as the fusion for different batch size.
        eager_transfer: The ratio of maximum trials (`num_samples`) to start a
            target model trial. Must in (0, 1].
            1 = Disabling eager transfer. Default value is 0.5.
        trial_compile: Whether to enable torch.compile() to further accelerate model
            training throughput. If enabled, Hydro does not support model checkpointing
            and multi-fidelity tuning algorithms. Default is False.



        ======================================================================
        Ray args:
        metric: Metric to optimize. This metric should be reported
            with `tune.report()`. If set, will be passed to the search
            algorithm and scheduler.
        mode: Must be one of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute. If set, will be
            passed to the search algorithm and scheduler.
        search_alg: Search algorithm for optimization. Default to
            random search.
        scheduler: Scheduler for executing the experiment.
            Choose among FIFO (default), MedianStopping,
            AsyncHyperBand, HyperBand and PopulationBasedTraining. Refer to
            ray.tune.schedulers for more options.
        num_samples: Number of times to sample from the
            hyperparameter space. Defaults to 1. If `grid_search` is
            provided as an argument, the grid will be repeated
            `num_samples` of times. If this is -1, (virtually) infinite
            samples are generated until a stopping condition is met.
        max_concurrent_trials: Maximum number of trials to run
            concurrently. Must be non-negative. If None or 0, no limit will
            be applied. This is achieved by wrapping the ``search_alg`` in
            a :class:`ConcurrencyLimiter`, and thus setting this argument
            will raise an exception if the ``search_alg`` is already a
            :class:`ConcurrencyLimiter`. Defaults to None.
        time_budget_s: Global time budget in
            seconds after which all trials are stopped. Can also be a
            ``datetime.timedelta`` object.
        reuse_actors: Whether to reuse actors between different trials
            when possible. This can drastically speed up experiments that start
            and stop actors often (e.g., PBT in time-multiplexing mode). This
            requires trials to have the same resource requirements.
            Defaults to ``True`` for function trainables (including most
            Ray AIR trainers) and ``False`` for class and registered trainables
            (e.g. RLlib).
        trial_name_creator: Optional function that takes in a Trial and returns
            its name (i.e. its string representation). Be sure to include some unique
            identifier (such as `Trial.trial_id`) in each trial's name.
            NOTE: This API is in alpha and subject to change.
        trial_dirname_creator: Optional function that takes in a trial and
            generates its trial directory name as a string. Be sure to include some
            unique identifier (such as `Trial.trial_id`) is used in each trial's
            directory name. Otherwise, trials could overwrite artifacts and checkpoints
            of other trials. The return value cannot be a path.
            NOTE: This API is in alpha and subject to change.
        chdir_to_trial_dir: Whether to change the working directory of each worker
            to its corresponding trial directory. Defaults to `True` to prevent
            contention between workers saving trial-level outputs.
            If set to `False`, files are accessible with paths relative to the
            original working directory. However, all workers on the same node now
            share the same working directory, so be sure to use
            `session.get_trial_dir()` as the path to save any outputs.
    """

    # Currently this is not at feature parity with `tune.run`, nor should it be.
    # The goal is to reach a fine balance between API flexibility and conciseness.
    # We should carefully introduce arguments here instead of just dumping everything.
    mode: Optional[str] = None
    metric: Optional[str] = None
    search_alg: Optional[Union[Searcher, SearchAlgorithm]] = None
    scheduler: Optional[TrialScheduler] = None
    num_samples: int = 1
    max_concurrent_trials: Optional[int] = None
    time_budget_s: Optional[Union[int, float, datetime.timedelta]] = None
    reuse_actors: Optional[bool] = None
    trial_name_creator: Optional[Callable[[Trial], str]] = None
    trial_dirname_creator: Optional[Callable[[Trial], str]] = None
    chdir_to_trial_dir: bool = True
    scaling_num: int = 8
    fusion_limit: Optional[Union[int, Dict]] = None
    eager_transfer: float = 0.5
    trial_compile: bool = False
