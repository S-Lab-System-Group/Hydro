from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING, Tuple, Mapping, Sequence

import datetime
import logging
import os
import signal
import sys
import threading
import time
import warnings
import copy
import math
import shutil
import tempfile
from pathlib import Path

import ray
from ray import tune
from ray.air import CheckpointConfig
from ray.air.util.node import _force_on_current_node
from ray.air.config import RunConfig, ScalingConfig
from ray.tune import Experiment, TuneError, Tuner, Callback, Stopper
from ray.tune.execution.trial_runner import _ResumeConfig
from ray.tune.impl.tuner_internal import TunerInternal
from ray.tune.trainable import Trainable
from ray.tune.execution.ray_trial_executor import RayTrialExecutor
from ray.tune.progress_reporter import (
    ProgressReporter,
    RemoteReporterMixin,
    _detect_progress_metrics,
)
from ray.tune.tune import (
    _get_trainable,
    _check_default_resources_override,
    _check_mixin,
    _check_gpus_in_resources,
    _ray_auto_init,
    _setup_signal_catching,
)
from ray.tune.registry import is_function_trainable
from ray.util import PublicAPI

# Must come last to avoid circular imports
from ray.tune.schedulers import (
    FIFOScheduler,
    PopulationBasedTraining,
    PopulationBasedTrainingReplay,
    ResourceChangingScheduler,
    TrialScheduler,
)
from ray.tune.schedulers.util import _set_search_properties_backwards_compatible as scheduler_set_search_props
from ray.tune.search import (
    BasicVariantGenerator,
    SearchAlgorithm,
    SearchGenerator,
    ConcurrencyLimiter,
    Searcher,
    create_searcher,
)
from ray.tune.search.sample import Categorical
from ray.tune.search.util import _set_search_properties_backwards_compatible as searcher_set_search_props
from ray.tune.search.variant_generator import _has_unresolved_values
from ray.tune.syncer import SyncConfig, SyncerCallback
from ray.tune.experiment import Trial
from ray.tune.execution.trial_runner import TrialRunner
from ray.tune.utils.callback import _create_default_callbacks
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.queue import Empty, Queue

from hydro.tune.search import HydroBasicVariantGenerator
from hydro.tune import HydroTrialExecutor, HydroTrialRunner
from hydro.tune.progress_reporter import (
    _detect_reporter,
    _prepare_progress_reporter_for_ray_client,
    _stream_client_output,
)
from hydro.tune.analysis import ExperimentAnalysis, ResultGrid
from hydro.tune.tune_config import TuneConfig

if TYPE_CHECKING:
    from ray.train.trainer import BaseTrainer

_TRAINABLE_PKL = "trainable.pkl"
_TUNER_PKL = "tuner.pkl"
_TRAINABLE_KEY = "_trainable"
_PARAM_SPACE_KEY = "_param_space"

ClientActorHandle = Any

# The magic key that is used when instantiating Tuner during resume.
_TUNER_INTERNAL = "_tuner_internal"
_SELF = "self"

_TUNER_FAILED_MSG = (
    "The Ray Tune run failed. Please inspect the previous error messages for a "
    "cause. After fixing the issue, you can restart the run from scratch or "
    "continue this run. To continue this run, you can use "
    '`tuner = Tuner.restore("{path}")`.'
)

logger = logging.getLogger(__name__)


def _report_progress(runner: TrialRunner, reporter: ProgressReporter, done: bool = False):
    """Reports experiment progress.

    Args:
        runner: Trial runner to report on.
        reporter: Progress reporter.
        done: Whether this is the last progress report attempt.
    """
    if isinstance(runner, HydroTrialRunner):
        trials = runner.get_hydrotrials()
    else:
        trials = runner.get_trials()
    if reporter.should_report(trials, done=done):
        sched_debug_str = runner.scheduler_alg.debug_string()
        executor_debug_str = runner.trial_executor.debug_string()
        reporter.report(trials, done, sched_debug_str, executor_debug_str)


@PublicAPI
def run(
    run_or_experiment: Union[str, Callable, Type],
    *,
    scaling_num: int = 8,
    fusion_limit: Optional[Union[int, Dict]] = None,
    eager_transfer: float = 0.5,
    trial_compile: bool = False,
    name: Optional[str] = None,
    metric: Optional[str] = None,
    mode: Optional[str] = None,
    stop: Optional[Union[Mapping, Stopper, Callable[[str, Mapping], bool]]] = None,
    time_budget_s: Optional[Union[int, float, datetime.timedelta]] = None,
    config: Optional[Dict[str, Any]] = None,
    resources_per_trial: Union[None, Mapping[str, Union[float, int, Mapping]], PlacementGroupFactory] = None,
    num_samples: int = 1,
    local_dir: Optional[str] = None,
    search_alg: Optional[Union[Searcher, SearchAlgorithm, str]] = None,
    scheduler: Optional[Union[TrialScheduler, str]] = None,
    keep_checkpoints_num: Optional[int] = None,
    checkpoint_score_attr: Optional[str] = None,
    checkpoint_freq: int = 0,
    checkpoint_at_end: bool = False,
    verbose: Union[int, Verbosity] = Verbosity.V3_TRIAL_DETAILS,
    progress_reporter: Optional[ProgressReporter] = None,
    log_to_file: bool = False,
    trial_name_creator: Optional[Callable[[Trial], str]] = None,
    trial_dirname_creator: Optional[Callable[[Trial], str]] = None,
    chdir_to_trial_dir: bool = True,
    sync_config: Optional[SyncConfig] = None,
    export_formats: Optional[Sequence] = None,
    max_failures: int = 0,
    fail_fast: bool = False,
    restore: Optional[str] = None,
    server_port: Optional[int] = None,
    resume: Union[bool, str] = False,
    reuse_actors: Optional[bool] = None,
    raise_on_failed_trial: bool = True,
    callbacks: Optional[Sequence[Callback]] = None,
    max_concurrent_trials: Optional[int] = None,
    # Deprecated
    trial_executor: Optional[RayTrialExecutor] = None,
    # == internal only ==
    _experiment_checkpoint_dir: Optional[str] = None,
    _remote: Optional[bool] = None,
    # Passed by the Tuner.
    _remote_string_queue: Optional[Queue] = None,
) -> ExperimentAnalysis:
    """Executes training.
    Support both the Hydro tuning and original Ray Tune.

    When a SIGINT signal is received (e.g. through Ctrl+C), the tuning run
    will gracefully shut down and checkpoint the latest experiment state.
    Sending SIGINT again (or SIGKILL/SIGTERM instead) will skip this step.

    Many aspects of Tune, such as the frequency of global checkpointing,
    maximum pending placement group trials and the path of the result
    directory be configured through environment variables. Refer to
    :ref:`tune-env-vars` for a list of environment variables available.

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
        Ray Tune args:
        run_or_experiment: If function|class|str, this is the algorithm or
            model to train. This may refer to the name of a built-on algorithm
            (e.g. RLlib's DQN or PPO), a user-defined trainable
            function or class, or the string identifier of a
            trainable function or class registered in the tune registry.
            If Experiment, then Tune will execute training based on
            Experiment.spec. If you want to pass in a Python lambda, you
            will need to first register the function:
            ``tune.register_trainable("lambda_id", lambda x: ...)``. You can
            then use ``tune.run("lambda_id")``.
        metric: Metric to optimize. This metric should be reported
            with `tune.report()`. If set, will be passed to the search
            algorithm and scheduler.
        mode: Must be one of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute. If set, will be
            passed to the search algorithm and scheduler.
        name: Name of experiment.
        stop: Stopping criteria. If dict,
            the keys may be any field in the return result of 'train()',
            whichever is reached first. If function, it must take (trial_id,
            result) as arguments and return a boolean (True if trial should be
            stopped, False otherwise). This can also be a subclass of
            ``ray.tune.Stopper``, which allows users to implement
            custom experiment-wide stopping (i.e., stopping an entire Tune
            run based on some time constraint).
        time_budget_s: Global time budget in
            seconds after which all trials are stopped. Can also be a
            ``datetime.timedelta`` object.
        config: Algorithm-specific configuration for Tune variant
            generation (e.g. env, hyperparams). Defaults to empty dict.
            Custom search algorithms may ignore this.
        resources_per_trial: Machine resources
            to allocate per trial, e.g. ``{"cpu": 64, "gpu": 8}``.
            Note that GPUs will not be assigned unless you specify them here.
            Defaults to 1 CPU and 0 GPUs in
            ``Trainable.default_resource_request()``. This can also
            be a PlacementGroupFactory object wrapping arguments to create a
            per-trial placement group.
        num_samples: Number of times to sample from the
            hyperparameter space. Defaults to 1. If `grid_search` is
            provided as an argument, the grid will be repeated
            `num_samples` of times. If this is -1, (virtually) infinite
            samples are generated until a stopping condition is met.
        local_dir: Local dir to save training results to.
            Defaults to ``~/ray_results``.
        search_alg: Search algorithm for
            optimization. You can also use the name of the algorithm.
        scheduler: Scheduler for executing
            the experiment. Choose among FIFO (default), MedianStopping,
            AsyncHyperBand, HyperBand and PopulationBasedTraining. Refer to
            ray.tune.schedulers for more options. You can also use the
            name of the scheduler.
        keep_checkpoints_num: Number of checkpoints to keep. A value of
            `None` keeps all checkpoints. Defaults to `None`. If set, need
            to provide `checkpoint_score_attr`.
        checkpoint_score_attr: Specifies by which attribute to rank the
            best checkpoint. Default is increasing order. If attribute starts
            with `min-` it will rank attribute in decreasing order, i.e.
            `min-validation_loss`.
        checkpoint_freq: How many training iterations between
            checkpoints. A value of 0 (default) disables checkpointing.
            This has no effect when using the Functional Training API.
        checkpoint_at_end: Whether to checkpoint at the end of the
            experiment regardless of the checkpoint_freq. Default is False.
            This has no effect when using the Functional Training API.
        verbose: 0, 1, 2, or 3. Verbosity mode.
            0 = silent, 1 = only status updates, 2 = status and brief trial
            results, 3 = status and detailed trial results. Defaults to 3.
        progress_reporter: Progress reporter for reporting
            intermediate experiment progress. Defaults to CLIReporter if
            running in command-line, or JupyterNotebookReporter if running in
            a Jupyter notebook.
        log_to_file: Log stdout and stderr to files in
            Tune's trial directories. If this is `False` (default), no files
            are written. If `true`, outputs are written to `trialdir/stdout`
            and `trialdir/stderr`, respectively. If this is a single string,
            this is interpreted as a file relative to the trialdir, to which
            both streams are written. If this is a Sequence (e.g. a Tuple),
            it has to have length 2 and the elements indicate the files to
            which stdout and stderr are written, respectively.
        trial_name_creator: Optional function that takes in a Trial and returns
            its name (i.e. its string representation). Be sure to include some unique
            identifier (such as `Trial.trial_id`) in each trial's name.
        trial_dirname_creator: Optional function that takes in a trial and
            generates its trial directory name as a string. Be sure to include some
            unique identifier (such as `Trial.trial_id`) is used in each trial's
            directory name. Otherwise, trials could overwrite artifacts and checkpoints
            of other trials. The return value cannot be a path.
        chdir_to_trial_dir: Whether to change the working directory of each worker
            to its corresponding trial directory. Defaults to `True` to prevent
            contention between workers saving trial-level outputs.
            If set to `False`, files are accessible with paths relative to the
            original working directory. However, all workers on the same node now
            share the same working directory, so be sure to use
            `session.get_trial_dir()` as the path to save any outputs.
        sync_config: Configuration object for syncing. See
            tune.SyncConfig.
        export_formats: List of formats that exported at the end of
            the experiment. Default is None.
        max_failures: Try to recover a trial at least this many times.
            Ray will recover from the latest checkpoint if present.
            Setting to -1 will lead to infinite recovery retries.
            Setting to 0 will disable retries. Defaults to 0.
        fail_fast: Whether to fail upon the first error.
            If fail_fast='raise' provided, Tune will automatically
            raise the exception received by the Trainable. fail_fast='raise'
            can easily leak resources and should be used with caution (it
            is best used with `ray.init(local_mode=True)`).
        restore: Path to checkpoint. Only makes sense to set if
            running 1 trial. Defaults to None.
        server_port: Port number for launching TuneServer.
        resume: One of [True, False, "LOCAL", "REMOTE", "PROMPT", "AUTO"]. Can
            be suffixed with one or more of ["+ERRORED", "+ERRORED_ONLY",
            "+RESTART_ERRORED", "+RESTART_ERRORED_ONLY"] (e.g. ``AUTO+ERRORED``).
            "LOCAL"/True restores the checkpoint from the
            local experiment directory, determined
            by ``name`` and ``local_dir``.
            "REMOTE" restores the checkpoint
            from ``upload_dir`` (as passed to ``sync_config``).
            "PROMPT" provides the CLI feedback.
            False forces a new experiment.
            "AUTO" will attempt to resume from a checkpoint and otherwise
            start a new experiment.
            The suffix "+ERRORED" resets and reruns errored trials upon resume -
            previous trial artifacts will be left untouched. It will try to continue
            from the last observed checkpoint.
            The suffix "+RESTART_ERRORED" will instead start the errored trials from
            scratch. "+ERRORED_ONLY" and "+RESTART_ERRORED_ONLY" will disable
            resuming non-errored trials - they will be added as finished instead. New
            trials can still be generated by the search algorithm.
            If resume is set but checkpoint does not exist,
            ValueError will be thrown.
        reuse_actors: Whether to reuse actors between different trials
            when possible. This can drastically speed up experiments that start
            and stop actors often (e.g., PBT in time-multiplexing mode). This
            requires trials to have the same resource requirements.
            Defaults to ``True`` for function trainables and ``False`` for
            class and registered trainables.
        raise_on_failed_trial: Raise TuneError if there exists failed
            trial (of ERROR state) when the experiments complete.
        callbacks: List of callbacks that will be called at different
            times in the training loop. Must be instances of the
            ``ray.tune.callback.Callback`` class. If not passed,
            `LoggerCallback` and `SyncerCallback` callbacks are automatically
            added.
        max_concurrent_trials: Maximum number of trials to run
            concurrently. Must be non-negative. If None or 0, no limit will
            be applied. This is achieved by wrapping the ``search_alg`` in
            a :class:`ConcurrencyLimiter`, and thus setting this argument
            will raise an exception if the ``search_alg`` is already a
            :class:`ConcurrencyLimiter`. Defaults to None.
        _remote: Whether to run the Tune driver in a remote function.
            This is disabled automatically if a custom trial executor is
            passed in. This is enabled by default in Ray client mode.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """

    """Stage 1: Preparation"""
    remote_run_kwargs = locals().copy()
    remote_run_kwargs.pop("_remote")

    if _remote is None:
        _remote = ray.util.client.ray.is_connected()

    if _remote is True and trial_executor:
        raise ValueError("cannot use custom trial executor")
    elif trial_executor:
        warnings.warn(
            "Passing a custom `trial_executor` is deprecated and will be removed " "in the future.",
            DeprecationWarning,
        )

    if not trial_executor or isinstance(trial_executor, RayTrialExecutor):
        _ray_auto_init()

    # Determine whether to use hydro
    if scaling_num < 0:
        raise ValueError("`scaling_num` should be >= 0")
    elif scaling_num == 0:
        hydro_enable = False
    else:
        hydro_enable = True

    if fusion_limit is not None:
        if fusion_limit < 0:
            raise ValueError("`fusion_limit` should be >= 0")

    if eager_transfer <= 0 or eager_transfer > 1:
        raise ValueError("`eager_transfer` should be in (0, 1]")

    if _remote:
        remote_run = ray.remote(num_cpus=0)(run)

        # Make sure tune.run is called on the sever node.
        remote_run = _force_on_current_node(remote_run)

        progress_reporter, string_queue = _prepare_progress_reporter_for_ray_client(
            hydro_enable, progress_reporter, verbose, _remote_string_queue
        )

        # Override with detected progress reporter
        remote_run_kwargs["progress_reporter"] = progress_reporter
        remote_future = remote_run.remote(_remote=False, **remote_run_kwargs)

        _stream_client_output(
            remote_future,
            progress_reporter,
            string_queue,
        )
        return ray.get(remote_future)

    del remote_run_kwargs

    ray._private.usage.usage_lib.record_library_usage("tune")

    all_start = time.time()

    if mode and mode not in ["min", "max"]:
        raise ValueError("The `mode` parameter passed to `tune.run()` has to be one of " "['min', 'max']")

    set_verbosity(verbose)

    config = config or {}
    sync_config = sync_config or SyncConfig()
    sync_config.validate_upload_dir()

    checkpoint_score_attr = checkpoint_score_attr or ""
    if checkpoint_score_attr.startswith("min-"):
        checkpoint_score_attr = checkpoint_score_attr[4:]
        checkpoint_score_order = "min"
    else:
        checkpoint_score_order = "max"

    checkpoint_config = CheckpointConfig(
        num_to_keep=keep_checkpoints_num,
        checkpoint_score_attribute=checkpoint_score_attr,
        checkpoint_score_order=checkpoint_score_order,
        checkpoint_frequency=checkpoint_freq,
        checkpoint_at_end=checkpoint_at_end,
    )

    if num_samples == -1:
        num_samples = sys.maxsize

    result_buffer_length = None

    if scheduler is None:
        scheduler = "fifo"  # Default scheduler

    # Create scheduler here as we need access to some of its properties
    if isinstance(scheduler, str):
        scheduler = scheduler.lower()

        # importing at top level causes a recursive dependency
        if hydro_enable:
            from hydro.tune.schedule import create_scheduler
        else:
            from ray.tune.schedulers import create_scheduler

        scheduler = create_scheduler(scheduler)
    scheduler = scheduler or FIFOScheduler()

    if hydro_enable:
        # Obtain batch_size list
        search_config = copy.deepcopy(config)
        if "train_loop_config" in search_config:
            search_config = search_config["train_loop_config"]
        if "batch_size" in search_config and isinstance(search_config["batch_size"], Categorical):
            assert isinstance(search_config["batch_size"], Categorical), "batch_size must be Categorical (tune.choice)"
            batch_size_list = search_config["batch_size"].categories
        else:
            batch_size_list = None

    if not scheduler.supports_buffered_results:
        # Result buffering with e.g. a Hyperband scheduler is a bad idea, as
        # hyperband tries to stop trials when processing brackets. With result
        # buffering, we might trigger this multiple times when evaluating
        # a single trial, which leads to unexpected behavior.
        env_result_buffer_length = os.getenv("TUNE_RESULT_BUFFER_LENGTH", "")
        if env_result_buffer_length:
            warnings.warn(
                f"You are using a {type(scheduler)} scheduler, but "
                f"TUNE_RESULT_BUFFER_LENGTH is set "
                f"({env_result_buffer_length}). This can lead to undesired "
                f"and faulty behavior, so the buffer length was forcibly set "
                f"to 1 instead."
            )
        result_buffer_length = 1

    # If reuse_actors is unset, default to False for string and class trainables,
    # and default to True for everything else (i.e. function trainables)
    if reuse_actors is None:
        trainable = run_or_experiment.run_identifier if isinstance(run_or_experiment, Experiment) else run_or_experiment
        reuse_actors = (
            # Only default to True for function trainables that meet certain conditions
            is_function_trainable(trainable)
            and not (
                # Changing resources requires restarting actors
                scheduler
                and isinstance(scheduler, ResourceChangingScheduler)
            )
            and not (
                # If GPUs are requested we could run into problems with device memory
                _check_gpus_in_resources(resources_per_trial)
            )
            and not (
                # If the resource request is overridden, we don't know if GPUs
                # will be requested, yet, so default to False
                _check_default_resources_override(trainable)
            )
            and not (
                # Mixins do not work with reuse_actors as the mixin setup will only
                # be invoked once
                _check_mixin(trainable)
            )
        )
        logger.debug(f"Auto-detected `reuse_actors={reuse_actors}`")

    if isinstance(scheduler, (PopulationBasedTraining, PopulationBasedTrainingReplay)) and not reuse_actors:
        warnings.warn(
            "Consider boosting PBT performance by enabling `reuse_actors` as "
            "well as implementing `reset_config` for Trainable."
        )

    # trial_executor = trial_executor or RayTrialExecutor(reuse_actors=reuse_actors, result_buffer_length=result_buffer_length)
    assert trial_executor is None
    if hydro_enable:
        trial_executor = HydroTrialExecutor(reuse_actors=reuse_actors, result_buffer_length=result_buffer_length)
    else:
        trial_executor = RayTrialExecutor(reuse_actors=reuse_actors, result_buffer_length=result_buffer_length)

    if isinstance(run_or_experiment, list):
        experiments = run_or_experiment
    else:
        experiments = [run_or_experiment]

    for i, exp in enumerate(experiments):
        if not isinstance(exp, Experiment):
            experiments[i] = Experiment(
                name=name,
                run=exp,
                stop=stop,
                time_budget_s=time_budget_s,
                config=config,
                resources_per_trial=resources_per_trial,
                num_samples=num_samples,
                local_dir=local_dir,
                _experiment_checkpoint_dir=_experiment_checkpoint_dir,
                sync_config=sync_config,
                checkpoint_config=checkpoint_config,
                trial_name_creator=trial_name_creator,
                trial_dirname_creator=trial_dirname_creator,
                log_to_file=log_to_file,
                export_formats=export_formats,
                max_failures=max_failures,
                restore=restore,
            )
    else:
        logger.debug("Ignoring some parameters passed into tune.run.")

    if fail_fast and max_failures != 0:
        raise ValueError("max_failures must be 0 if fail_fast=True.")

    if isinstance(search_alg, str):
        search_alg = create_searcher(search_alg)
        if hydro_enable:
            raise NotImplementedError("Hydro currently does not support more complex search algorithms.")

    # if local_mode=True is set during ray.init().
    is_local_mode = ray._private.worker._mode() == ray._private.worker.LOCAL_MODE

    if is_local_mode:
        max_concurrent_trials = 1

    if not search_alg:
        if hydro_enable:
            search_alg = HydroBasicVariantGenerator(max_concurrent=max_concurrent_trials or 0)
        else:
            search_alg = BasicVariantGenerator(max_concurrent=max_concurrent_trials or 0)
    elif max_concurrent_trials or is_local_mode:
        if isinstance(search_alg, ConcurrencyLimiter):
            if not is_local_mode:
                if search_alg.max_concurrent != max_concurrent_trials:
                    raise ValueError(
                        "You have specified `max_concurrent_trials="
                        f"{max_concurrent_trials}`, but the `search_alg` is "
                        "already a `ConcurrencyLimiter` with `max_concurrent="
                        f"{search_alg.max_concurrent}. FIX THIS by setting "
                        "`max_concurrent_trials=None`."
                    )
                else:
                    logger.warning(
                        "You have specified `max_concurrent_trials="
                        f"{max_concurrent_trials}`, but the `search_alg` is "
                        "already a `ConcurrencyLimiter`. "
                        "`max_concurrent_trials` will be ignored."
                    )
        else:
            if max_concurrent_trials < 1:
                raise ValueError("`max_concurrent_trials` must be greater or equal than 1, " f"got {max_concurrent_trials}.")
            if isinstance(search_alg, Searcher):
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent_trials)
            elif not is_local_mode:
                logger.warning(
                    "You have passed a `SearchGenerator` instance as the "
                    "`search_alg`, but `max_concurrent_trials` requires a "
                    "`Searcher` instance`. `max_concurrent_trials` "
                    "will be ignored."
                )

    if isinstance(search_alg, Searcher):
        search_alg = SearchGenerator(search_alg)

    if config and not searcher_set_search_props(
        search_alg.set_search_properties,
        metric,
        mode,
        config,
        **experiments[0].public_spec,
    ):
        if _has_unresolved_values(config):
            raise ValueError(
                "You passed a `config` parameter to `tune.run()` with "
                "unresolved parameters, but the search algorithm was already "
                "instantiated with a search space. Make sure that `config` "
                "does not contain any more parameter definitions - include "
                "them in the search algorithm's search space if necessary."
            )

    if not scheduler_set_search_props(scheduler.set_search_properties, metric, mode, **experiments[0].public_spec):
        raise ValueError(
            "You passed a `metric` or `mode` argument to `tune.run()`, but "
            "the scheduler you are using was already instantiated with their "
            "own `metric` and `mode` parameters. Either remove the arguments "
            "from your scheduler or from your call to `tune.run()`"
        )

    progress_metrics = _detect_progress_metrics(_get_trainable(run_or_experiment))

    # Create syncer callbacks
    callbacks = _create_default_callbacks(callbacks, sync_config, metric=metric, progress_metrics=progress_metrics)

    # User Warning for GPUs
    if ray.cluster_resources().get("GPU", 0):
        if _check_gpus_in_resources(resources=resources_per_trial):
            # "gpu" is manually set.
            pass
        elif _check_default_resources_override(experiments[0].run_identifier):
            # "default_resources" is manually overridden.
            pass
        else:
            logger.warning(
                "Tune detects GPUs, but no trials are using GPUs. "
                "To enable trials to use GPUs, wrap `train_func` with "
                "`tune.with_resources(train_func, resources_per_trial={'gpu': 1})` "
                "which allows Tune to expose 1 GPU to each trial. "
                "For Ray AIR Trainers, you can specify GPU resources "
                "through `ScalingConfig(use_gpu=True)`. "
                "You can also override "
                "`Trainable.default_resource_request` if using the "
                "Trainable API."
            )

    experiment_interrupted_event = _setup_signal_catching()

    progress_reporter = progress_reporter or _detect_reporter(hydro_enable=hydro_enable)

    if hydro_enable:
        runner = HydroTrialRunner(
            search_alg=search_alg,
            scheduler=scheduler,
            local_checkpoint_dir=experiments[0].checkpoint_dir,
            sync_config=sync_config,
            stopper=experiments[0].stopper,
            resume=resume,
            server_port=server_port,
            fail_fast=fail_fast,
            trial_executor=trial_executor,
            callbacks=callbacks,
            metric=metric,
            batch_size_list=batch_size_list,
            scaling_num=scaling_num,
            trial_compile=trial_compile,
            fusion_limit=fusion_limit,
            eager_transfer_num=int(eager_transfer * num_samples),
            mode=mode,
            # Driver should only sync trial checkpoints if
            # checkpoints are not synced to cloud
            driver_sync_trial_checkpoints=not bool(sync_config.upload_dir),
        )
    else:
        runner = TrialRunner(
            search_alg=search_alg,
            scheduler=scheduler,
            local_checkpoint_dir=experiments[0].checkpoint_dir,
            sync_config=sync_config,
            stopper=experiments[0].stopper,
            resume=resume,
            server_port=server_port,
            fail_fast=fail_fast,
            trial_executor=trial_executor,
            callbacks=callbacks,
            metric=metric,
            # Driver should only sync trial checkpoints if
            # checkpoints are not synced to cloud
            driver_sync_trial_checkpoints=not bool(sync_config.upload_dir),
        )

    if not runner.resumed:
        for exp in experiments:
            search_alg.add_configurations([exp])
    else:
        logger.info("TrialRunner resumed, ignoring new add_experiment but " "updating trial resources.")
        if resources_per_trial:
            runner.update_pending_trial_resources(resources_per_trial)

    # Calls setup on callbacks
    runner.setup_experiments(experiments=experiments, total_num_samples=search_alg.total_samples)

    """Stage 2: Start Tuning"""
    tune_start = time.time()

    progress_reporter.setup(
        start_time=tune_start,
        total_samples=search_alg.total_samples,
        metric=metric,
        mode=mode,
    )

    # Enabling profiling to get the fusion limit
    if fusion_limit is None:
        assert len(experiments) == 1

        if batch_size_list is None:
            max_profile_trials = 3  # fusion=1 + fusion=2
        else:
            max_profile_trials = len(batch_size_list) * 3

        exp_prof = Experiment(
            name=name,
            run=run_or_experiment,
            stop={"training_iteration": 1},
            time_budget_s=600,  # Maximum 10 minutes for profiling by default
            config=config,
            resources_per_trial=resources_per_trial,
            num_samples=max_profile_trials,
            local_dir=local_dir,
            _experiment_checkpoint_dir=_experiment_checkpoint_dir,
            sync_config=sync_config,
            checkpoint_config=checkpoint_config,
            trial_name_creator=trial_name_creator,
            trial_dirname_creator=trial_dirname_creator,
            log_to_file=log_to_file,
            export_formats=export_formats,
            max_failures=max_failures,
            restore=restore,
        )
        search_alg_prof = HydroBasicVariantGenerator()
        search_alg_prof.add_configurations([exp_prof])

        runner_prof = HydroTrialRunner(
            profile_stage=True,
            search_alg=search_alg_prof,
            scheduler=scheduler,
            local_checkpoint_dir=experiments[0].checkpoint_dir,
            sync_config=sync_config,
            stopper=exp_prof.stopper,
            resume=resume,
            server_port=server_port,
            fail_fast=fail_fast,
            trial_executor=trial_executor,
            callbacks=callbacks,
            metric=metric,
            batch_size_list=batch_size_list,
            scaling_num=scaling_num,
            trial_compile=trial_compile,
            fusion_limit=fusion_limit,
            eager_transfer_num=int(eager_transfer * num_samples),
            mode=mode,
            # Driver should only sync trial checkpoints if
            # checkpoints are not synced to cloud
            driver_sync_trial_checkpoints=not bool(sync_config.upload_dir),
        )

        while not runner_prof.is_finished() and not experiment_interrupted_event.is_set():
            runner_prof.step()
            if has_verbosity(Verbosity.V1_EXPERIMENT):
                _report_progress(runner_prof, progress_reporter)
        profile_taken = time.time() - tune_start

        runner_prof.planer.parse_memory_record()
        runner.fusion_limit = runner_prof.planer.plan
        if has_verbosity(Verbosity.V1_EXPERIMENT):
            print(f"Total profiling time: {profile_taken:.2f} seconds. Obtain fusion limit: {runner_prof.planer.plan}")
        runner_prof.cleanup()
        time.sleep(3)  # wait for trial termination

    while not runner.is_finished() and not experiment_interrupted_event.is_set():
        runner.step()
        if has_verbosity(Verbosity.V1_EXPERIMENT):
            _report_progress(runner, progress_reporter)
    tune_taken = time.time() - tune_start

    try:
        runner.checkpoint(force=True)
        # Wait for the final remote directory sync to finish before exiting
        if runner._syncer:
            runner._syncer.wait()
    except Exception as e:
        logger.warning(f"Trial Runner checkpointing failed: {str(e)}")

    if has_verbosity(Verbosity.V1_EXPERIMENT):
        _report_progress(runner, progress_reporter, done=True)

    if hydro_enable:
        all_trials = runner.get_hydrotrials()
    else:
        all_trials = runner.get_trials()
    experiment_checkpoint = runner.checkpoint_file

    # Wait for syncing to finish
    for callback in callbacks:
        if isinstance(callback, SyncerCallback):
            try:
                callback.wait_for_all()
            except TuneError as e:
                logger.error(e)

    runner.cleanup()

    incomplete_trials = []
    for trial in all_trials:
        if trial.status != Trial.TERMINATED:
            incomplete_trials += [trial]

    if incomplete_trials:
        if raise_on_failed_trial and not experiment_interrupted_event.is_set():
            raise TuneError("Trials did not complete", incomplete_trials)
        else:
            logger.error("Trials did not complete: %s", incomplete_trials)

    """Stage 3: Return Results"""
    all_taken = time.time() - all_start
    if has_verbosity(Verbosity.V1_EXPERIMENT):
        logger.info(f"Total run time: {all_taken:.2f} seconds " f"({tune_taken:.2f} seconds for the tuning loop).")

    if experiment_interrupted_event.is_set():
        logger.warning(
            "Experiment has been interrupted, but the most recent state was "
            "saved. You can continue running this experiment by passing "
            "`resume=True` to `tune.run()`"
        )

    return ExperimentAnalysis(
        experiment_checkpoint,
        trials=all_trials,
        default_metric=metric,
        default_mode=mode,
        sync_config=sync_config,
        hydro_enable=hydro_enable,
    )


class HydroTunerInternal(TunerInternal):
    """The real implementation behind external facing ``Tuner``.

    The external facing ``Tuner`` multiplexes between local Tuner and remote Tuner
    depending on whether in Ray client mode.

    In Ray client mode, external ``Tuner`` wraps ``TunerInternal`` into a remote actor,
    which is guaranteed to be placed on head node.

    ``TunerInternal`` can be constructed from fresh, in which case, ``trainable`` needs
    to be provided, together with optional ``param_space``, ``tune_config`` and
    ``run_config``.

    It can also be restored from a previous failed run (given ``restore_path``).

    Args:
        restore_path: The path from where the Tuner can be restored. If provided, None
            of the rest args are needed.
        resume_config: Resume config to configure which trials to continue.
        trainable: The trainable to be tuned.
        param_space: Search space of the tuning job.
            One thing to note is that both preprocessor and dataset can be tuned here.
        tune_config: Tuning algorithm specific configs.
            Refer to ray.tune.tune_config.TuneConfig for more info.
        run_config: Runtime configuration that is specific to individual trials.
            If passed, this will overwrite the run config passed to the Trainer,
            if applicable. Refer to ray.air.config.RunConfig for more info.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self) -> ResultGrid:
        # trainable = self._convert_trainable(self._trainable)
        trainable = self.converted_trainable
        assert self._experiment_checkpoint_dir
        if not self._is_restored:
            param_space = copy.deepcopy(self._param_space)
            analysis = self._fit_internal(trainable, param_space)
        else:
            analysis = self._fit_resume(trainable)

        self._experiment_analysis = analysis

        return ResultGrid(self._experiment_analysis)

    def _fit_internal(self, trainable, param_space) -> ExperimentAnalysis:
        """Fitting for a fresh Tuner."""
        args = {
            **self._get_tune_run_arguments(trainable),
            **dict(
                run_or_experiment=trainable,
                config={**param_space},
                num_samples=self._tune_config.num_samples,
                search_alg=self._tune_config.search_alg,
                scheduler=self._tune_config.scheduler,
                name=self._run_config.name,
                log_to_file=self._run_config.log_to_file,
                scaling_num=self._tune_config.scaling_num,
                fusion_limit=self._tune_config.fusion_limit,
                eager_transfer=self._tune_config.eager_transfer,
                trial_compile=self._tune_config.trial_compile,
            ),
            **self._tuner_kwargs,
        }
        analysis = run(
            **args,
        )
        self.clear_remote_string_queue()
        return analysis

    def _fit_resume(self, trainable) -> ExperimentAnalysis:
        """Fitting for a restored Tuner."""
        if self._missing_params_error_message:
            raise ValueError(self._missing_params_error_message)

        resume = "AUTO"

        if self._resume_config:
            if not self._resume_config.resume_unfinished:
                if self._resume_config.resume_errored:
                    resume += "+ERRORED_ONLY"
                elif self._resume_config.restart_errored:
                    resume += "+RESTART_ERRORED_ONLY"
            else:
                if self._resume_config.resume_errored:
                    resume += "+ERRORED"
                elif self._resume_config.restart_errored:
                    resume += "+RESTART_ERRORED"

        args = {
            **self._get_tune_run_arguments(trainable),
            **dict(
                run_or_experiment=trainable,
                resume=resume,
                search_alg=self._tune_config.search_alg,
                scheduler=self._tune_config.scheduler,
            ),
            **self._tuner_kwargs,
        }
        analysis = run(**args)
        self.clear_remote_string_queue()
        return analysis


@PublicAPI(stability="beta")
class HydroTuner:
    """Tuner is the recommended way of launching hyperparameter tuning jobs with Ray Tune.

    Args:
        trainable: The trainable to be tuned.
        param_space: Search space of the tuning job.
            One thing to note is that both preprocessor and dataset can be tuned here.
        tune_config: Tuning algorithm specific configs.
            Refer to ray.tune.tune_config.TuneConfig for more info.
        run_config: Runtime configuration that is specific to individual trials.
            If passed, this will overwrite the run config passed to the Trainer,
            if applicable. Refer to ray.air.config.RunConfig for more info.
    """

    # One of the following is assigned.
    _local_tuner: Optional[HydroTunerInternal]  # Only used in none ray client mode.
    _remote_tuner: Optional[ClientActorHandle]  # Only used in ray client mode.

    def __init__(
        self,
        trainable: Optional[Union[str, Callable, Type[Trainable], "BaseTrainer"]] = None,
        *,
        param_space: Optional[Dict[str, Any]] = None,
        tune_config: Optional[TuneConfig] = None,
        run_config: Optional[RunConfig] = None,
        # This is internal only arg.
        # Only for dogfooding purposes. We can slowly promote these args
        # to RunConfig or TuneConfig as needed.
        # TODO(xwjiang): Remove this later.
        _tuner_kwargs: Optional[Dict] = None,
        _tuner_internal: Optional[TunerInternal] = None,
    ):
        """Configure and construct a tune run."""
        kwargs = locals().copy()
        self._is_ray_client = ray.util.client.ray.is_connected()
        if _tuner_internal:
            if not self._is_ray_client:
                self._local_tuner = kwargs[_TUNER_INTERNAL]
            else:
                self._remote_tuner = kwargs[_TUNER_INTERNAL]
        else:
            kwargs.pop(_TUNER_INTERNAL, None)
            kwargs.pop(_SELF, None)
            if not self._is_ray_client:
                self._local_tuner = HydroTunerInternal(**kwargs)
            else:
                self._remote_tuner = _force_on_current_node(ray.remote(num_cpus=0)(HydroTunerInternal)).remote(**kwargs)

    @classmethod
    def restore(
        cls,
        path: str,
        trainable: Optional[Union[str, Callable, Type[Trainable], "BaseTrainer"]] = None,
        resume_unfinished: bool = True,
        resume_errored: bool = False,
        restart_errored: bool = False,
    ) -> "Tuner":
        """Restores Tuner after a previously failed run.

        All trials from the existing run will be added to the result table. The
        argument flags control how existing but unfinished or errored trials are
        resumed.

        Finished trials are always added to the overview table. They will not be
        resumed.

        Unfinished trials can be controlled with the ``resume_unfinished`` flag.
        If ``True`` (default), they will be continued. If ``False``, they will
        be added as terminated trials (even if they were only created and never
        trained).

        Errored trials can be controlled with the ``resume_errored`` and
        ``restart_errored`` flags. The former will resume errored trials from
        their latest checkpoints. The latter will restart errored trials from
        scratch and prevent loading their last checkpoints.

        Args:
            path: The path where the previous failed run is checkpointed.
                This information could be easily located near the end of the
                console output of previous run.
                Note: depending on whether ray client mode is used or not,
                this path may or may not exist on your local machine.
            trainable: The trainable to use upon resuming the experiment.
                This should be the same trainable that was used to initialize
                the original Tuner.
                NOTE: Starting in 2.5, this will be a required parameter.
            resume_unfinished: If True, will continue to run unfinished trials.
            resume_errored: If True, will re-schedule errored trials and try to
                restore from their latest checkpoints.
            restart_errored: If True, will re-schedule errored trials but force
                restarting them from scratch (no checkpoint will be loaded).

        """
        # TODO(xwjiang): Add some comments to clarify the config behavior across
        #  retored runs.
        #  For example, is callbacks supposed to be automatically applied
        #  when a Tuner is restored and fit again?

        if not trainable:
            warning_message = (
                "Passing in the experiment's `trainable` will be a required argument "
                "to `Tuner.restore` starting from version 2.5. "
                "Please specify the trainable to avoid this warning."
            )
            warnings.warn(warning_message)

        resume_config = _ResumeConfig(
            resume_unfinished=resume_unfinished,
            resume_errored=resume_errored,
            restart_errored=restart_errored,
        )

        if not ray.util.client.ray.is_connected():
            tuner_internal = HydroTunerInternal(
                restore_path=path,
                resume_config=resume_config,
                trainable=trainable,
            )
            return Tuner(_tuner_internal=tuner_internal)
        else:
            tuner_internal = _force_on_current_node(ray.remote(num_cpus=0)(HydroTunerInternal)).remote(
                restore_path=path,
                resume_config=resume_config,
                trainable=trainable,
            )
            return Tuner(_tuner_internal=tuner_internal)

    def _prepare_remote_tuner_for_jupyter_progress_reporting(self):
        run_config: RunConfig = ray.get(self._remote_tuner.get_run_config.remote())
        progress_reporter, string_queue = _prepare_progress_reporter_for_ray_client(
            hydro_enable=False, progress_reporter=run_config.progress_reporter, verbosity=run_config.verbose
        )
        run_config.progress_reporter = progress_reporter
        ray.get(self._remote_tuner.set_run_config_and_remote_string_queue.remote(run_config, string_queue))

        return progress_reporter, string_queue

    def fit(self) -> ResultGrid:
        """Executes hyperparameter tuning job as configured and returns result.

        Failure handling:
        For the kind of exception that happens during the execution of a trial,
        one may inspect it together with stacktrace through the returned result grid.
        See ``ResultGrid`` for reference. Each trial may fail up to a certain number.
        This is configured by ``RunConfig.FailureConfig.max_failures``.

        Exception that happens beyond trials will be thrown by this method as well.
        In such cases, there will be instruction like the following printed out
        at the end of console output to inform users on how to resume.

        Please use tuner = Tuner.restore("~/ray_results/tuner_resume")
        to resume.

        Raises:
            RayTaskError: If user-provided trainable raises an exception
            TuneError: General Ray Tune error.
        """

        if not self._is_ray_client:
            try:
                return self._local_tuner.fit()
            except TuneError as e:
                raise TuneError(_TUNER_FAILED_MSG.format(path=self._local_tuner.get_experiment_checkpoint_dir())) from e
        else:
            experiment_checkpoint_dir = ray.get(self._remote_tuner.get_experiment_checkpoint_dir.remote())
            (
                progress_reporter,
                string_queue,
            ) = self._prepare_remote_tuner_for_jupyter_progress_reporting()
            try:
                fit_future = self._remote_tuner.fit.remote()
                _stream_client_output(
                    fit_future,
                    progress_reporter,
                    string_queue,
                )
                return ray.get(fit_future)
            except TuneError as e:
                raise TuneError(_TUNER_FAILED_MSG.format(path=experiment_checkpoint_dir)) from e

    def get_results(self) -> ResultGrid:
        """Get results of a hyperparameter tuning run.

        This method returns the same results as :meth:`fit() <ray.tune.tuner.Tuner.fit>`
        and can be used to retrieve the results after restoring a tuner without
        calling ``fit()`` again.

        If the tuner has not been fit before, an error will be raised.

        .. code-block:: python

            from ray.tune import Tuner

            tuner = Tuner.restore("/path/to/experiment')
            results = tuner.get_results()

        Returns:
            Result grid of a previously fitted tuning run.

        """
        if not self._is_ray_client:
            return self._local_tuner.get_results()
        else:
            (
                progress_reporter,
                string_queue,
            ) = self._prepare_remote_tuner_for_jupyter_progress_reporting()
            fit_future = self._remote_tuner.fit.remote()
            _stream_client_output(
                fit_future,
                progress_reporter,
                string_queue,
            )
            return ray.get(fit_future)
