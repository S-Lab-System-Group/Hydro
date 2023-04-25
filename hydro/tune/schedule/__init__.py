import inspect

from hydro.tune.schedule.asha import ASHAHydroScheduler
from hydro.tune.schedule.fifo import FIFOHydroScheduler
from ray.util import PublicAPI
from ray._private.utils import get_function_args

# Values in this dictionary will be one two kinds:
#    class of the scheduler object to create
#    wrapper function to support a lazy import of the scheduler class
SCHEDULER_IMPORT = {
    "fifo": FIFOHydroScheduler,
    "asha": ASHAHydroScheduler,
}


@PublicAPI(stability="beta")
def create_scheduler(
    scheduler, **kwargs,
):
    """Instantiate a scheduler based on the given string.

    This is useful for swapping between different schedulers.

    Args:
        scheduler: The scheduler to use.
        **kwargs: Scheduler parameters.
            These keyword arguments will be passed to the initialization
            function of the chosen scheduler.
    Returns:
        ray.tune.schedulers.trial_scheduler.TrialScheduler: The scheduler.
    Example:
        >>> from ray import tune
        >>> pbt_kwargs = {}
        >>> scheduler = tune.create_scheduler('pbt', **pbt_kwargs) # doctest: +SKIP
    """

    scheduler = scheduler.lower()
    if scheduler not in SCHEDULER_IMPORT:
        raise ValueError(f"The `scheduler` argument must be one of " f"{list(SCHEDULER_IMPORT)}. " f"Got: {scheduler}")

    SchedulerClass = SCHEDULER_IMPORT[scheduler]

    if inspect.isfunction(SchedulerClass):
        # invoke the wrapper function to retrieve class
        SchedulerClass = SchedulerClass()

    scheduler_args = get_function_args(SchedulerClass)
    trimmed_kwargs = {k: v for k, v in kwargs.items() if k in scheduler_args}

    return SchedulerClass(**trimmed_kwargs)
