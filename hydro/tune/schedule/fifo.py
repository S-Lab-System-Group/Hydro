from typing import Dict, Optional
import logging

from ray.util.annotations import PublicAPI
from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from hydro.tune import HydroTrialRunner, HydroTrial


logger = logging.getLogger(__name__)


@PublicAPI
class FIFOHydroScheduler(FIFOScheduler):
    """Simple scheduler that just runs trials in submission order."""

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def choose_trial_to_run(self, trial_runner: HydroTrialRunner) -> Optional[HydroTrial]:
        for trial in trial_runner.get_hydrotrials():
            if trial.status == HydroTrial.PENDING and trial_runner.trial_executor.has_resources_for_trial(trial):
                return trial
        for trial in trial_runner.get_hydrotrials():
            if trial.status == HydroTrial.PAUSED and trial_runner.trial_executor.has_resources_for_trial(trial):
                return trial
        return None
