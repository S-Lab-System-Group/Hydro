from typing import Any, List, Dict, Set, Mapping, Optional, Union, Tuple

import click
from datetime import datetime
from dataclasses import dataclass
import json
import logging
import os
import time
import traceback
import warnings
import copy

import ray
from ray.air._internal.checkpoint_manager import CheckpointStorage
from ray.exceptions import RayTaskError
from ray.tune.error import _TuneStopTrialError
from ray.tune.impl.out_of_band_serialize_dataset import out_of_band_serialize_dataset
from ray.util import get_node_ip_address
from ray.tune import TuneError
from ray.tune.callback import CallbackList, Callback
from ray.tune.experiment import Experiment
from ray.tune.execution.trial_runner import TrialRunner, TrialRunnerWrapper, _ExperimentCheckpointManager, _TrialExecutorWrapper
from ray.tune.execution.insufficient_resources_manager import _InsufficientResourcesManager
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
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
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

MAX_DEBUG_TRIALS = 20

logger = logging.getLogger(__name__)


import json
import os
import subprocess
import sys
import time
import traceback

from xml.dom import minidom


def smi_getter(smi_list, gpu_id):
    metrics_output_dir = "./"
    cmd = f"nvidia-smi -q -x -i {gpu_id}".split()
    while True:
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, encoding="utf-8")
            smi_output = p.stdout.read()
        except Exception:
            traceback.print_exc()
            break
        output = parse_nvidia_smi_result(smi_output, metrics_output_dir, gpu_id)
        smi_list.extend(output)
        # TODO: change to sleep time configurable via arguments
        time.sleep(0.2)


def parse_nvidia_smi_result(smi, outputDir, gpu_id):
    try:
        old_umask = os.umask(0)
        xmldoc = minidom.parseString(smi)
        gpuList = xmldoc.getElementsByTagName("gpu")
        gpuInfo = []
        outPut = {}
        outPut["Timestamp"] = time.asctime(time.localtime())
        for gpuIndex, gpu in enumerate(gpuList):
            outPut["index"] = gpu_id[gpuIndex]
            outPut["gpuUtil"] = (
                gpu.getElementsByTagName("utilization")[0]
                .getElementsByTagName("gpu_util")[0]
                .childNodes[0]
                .data.replace("%", "")
                .strip()
            )
            outPut["gpuMemUtil"] = (
                gpu.getElementsByTagName("utilization")[0]
                .getElementsByTagName("memory_util")[0]
                .childNodes[0]
                .data.replace("%", "")
                .strip()
            )
            outPut["gpuMem"] = gpu.getElementsByTagName("fb_memory_usage")[0].getElementsByTagName("used")[0].childNodes[0].data

            gpuInfo.append(outPut.copy())
        return gpuInfo

    except Exception as error:
        # e_info = sys.exc_info()
        print("gpu_metrics_collector error: %s" % error)
    finally:
        os.umask(old_umask)


class HydroPlanner:
    """Determine the fusion limit for each HydroTrial."""

    def __init__(
        self,
        batch_size_list: Optional[List],
        scaling_num,
        # trial_executor: HydroTrialExecutor,
    ):
        self.batch_size_list = batch_size_list
        self.scaling_num = scaling_num
        # self.search_alg = BasicVariantGenerator(max_concurrent_trials=1)
        # self.trial_executor = trial_executor

        # Init plan
        if not self.batch_size_list:
            self.plan = {batch_size: -1 for batch_size in self.batch_size_list}
        else:
            self.plan = -1

    def set_plan_manually(self, fusion_limit: Union[int, Dict]):
        """Determine the fusion plan for a HydroTrial based on user provided setting."""
        if isinstance(fusion_limit, int):
            if self.batch_size_list is None:
                self.plan = fusion_limit
            else:
                self.plan = {batch_size: fusion_limit for batch_size in self.batch_size_list}

        if isinstance(fusion_limit, dict):
            assert len(self.batch_size_list) == len(fusion_limit)

            self.plan = {batch_size: fusion_limit[batch_size] for batch_size in self.batch_size_list}

        return self.plan

    def update_trial_batch_size(self, trial: Trial, largest_batch_size: int):
        config = copy.deepcopy(trial.config)
        if "train_loop_config" in config:
            config = config["train_loop_config"]
        assert "batch_size" in config
        config["batch_size"] = largest_batch_size
        trial.config = {"train_loop_config": config}

    def set_plan_automatically(self, trial, strategy: str = "default", with_check: bool = False):
        """Determine the fusion plan for a HydroTrial based on profiling."""

        print(f"TONY1111: {trial}")
        # NOTE: Currently only profile one trial with largest batch size to avoid GPU OOM.
        if self.batch_size_list is not None:
            largest_batch_size = max(self.batch_size_list)
            self.update_trial_batch_size(trial, largest_batch_size)

        trial_s = HydroTrial(
            [
                trial,
            ],
            hydro_id="prof_s",
            scaling_num=self.scaling_num,
        )
        trial_d = HydroTrial([trial, trial], hydro_id="prof_d", scaling_num=self.scaling_num)

        return self.set_plan_manually(10)

        # self.trial_executor.start_trial(trial_s)
        # self.trial_executor.stop_trial(trial_s)

        # self.trial_executor.start_trial(trial_d)
        # self.trial_executor.stop_trial(trial_d)

        # TODO: support cood check
        pass
