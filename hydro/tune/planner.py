from typing import Any, List, Dict, Set, Mapping, Optional, Union, Tuple

import logging
import os
import time
import math
import traceback
import subprocess
import copy
import ast
from xml.dom import minidom

from ray.tune.experiment import Trial

MAX_DEBUG_TRIALS = 20

logger = logging.getLogger(__name__)


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
        self.total_gpu_mem = None
        self.memory_record = {}  # For unified batch size
        self.memory_record_bs_s = {}
        self.memory_record_bs_d = {}
        # self.search_alg = BasicVariantGenerator(max_concurrent_trials=1)
        # self.trial_executor = trial_executor

        # Init plan
        if self.batch_size_list is not None:
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

    def report_memory_usage(self, trial_id: str, used_mem: int, total_mem: int):
        prof_tag, trial_tag = trial_id.split("_")[-2], trial_id.split("_")[-1]
        assert prof_tag == "prof"
        if self.total_gpu_mem is None:
            self.total_gpu_mem = total_mem

        if len(trial_tag) > 1:
            batch_size = ast.literal_eval(trial_tag[1:])
            trial_tag = trial_tag[0]
            if trial_tag == "s":
                self.memory_record_bs_s[batch_size] = used_mem
            elif trial_tag == "d":
                self.memory_record_bs_d[batch_size] = used_mem
            else:
                raise ValueError(f"Unknown trial tag: {trial_tag}")
        else:
            self.memory_record[trial_tag] = used_mem

    def obtain_memory_bounded_plan(self):
        # if self.batch_size_list is None:
        #     assert self.plan == -1  # Not modified yet

        s, d = self.memory_record["s"], self.memory_record["d"]
        self.plan = math.floor((self.total_gpu_mem - s) / (d - s))
        assert self.plan > 1

    def obtain_memory_bounded_plan_per_batch(self):
        for batch_size in self.batch_size_list:
            # assert self.plan[batch_size] == -1, f"Got plan: {self.plan}"  # Not modified yet
            s, d = self.memory_record_bs_s[batch_size], self.memory_record_bs_d[batch_size]
            self.plan[batch_size] = math.floor((self.total_gpu_mem - s) / (d - s))
            assert (
                self.plan[batch_size] > 1
            ), f"Got plan: {self.plan}. GPU memory of terminated trials are not released correctly.Please try to set `num_workers` to 1 of dataloader or use manually set `fusion_limit`."

    def parse_memory_record(self):
        """Parse the memory record to determine the fusion plan."""
        if len(self.memory_record) == 2:
            self.obtain_memory_bounded_plan()
        elif len(self.memory_record) == 0 and len(self.memory_record_bs_s) > 0 and len(self.memory_record_bs_d) > 0:
            self.obtain_memory_bounded_plan_per_batch()
        else:
            raise ValueError("Incomplete memory record.")

        return self.plan

    # def set_plan_automatically(self, trial, strategy: str = "default", with_check: bool = False):
    #     """Determine the fusion plan for a HydroTrial based on profiling."""

    #     # NOTE: Currently only profile one trial with largest batch size to avoid GPU OOM.
    #     if self.batch_size_list is not None:
    #         largest_batch_size = max(self.batch_size_list)
    #         self.update_trial_batch_size(trial, largest_batch_size)

    #     trial_s = HydroTrial(
    #         [
    #             trial,
    #         ],
    #         hydro_id="prof_s",
    #         scaling_num=self.scaling_num,
    #     )
    #     trial_d = HydroTrial([trial, trial], hydro_id="prof_d", scaling_num=self.scaling_num)

    #     return self.set_plan_manually(10)

    # TODO: support cood check
