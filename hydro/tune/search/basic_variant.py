from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings
import numpy as np

from ray.tune.error import TuneError
from ray.tune.experiment.config_parser import _make_parser, _create_trial_from_spec
from ray.tune.experiment import _convert_to_experiment_list

from ray.tune.search.sample import np_random_generator, _BackwardsCompatibleNumpyRng
from ray.tune.search.variant_generator import (
    _count_variants,
    _count_spec_samples,
    generate_variants,
    format_vars,
    _flatten_resolved_vars,
    _get_preset_variants,
)
from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.search.basic_variant import BasicVariantGenerator, _VariantIterator, _TrialIterator


from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
from ray.util import PublicAPI

if TYPE_CHECKING:
    from ray.tune.experiment import Experiment

SERIALIZATION_THRESHOLD = 1e6


class _HydroTrialIterator(_TrialIterator):
    """Generates trials from the spec.

    Args:
        uuid_prefix: Used in creating the trial name.
        num_samples: Number of samples from distribution
             (same as tune.TuneConfig).
        unresolved_spec: Experiment specification
            that might have unresolved distributions.
        constant_grid_search: Should random variables be sampled
            first before iterating over grid variants (True) or not (False).
        output_path: A specific output path within the local_dir.
        points_to_evaluate: Configurations that will be tried out without sampling.
        lazy_eval: Whether variants should be generated
            lazily or eagerly. This is toggled depending
            on the size of the grid search.
        start: index at which to start counting trials.
        random_state (int | np.random.Generator | np.random.RandomState):
            Seed or numpy random generator to use for reproducible results.
            If None (default), will use the global numpy random generator
            (``np.random``). Please note that full reproducibility cannot
            be guaranteed in a distributed enviroment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __next__(self):
        """Generates Trial objects with the variant generation process.

        Uses a fixed point iteration to resolve variants. All trials
        should be able to be generated at once.

        See also: `ray.tune.search.variant_generator`.

        Returns:
            Trial object
        """

        if "run" not in self.unresolved_spec:
            raise TuneError("Must specify `run` in {}".format(self.unresolved_spec))

        if self.variants and self.variants.has_next():
            # This block will be skipped upon instantiation.
            # `variants` will be set later after the first loop.
            resolved_vars, spec = next(self.variants)
            return self.create_trial(resolved_vars, spec)

        if self.points_to_evaluate:
            config = self.points_to_evaluate.pop(0)
            self.num_samples_left -= 1
            self.variants = _VariantIterator(
                _get_preset_variants(
                    self.unresolved_spec,
                    config,
                    constant_grid_search=self.constant_grid_search,
                    random_state=self.random_state,
                ),
                lazy_eval=self.lazy_eval,
            )
            resolved_vars, spec = next(self.variants)
            return self.create_trial(resolved_vars, spec)
        elif self.num_samples_left > 0:
            self.variants = _VariantIterator(
                generate_variants(
                    self.unresolved_spec,
                    constant_grid_search=self.constant_grid_search,
                    random_state=self.random_state,
                ),
                lazy_eval=self.lazy_eval,
            )
            self.num_samples_left -= 1
            resolved_vars, spec = next(self.variants)
            return self.create_trial(resolved_vars, spec)
        else:
            raise StopIteration

    def __iter__(self):
        return self


@PublicAPI
class HydroBasicVariantGenerator(BasicVariantGenerator):
    """Uses Tune's variant generation for resolving variables.

    This is the default search algorithm used if no other search algorithm
    is specified.


    Args:
        points_to_evaluate: Initial parameter suggestions to be run
            first. This is for when you already have some good parameters
            you want to run first to help the algorithm make better suggestions
            for future parameters. Needs to be a list of dicts containing the
            configurations.
        max_concurrent: Maximum number of concurrently running trials.
            If 0 (default), no maximum is enforced.
        constant_grid_search: If this is set to ``True``, Ray Tune will
            *first* try to sample random values and keep them constant over
            grid search parameters. If this is set to ``False`` (default),
            Ray Tune will sample new random parameters in each grid search
            condition.
        random_state:
            Seed or numpy random generator to use for reproducible results.
            If None (default), will use the global numpy random generator
            (``np.random``). Please note that full reproducibility cannot
            be guaranteed in a distributed environment.


    Example:

    .. code-block:: python

        from ray import tune

        # This will automatically use the `BasicVariantGenerator`
        tuner = tune.Tuner(
            lambda config: config["a"] + config["b"],
            tune_config=tune.TuneConfig(
                num_samples=4
            ),
            param_space={
                "a": tune.grid_search([1, 2]),
                "b": tune.randint(0, 3)
            },
        )
        tuner.fit()

    In the example above, 8 trials will be generated: For each sample
    (``4``), each of the grid search variants for ``a`` will be sampled
    once. The ``b`` parameter will be sampled randomly.

    The generator accepts a pre-set list of points that should be evaluated.
    The points will replace the first samples of each experiment passed to
    the ``BasicVariantGenerator``.

    Each point will replace one sample of the specified ``num_samples``. If
    grid search variables are overwritten with the values specified in the
    presets, the number of samples will thus be reduced.

    Example:

    .. code-block:: python

        from ray import tune
        from ray.tune.search.basic_variant import BasicVariantGenerator

        tuner = tune.Tuner(
            lambda config: config["a"] + config["b"],
            tune_config=tune.TuneConfig(
                search_alg=BasicVariantGenerator(points_to_evaluate=[
                    {"a": 2, "b": 2},
                    {"a": 1},
                    {"b": 2}
                ]),
                num_samples=4
            ),
            param_space={
                "a": tune.grid_search([1, 2]),
                "b": tune.randint(0, 3)
            },
        )
        tuner.fit()

    The example above will produce six trials via four samples:

    - The first sample will produce one trial with ``a=2`` and ``b=2``.
    - The second sample will produce one trial with ``a=1`` and ``b`` sampled
      randomly
    - The third sample will produce two trials, one for each grid search
      value of ``a``. It will be ``b=2`` for both of these trials.
    - The fourth sample will produce two trials, one for each grid search
      value of ``a``. ``b`` will be sampled randomly and independently for
      both of these trials.

    """

    CKPT_FILE_TMPL = "basic-variant-state-{}.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def add_configurations(self, experiments: Union["Experiment", List["Experiment"], Dict[str, Dict]]):
    #     """Chains generator given experiment specifications.

    #     Arguments:
    #         experiments: Experiments to run.
    #     """

    #     experiment_list = _convert_to_experiment_list(experiments)

    #     for experiment in experiment_list:
    #         grid_vals = _count_spec_samples(experiment.spec, num_samples=1)
    #         lazy_eval = grid_vals > SERIALIZATION_THRESHOLD
    #         if lazy_eval:
    #             warnings.warn(
    #                 f"The number of pre-generated samples ({grid_vals}) "
    #                 "exceeds the serialization threshold "
    #                 f"({int(SERIALIZATION_THRESHOLD)}). Resume ability is "
    #                 "disabled. To fix this, reduce the number of "
    #                 "dimensions/size of the provided grid search."
    #             )

    #         previous_samples = self._total_samples
    #         points_to_evaluate = copy.deepcopy(self._points_to_evaluate)
    #         self._total_samples += _count_variants(experiment.spec, points_to_evaluate)
    #         iterator = _HydroTrialIterator(
    #             uuid_prefix=self._uuid_prefix,
    #             num_samples=experiment.spec.get("num_samples", 1),
    #             unresolved_spec=experiment.spec,
    #             constant_grid_search=self._constant_grid_search,
    #             output_path=experiment.dir_name,
    #             points_to_evaluate=points_to_evaluate,
    #             lazy_eval=lazy_eval,
    #             start=previous_samples,
    #             random_state=self._random_state,
    #         )
    #         self._iterators.append(iterator)
    #         self._trial_generator = itertools.chain(self._trial_generator, iterator)
