# Examples


## `vision`: Image Classification Example

- [run_hydro.py](vision/run_hydro.py)

    Tuning ResNet-18 on CIFAR-10 dataset using Hydro.

- [run_ray.py](vision/run_ray.py)

    The original Ray Tune script for reference.

## `language`: Language Modeling Example

- [run_hydro_lm.py](language/run_hydro_lm.py)

    Tuning HuggingFace GPT-2 on WikiText dataset using Hydro. To be compatible with most machines, we set `n_layer=2` by default.

- [run_ray_lm.py](language/run_ray_lm.py)

    The original Ray Tune script for reference.
