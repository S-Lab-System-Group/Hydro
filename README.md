<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
  <img alt="Hydro logo" width="60%" src="docs/assets/logo.png">
</picture>
<h1>Surrogate-based Hyperparameter Tuning System</h1>
</div>

[![Documentation](https://custom-icon-badges.demolab.com/badge/Documentation-blue.svg?logo=repo)](https://s-lab-system-group.github.io/Hydro/)
[![Build](https://custom-icon-badges.demolab.com/github/actions/workflow/status/DenverCoder1/custom-icon-badges/ci.yml?branch=main&logo=check-circle-fill&logoColor=white)](https://github.com/S-Lab-System-Group/Hydro/actions)
[![License](https://custom-icon-badges.herokuapp.com/github/license/S-Lab-System-Group/Hydro?logo=law)](https://opensource.org/licenses/Apache-2.0)


Hydro is a system that automatically applies the hyperparameter transfer theory together with multiple system techniques to jointly improve the tuning efficiency. To learn more about how Hydro works, please refer our [paper](https://www.usenix.org/conference/osdi23/presentation/hu).

We highlight three key features of Hydro:

🚀 **Efficient Tuning**. Hydro scales down the model size and fuses multiple trials can significantly improve training throughput and hardware efficiency.

✨ **Automatic Pipeline**. Hydro streamlines the surrogate model generation process and seamlessly integrates with Ray, offering a user-friendly solution for tuning.

🎉 **Quality Maintenance**. Hydro typically can well maintain the tuned model quality even using a scaled and fused model for tuning.

## Getting Started

Refer to [Getting started](https://s-lab-system-group.github.io/Hydro/) for complete instructions on environment setup, installation, and integration.



## Installation

```
pip install hydro-tune
```

### Docker Image

We also provide a Docker image fully equipped with all dependencies and environments.

```sh
docker pull tonyhao96/hydro
```
## Examples

We provide working examples for end-to-end hyperparameter tuning inside [`examples`](examples) folder.

+ `vision`: Image Classification Example

+ `language`: Language Modeling Example


## Artifact Evaluation
Please check `osdi23-artifact` branch for the artifact evaluation version of Hydro.


## Citation
If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{Hydro,
  author    = {Qinghao Hu and Zhisheng Ye and Meng Zhang and Qiaoling Chen and Peng Sun and Yonggang Wen and Tianwei Zhang},
  title     = {Hydro: {Surrogate-Based} Hyperparameter Tuning Service in Datacenters},
  booktitle = {17th USENIX Symposium on Operating Systems Design and Implementation},
  year      = {2023},
  publisher = {USENIX Association},
  series    = {OSDI '23}
}
```

## Acknowledgements

Hydro is built upon many fabulous open-source repositories, including

<img src="docs/assets/external/ray.ico" alt="Image" width="15" height="15">[ray](https://github.com/ray-project/ray) | [mup](https://github.com/microsoft/mup) | [hfta](https://github.com/UofT-EcoSystem/hfta) | <img src="docs/assets/external/pytorch-icon.svg" alt="Image" width="15" height="15">[pytorch](https://github.com/pytorch/pytorch) | <img src="docs/assets/external/hf-logo.svg" alt="Image" width="15" height="15">[
transformers](https://github.com/huggingface/transformers)
