---
hide:
  - toc
  - navigation
  - footer
---
<div align="center">
<img src="assets/logo-dark.png#only-dark" width="60%" alt="Hydro logo" style="margin-bottom: 1em">
<img src="assets/logo-light.png#only-light" width="60%" alt="Hydro logo" style="margin-bottom: 1em">
<h1>Surrogate-based Hyperparameter Tuning System</h1>
</div>


Hydro is a system that automatically applies the hyperparameter transfer theory together with multiple system techniques to jointly improve the tuning efficiency.

🚀 **Efficient Tuning**. Hydro scales down the model size and fuses multiple trials can significantly improve training throughput and hardware efficiency.

✨ **Automatic Pipeline**. Hydro streamlines the surrogate model generation process and seamlessly integrates with Ray, offering a user-friendly solution for tuning.

🎉 **Quality Maintenance**. Hydro typically can well maintain the tuned model quality even using a scaled and fused model for tuning.


!!! success "Getting Started"

    + [**Tutorial: Walkthrough of a Hydro Tuning Example**](getting_started/quick_start.md)

    + **More Examples**:
        + [**`vision`: Image Classification Example**](https://github.com/S-Lab-System-Group/Hydro/tree/master/examples/vision)
        + [**`language`: Language Modeling Example**](https://github.com/S-Lab-System-Group/Hydro/tree/master/examples/language)

??? question "Need More Support?"

    If you have any question or suggestion on improving Hydro, please [let us know](https://github.com/S-Lab-System-Group/Hydro/issues/new/choose).


## Acknowledgements

Hydro is built upon many fabulous open-source repositories, including

<img src="docs/assets/external/ray.ico" alt="Image" width="15" height="15">[ray](https://github.com/ray-project/ray) | [mup](https://github.com/microsoft/mup) | [hfta](https://github.com/UofT-EcoSystem/hfta) | <img src="docs/assets/external/pytorch-icon.svg" alt="Image" width="15" height="15">[pytorch](https://github.com/pytorch/pytorch) | <img src="docs/assets/external/hf-logo.svg" alt="Image" width="15" height="15">[
transformers](https://github.com/huggingface/transformers)