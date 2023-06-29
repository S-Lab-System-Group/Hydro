# **Installation**

We provide three ways to install Hydro, including `pip`, `docker` and `git`. 

!!! tip "Tip"

    We recommend create a new conda environment to install Hydro.
    ```
    conda create -n hydro python=3.9
    ```

### **with pip** <small>recommended</small> { #with-pip data-toc-label="with pip" }

Hydro is published as a Python package and can be installed with `pip` directly.


=== "pip"
    ``` sh
    pip install hydro-tune
    ```


### **with docker**

The official Docker image of Hydro is a great way to get up and running in a few minutes, as it comes with all dependencies pre-installed. 

=== "docker"

    ```
    docker pull tonyhao96/hydro
    ```
Here is an example command of running Hydro in a docker container:
```sh
docker run --name hydro --shm-size=16g --gpus all -it tonyhao96/hydro
```

The source code and example are inside `/workspace/Hydro`.

### **with git**

If you want to use the very latest version, Hydro can be directly cloned from GitHub.

=== "git"

  ```
  git clone https://github.com/S-Lab-System-Group/Hydro.git
  ```

Then you can install with `pip` in the root directory of Hydro.

```
pip install -e Hydro
```