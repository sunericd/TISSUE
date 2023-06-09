# TISSUE
TISSUE (Transcript Imputation with Spatial Single-cell Uncertainty Estimation) provides tools for estimating well-calibrated uncertainty measures for gene expression predictions in single-cell spatial transcriptomics datasets and utilizing them in downstream analyses

For a Quick Start guide, please refer to ``` tutorial.ipynb ```. It should take less than 5 minutes to run all analyses in the tutorial notebook with a normal desktop/laptop setup.

![plot](./pipeline.png)



## Installation and setup

Complete installation (including of dependencies) in a new Conda environment should take less than 10 minutes on a normal desktop/laptop setup (Windows, Mac OSX, Linux).

### Future Option: PyPI

This will be an option for installing the final version of the software (TBD). Install the package through PyPI with ```pip```. We recommend setting up a conda environment (or another virtual environment) first since ```tissue-bio``` currently relies on specific versions for its dependencies (although it should generally work for other environment versions, but this hasn't been thoroughly tested):

```
conda create -n myenv python=3.8
conda activate myenv

pip install tissue-bio
```



### Current Option: Local installation

The current way to install the package along with associated test and tutorial files is to clone the directory and then install the requirements for using the package. To do this, first clone the repository using git (you can install git following the instructions [here](https://github.com/git-guides/install-git)):

```
git clone https://github.com/sunericd/TISSUE.git
```

We recommend setting up a conda environment to install the requirements for the package (instructions for installing conda and what conda environment can do can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)). Installation of requirements can then be done with the following commands:

```
conda create -n tissue python=3.8
conda activate tissue

cd TISSUE
pip install -r requirements.txt
```

To keep the requirements light, we have only included packages that are necessary for the core functionalities of TISSUE. For additional utilities such as gene prediction with Tangram, please install those packages separately.

To test that the installation is working correctly, you can use the Jupyter notebook ```tutorial.ipynb``` (requires installing Jupyter, instructions found [here](https://jupyter.org/install), and adding the conda environment we just created to the Jupyter notebook kernels, instructions found [here](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)).


For Jupyter notebooks and Python scripts associated with our original publication, please refer to REPO LINK TBD
If you find this code useful, please cite the following paper:

CITATION TBD

