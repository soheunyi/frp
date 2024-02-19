# Filter, Rank, and Prune (FRP)
## Introduction
This is the code for the paper "Learning Linear Cyclic Gaussian Graphical Models:
Filter, Rank, and Prune"
## Installation
This code is written in `Python`. To use it, you need to install packages introduced in `environment.yml` via `conda`. To do this, 
```bash
conda env create --file environment.yml
```
## Usage
To run the code, you need to activate the environment first.
```bash
conda activate frp
```
Then, you can look at `intro.ipynb` for a quick start.
To run the tests, run the following in the main directory (`code`).
```bash
pytest
```