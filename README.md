# Decadal Wind Speed Recovery Amplifies Vegetation Growth
codes for "Decadal Wind Speed Recovery Amplifies Vegetation Growth"

## Contents
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Package Tests](#package-tests)
- [Demo](#demo)
- [Data Verification](#data-verification)


# System Requirements
## Hardware Requirements
The environment requires only a standard computer with enough RAM to support the operations defined by a user. We use a computer with the following specs:

RAM: 16+ GB  
CPU: Intel® Core™ i7-10750H @ 2.60GHz, 6 cores / 12 threads

## Software Requirements

### OS Requirements

This project was developed using:

-Python 3.11.9 (via conda-forge)

-IPython 8.24.0

-OS: Windows 10/11 64-bit  |  Version 22H2 

-Architecture: AMD64

The codes should be compatible with Windows, Mac, and Linux operating systems.

# Installation Guide
First, download anaconda or miniconda

```
https://www.anaconda.com/download
```
which should install in a few minutes.
After finish downloading, open Anaconda Powershell Prompt or  Anaconda Prompt.
Then create a new environment:
```
conda create -n myenv python=3.11
```
We use python version = 3.11.9, when create the new environment, the version should be more than 3.9.
You can use whatever Python IDE you like, here we use spyder:
```
conda install spyder
```
### Package Installation
Then make sure the following Python libraries are installed: numpy, pandas, xarray, matplotlib, pwlf, cartopy，rpy2, geopandas, salem, R package relaimpo. You can use the following order:
```
pip install numpy pandas xarray matplotlib pwlf cartopy geopandas salem rpy2
```
For fig 4, you also need to download R software, here we use: R version 4.4.2 (2024-10-31 ucrt) -- "Pile of Leaves"
Copyright (C) 2024 The R Foundation for Statistical Computing
Platform: x86_64-w64-mingw32/x64
install relaimpo in R:
```
install.packages("relaimpo")
```
After install relaimpo in R and conda install rpy2, you can import rpy2 in Spyder.
The packages install in conda and R should finish in a few minutes.
# Package tests
Run test.py


# Demo
Run Fig1~Fig4.py cells one by one, and the path should be replaced by requirements.

The LAI data can download from: https://doi.org/10.5281/zenodo.7649107

The wind data need permission to download from: https://zenodo.org/records/5624401

The "LAIdataproduce.py" and "WINDdataproduce.py"  contain the preprocessing of the raw data, while "Fig1~Fig4.py" include data processing and visualization for the figures in the paper.

PS: the path in code: ('G:/CN05.1/month/CN05.1_Win_1982_2020_yearly_025x025.nc') should replaced by the nc file in wind year data. 

LAI data used in the code should run LAIdataproduce.py first.

The contents of the "quhua.zip" are the divisions of different vegetation zones in China, used for the data analysis in Fig. 3, which should replace the path **shp_file = 'path/vegzone_alb54.shp**

The "eleven.zip" contains the administrative divisions of China, which is used for **def china_map_feature()** in codes.

After completing the previous steps, Figures 1 through 4 should be visualized in Spyder within a few seconds.


# Data Verification
You can verification the LAI dealing process by visualization of 'G:/CN05.1/month/0.25_month_onlychina.nc' in LAI data process.






