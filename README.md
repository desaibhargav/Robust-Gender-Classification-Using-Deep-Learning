# Robust-Gender-Classification-Using-Deep-Learning
This repository has all our code and data for the paper (under review) titled, "An End-to-End Model for Robust One Step Voice Based Gender Classification in Real-Time" 


In addition to releasing a single instance (CPU) code and a Jupyter Notebook, we have also released a GCP compatible (multiple GPU) compatible code for training on Google ML Engine (GCP)


Below is a brief description of each file that is upoloaded in this repository:

1. hypertuning.py --> Assumes a .csv format of data (provided in the Google Drive link) with it's file path is available and allows the user to play around with the hyperparameters. On cloning/downloading, the parameters default to the ones proposed in the paper (under review)

2. task_og.py  --> Assumes the raw data format (provided in the Google Drive link) with it's file path is available and is the original file used for the project 

3. task_dist.py --> Assumes a .csv format (provided in the Google Drive link) with it's file path is available and is the original file used for training on a distributed framework. 

4. /glcloud  --> Contains the setup, config and code files for training on Google ML Engine (GPU). To train on Google ML Engine, a good starting point is here: https://cloud.google.com/sdk/gcloud/reference. Note: A GCP account is a must. 



Original Dataset:

Voxforge.org, . "Free Speech... Recognition (Linux, Windows and Mac) - voxforge.org." http://www.voxforge.org/.

Dataset used in the paper: 

1.            (Raw audio files)

2.            (.csv format)



Demonstration of our project: 

https://drive.google.com/open?id=1Pog1rNw5j4glnLm2HXeDhYjXAmvAaCEw


Audio used in demonstration: 

https://drive.google.com/open?id=1Z3zP7uOK0-VMVPKxRV_zVE2pMLMcPIUl





