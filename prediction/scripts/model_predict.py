#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys, os, re, time
import random
import timeit
import argparse
from configparser import ConfigParser
import pdb, traceback
import pickle
import gzip
from itertools import *
# Science
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

import torch

# Import Hierarchical latent var model
# Add path
sys.path.append('../src/prediction')

# Import Generative model src
from poisson_with_skipped_cycles_models import *

# And other helpful functions
from evaluation_utils import *

the_model = None
def load_model(model_dir):
    global the_model
    the_model = torch.load(model_dir)

def predict(data, model_name="generative_poisson_with_skipped_cycles"):
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # the_model = torch.load("/Users/xiyuan/Desktop/Baseline/period_prediction_api/prediction/scripts/model.pt")

    data = np.array([data])

    ###### Model config from file
    # Type and parameters of the generative model
    model_config = ConfigParser()
    model_config.read('./generative_model_config/{}'.format(model_name))

    # Model prediction MC type
    model_predict_MC_samples = model_config.get(
        'model_prediction_criterion', 'MC_samples',
        fallback='per_individual'
    )
    # Default is M samples for all, sample_size=(1,M)
    model_predict_M = (
        1,
        model_config.getint(
            'model_prediction_criterion', 'M',
            fallback=1000
        )
    )
    # Number of skipped cycles used for prediction:
    s_predict = model_config.getfloat(
        'model_prediction_criterion', 's_predict',
        fallback=100
    )

    # max x (cycle length) - for predictive posterior
    x_predict_max = model_config.getint(
        'model_prediction_criterion', 'x_predict_max',
        fallback=x_predict_max_default
    )

    if model_predict_MC_samples == 'per_individual':
        # M samples per individual, sample_size=(I,M)
        model_predict_M = (
            data.shape[0],
            model_config.getint(
                'model_prediction_criterion', 'M',
                fallback=1000
            )
        )

    my_model_predictions = the_model.predict(
        data,
        s_predict=s_predict,
        M=model_predict_M,
        x_predict_max=x_predict_max,
        posterior_type='mean',  # Only mean predictions
        day_range=np.arange(0, 30)  # First 30 days
    )
    print(list(my_model_predictions['mean'][0]))
    return list(my_model_predictions['mean'][0])

if __name__ == '__main__':
    load_model("./model.pt")
    predict([27, 25, 30, 26, 28, 30, 25, 27, 29, 29])
