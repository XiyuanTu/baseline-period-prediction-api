#!/usr/bin/env python
# coding: utf-8

import sys
import torch.optim as optim
from configparser import ConfigParser
# Import Generative model src
from poisson_with_skipped_cycles_models import *

def save_model(data_dir, model_name="generative_poisson_with_skipped_cycles"):
    def cast_dict_values(dict, dtype):
        '''
            Input:
                dict: dictionary for which to cast values
                dtype: what data-type to cast values to
            Output:
                dict: dictionary with casted values
        '''
        return {key: dtype(value) for (key, value) in dict.items()}

    #### Data handling
    # Open file
    with open('{}/cycle_lengths.npz'.format(data_dir), 'rb') as f:
        # Load
        loaded_data=np.load(f, allow_pickle=True)
        # Dataset details
        I=loaded_data['I']
        C=loaded_data['C']
        X=loaded_data['cycle_lengths']
    
    ###### Model config from file
    # Type and parameters of the generative model
    model_config = ConfigParser()
    model_config.read('./{}'.format(model_name))
    
    try:
        ### Create generative model object, with given parameters
        my_model=getattr(
                        sys.modules[__name__],
                        model_config.get('generative_model','model_name')
                    )(
                        **cast_dict_values(model_config._sections['model_params'], float),
                        config_file=model_name,
                    )
        
        #######################
        ### Model config
        # Model fit criterion
        model_fit_criterion = model_config.get(
                                'model_fitting_criterion', 'criterion'
                            ) #sampling_criterion
        # Model fit MC type
        model_fit_MC_samples = model_config.get(
                                'model_fitting_criterion', 'MC_samples',
                                fallback='per_individual'
                            )
        if model_fit_MC_samples == 'per_individual':
            # M samples per individual, sample_size=(I,M)
            model_fit_M = (
                            X.shape[0],
                            model_config.getint(
                                'model_fitting_criterion', 'M',
                                fallback=1000
                            )
                        )
        elif model_fit_MC_samples == 'per_cohort':
            # M samples for all, sample_size=(1,M)
            model_fit_M = (
                            1,
                            model_config.getint(
                                'model_fitting_criterion',
                                'M',
                                fallback=1000
                            )
                        )
        else:
            raise ValueError('Fitting MC sampling type {} not implemented yet'.format(model_fit_MC_samples))
        
        # Model optimizer
        model_optimizer = getattr(
                                optim,
                                model_config.get('model_optimizer', 'optimizer')
                            )(
                                my_model.parameters(),
                                lr=model_config.getfloat(
                                    'model_optimizer', 'learning_rate'
                                    )
                            )
        other_fitting_args=cast_dict_values(
                        model_config._sections['model_fitting_params'],
                        float
                    )
        
       
        #######################
        
        ###### Model fitting, given train data 
        my_model.fit(X,
                        optimizer=model_optimizer,
                        criterion=model_fit_criterion,
                        M=model_fit_M,
                        **other_fitting_args
                        )
        
        torch.save(my_model, "model.pt")
        
    except Exception as error:
        print('Could not create {} model with error {}'.format(model_name, error))

if __name__ == '__main__':
    save_model("../")