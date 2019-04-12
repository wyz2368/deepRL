import os
from attackgraph import json_op as jp

def nn_param():
    param = {}
    param['num_hidden'] = 256
    param['num_layers'] = 1
    param['lr'] = 5e-5
    param['total_timesteps'] = 1000
    param['exploration_fraction'] = 0.5
    param['exploration_final_eps'] = 0.03
    param['print_freq'] = 250
    param['param_noise'] = False
    param['gamma'] = 0.99
    param['prioritized_replay'] = True
    param['checkpoint_freq'] = 30000

    param_path = os.getcwd() + '/network_parameters/param.json'
    jp.save_json_data(param_path, param)
    print("Network parameters have been saved in a json file successfully.")


nn_param()