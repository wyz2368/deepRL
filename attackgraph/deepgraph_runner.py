# Packages import
import tensorflow as tf
import numpy as np
import os
import time

# Modules import
from attackgraph import DagGenerator as dag
from attackgraph import attacker,defender
from attackgraph import file_op as fp
from attackgraph import json_op as jp
from attackgraph import sim_Series, parallel_sim
from attackgraph import subproc
from attackgraph import training
from attackgraph import util

from baselines.deepq import deepq
from baselines.deepq import load_action


def run(new_env=None, env_name=None):
    if isinstance(new_env,str):
        path = os.getcwd() + new_env + '.pkl'
        if not fp.isExist(path):
            raise ValueError("The env named does not exist.")
        env = fp.load_pkl(path)
    else:
        env = dag.env_rand_gen_and_save(env_name)















