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
from attackgraph import game_data
from attackgraph import sample_strategy as ss
from attackgraph import gambit_analysis as ga

from baselines.deepq import deepq
from baselines.deepq import load_action



#TODO: check when to
def run(epsilon, HADO=False, load_env=None, env_name=None):

    # Create Environment
    if isinstance(load_env,str):
        path = os.getcwd() + load_env + '.pkl'
        if not fp.isExist(path):
            raise ValueError("The env being loaded does not exist.")
        env = fp.load_pkl(path)
    else:
        # env is created and saved.
        env = dag.env_rand_gen_and_save(env_name)

    # create players and point to their env
    env.create_players()
    env.defender.myenv = env
    env.attacker.myenv = env

    # initialize game data
    game = game_data.Game_data(env, num_layers=4, num_hidden=256, hiddens=[256,256],num_episodes=400)
    game.set_hado_param(param=(4, 0.7, 0.286))

    # save a copy of game data


    # initialize random strategy
    # TODO: whether to use uniform strategy
    epoch = env.epoch
    epoch += 1
    ss.rand_att_str_generator(env, game)
    ss.rand_def_str_generator(env, game)
    game.add_att_str('att_str_epoch1.pkl')
    game.add_def_str('def_str_epoch1.pkl')

    act_att = 'att_str_epoch1.pkl'
    act_def = 'def_str_epoch1.pkl'

    # simulate using random strategies and initialize payoff matrix
    aReward, dReward = parallel_sim.parallel_sim(env, game, act_att, act_def, game.num_episodes)
    game.init_payoffmatrix(dReward, aReward)
    ne = {}
    ne[0] = np.array([1], dtype=np.float32)
    ne[1] = np.array([1], dtype=np.float32)
    game.add_nasheq(epoch, ne)

    # save a copy of game data
    game_path = os.getcwd() + '/game_data/game.pkl'
    fp.save_pkl(game, game_path)

    # set flags for both players to indicate if RL finds beneficial
    def_BD_flag = True
    att_BD_flag = True

    # DO-EGTA
    while True:

        # fix opponent strategy
        mix_str_def = game.nasheq[epoch][0]
        mix_str_att = game.nasheq[epoch][1]

        # increase epoch
        epoch += 1

        # train and save RL agents
        training.training_att(game, mix_str_def, epoch)
        training.training_def(game, mix_str_att, epoch)

        # Judge beneficial deviation
        aPayoff, dPayoff = util.payoff_mixed_NE(game, epoch)
        # one plays nn and another plays ne strategy


        # simulate and extend the payoff matrix.
        sim_Series.sim_and_modifiy_Series_with_game(game)

        # find nash equilibrium using gambit analysis
        payoffmatrix_def = game.payoffmatrix_def
        payoffmatrix_att = game.payoffmatrix_att
        nash_att, nash_def = ga.do_gambit_analysis(payoffmatrix_def, payoffmatrix_att)
        ga.add_new_NE(game, nash_att, nash_def, epoch)
        fp.save_pkl(game, game_path)












