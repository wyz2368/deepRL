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



def run(HADO=False, load_env=None, env_name=None):

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
    game = game_data.Game_data(env, num_layers=4, num_hidden=256, hiddens=[256,256],num_episodes=400, threshold=0.1)
    game.set_hado_param(param=(4, 0.7, 0.286))


    # uniform strategy has been produced ahead of time
    epoch = env.epoch
    epoch += 1

    act_att = 'att_str_epoch1.pkl'
    act_def = 'def_str_epoch1.pkl'

    game.add_att_str(act_att)
    game.add_def_str(act_def)

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
        #TODO: set flag to env
        #TODO: length of str_set and mixed strategy does not match.
        nn_att = "att_str_epoch" + str(epoch) + ".pkl"
        nn_def = mix_str_def
        a_BD, _ = parallel_sim.parallel_sim(env, game, nn_att, nn_def, game.num_episodes)
        nn_att = mix_str_att
        nn_def = "def_str_epoch" + str(epoch) + ".pkl"
        _, d_BD = parallel_sim.parallel_sim(env, game, nn_att, nn_def, game.num_episodes)

        game.def_str.append("def_str_epoch" + str(epoch) + ".pkl")
        game.att_str.append("att_str_epoch" + str(epoch) + ".pkl")


        #TODO: This may lead to early stop.
        if a_BD - aPayoff < game.threshold and d_BD - dPayoff < game.threshold:
            print("*************************")
            print("aPayoff=", aPayoff, " ", "dPayoff=", dPayoff)
            print("a_BD=", a_BD, " ", "d_BD=", d_BD)
            print("*************************")
            break


        # simulate and extend the payoff matrix.
        sim_Series.sim_and_modifiy_Series_with_game(game)

        # find nash equilibrium using gambit analysis
        payoffmatrix_def = game.payoffmatrix_def
        payoffmatrix_att = game.payoffmatrix_att
        nash_att, nash_def = ga.do_gambit_analysis(payoffmatrix_def, payoffmatrix_att)
        ga.add_new_NE(game, nash_att, nash_def, epoch)
        fp.save_pkl(game, game_path)












