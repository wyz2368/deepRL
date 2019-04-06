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



def initialize(load_env=None, env_name=None):

    # Create Environment
    if isinstance(load_env,str):
        path = os.getcwd() + load_env + '.pkl'
        if not fp.isExist(path):
            raise ValueError("The env being loaded does not exist.")
        env = fp.load_pkl(path)
    else:
        # env is created and saved.
        env = dag.env_rand_gen_and_save(env_name)

    # save graph copy
    env.save_graph_copy()

    # create players and point to their env
    env.create_players()
    env.defender.myenv = env
    env.attacker.myenv = env

    # initialize game data
    game = game_data.Game_data(env, num_layers=4, num_hidden=256, hiddens=[256,256],num_episodes=100, threshold=0.1)
    game.set_hado_param(param=(4, 0.7, 0.286))


    # uniform strategy has been produced ahead of time
    epoch = env.epoch
    epoch += 1

    act_att = 'att_str_epoch1.pkl'
    act_def = 'def_str_epoch1.pkl'

    game.add_att_str(act_att)
    game.add_def_str(act_def)

    # simulate using random strategies and initialize payoff matrix
    t1 = time.time()
    aReward, dReward = parallel_sim.parallel_sim(env, game, act_att, act_def, game.num_episodes)
    print("Time for uniform sim:",time.time()-t1)
    game.init_payoffmatrix(dReward, aReward)
    ne = {}
    ne[0] = np.array([1], dtype=np.float32)
    ne[1] = np.array([1], dtype=np.float32)
    game.add_nasheq(epoch, ne)

    # save a copy of game data
    game_path = os.getcwd() + '/game_data/game.pkl'
    fp.save_pkl(game, game_path)

    return env, game

def DO_EGTA(env, game, epoch = 1, game_path = os.getcwd() + '/game_data/game.pkl'):
    #TODO: check length of str_set mismatch
    print("=======================================================")
    print("===============Begin Running DO-EGTA===================")
    print("=======================================================")
    while True:

        # fix opponent strategy
        mix_str_def = game.nasheq[epoch][0]
        mix_str_att = game.nasheq[epoch][1]
        aPayoff, dPayoff = util.payoff_mixed_NE(game, epoch)

        # increase epoch
        epoch += 1
        print("Current epoch is " + str(epoch))

        # train and save RL agents
        print("Begin training attacker......")
        training.training_att(game, mix_str_def, epoch)
        print("Attacker training done......")
        print("Begin training defender......")
        training.training_def(game, mix_str_att, epoch)
        print("Defender training done......")

        # Judge beneficial deviation
        # one plays nn and another plays ne strategy
        print("Simulating attacker payoff. New strategy vs. mixed opponent strategy.")
        nn_att = "att_str_epoch" + str(epoch) + ".pkl"
        nn_def = mix_str_def
        a_BD, _ = parallel_sim.parallel_sim(env, game, nn_att, nn_def, game.num_episodes)
        print("Simulation done.")

        print("Simulating defender's payoff. New strategy vs. mixed opponent strategy.")
        nn_att = mix_str_att
        nn_def = "def_str_epoch" + str(epoch) + ".pkl"
        _, d_BD = parallel_sim.parallel_sim(env, game, nn_att, nn_def, game.num_episodes)
        print("Simulation done.")

        #TODO: This may lead to early stop.
        if a_BD - aPayoff < game.threshold and d_BD - dPayoff < game.threshold:
            print("*************************")
            print("aPayoff=", aPayoff, " ", "dPayoff=", dPayoff)
            print("a_BD=", a_BD, " ", "d_BD=", d_BD)
            print("*************************")
            break

        game.add_att_str("att_str_epoch" + str(epoch) + ".pkl")
        game.add_def_str("def_str_epoch" + str(epoch) + ".pkl")

        # simulate and extend the payoff matrix.
        print("Begin extending payoff matrix.")
        sim_Series.sim_and_modifiy_Series_with_game(game)
        print("Extension finished.")

        # find nash equilibrium using gambit analysis
        payoffmatrix_def = game.payoffmatrix_def
        payoffmatrix_att = game.payoffmatrix_att
        print("Begin Gambit analysis.")
        nash_att, nash_def = ga.do_gambit_analysis(payoffmatrix_def, payoffmatrix_att)
        ga.add_new_NE(game, nash_att, nash_def, epoch)
        fp.save_pkl(game, game_path)
        print("Round_" + str(epoch) + " has done and game was saved.")
        print("=======================================================")



if __name__ == '__main__':
    env, game = initialize(env_name='test_env')









