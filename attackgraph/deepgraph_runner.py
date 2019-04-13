# Packages import
import tensorflow as tf
import numpy as np
import os
import time
import sys

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
from attackgraph.simulation import series_sim
from attackgraph.sim_MPI import do_MPI_sim

from baselines.deepq import deepq
from baselines.deepq.load_action import load_action_class




def initialize(load_env=None, env_name=None, MPI_flag = False):

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
    env.create_action_space()

    # initialize game data
    game = game_data.Game_data(env, num_layers=4, num_hidden=256, hiddens=[256,256],num_episodes=10, threshold=0.1)
    game.set_hado_param(param=(4, 0.7, 0.286))
    game.set_hado_time_step(700000)
    game.env.defender.set_env_belong_to(game.env)
    game.env.attacker.set_env_belong_to(game.env)

    # uniform strategy has been produced ahead of time
    epoch = env.epoch
    epoch += 1

    act_att = 'att_str_epoch1.pkl'
    act_def = 'def_str_epoch1.pkl'

    game.add_att_str(act_att)
    game.add_def_str(act_def)

    # simulate using random strategies and initialize payoff matrix
    t1 = time.time()
    # aReward, dReward = parallel_sim.parallel_sim(env, game, act_att, act_def, game.num_episodes)
    if MPI_flag:
        aReward, dReward = do_MPI_sim(act_att, act_def)
    else:
        aReward, dReward = series_sim(env, game, act_att, act_def, game.num_episodes)
    # aReward, dReward = -10, -10
    print("Time for uniform sim:",time.time()-t1)
    game.init_payoffmatrix(dReward, aReward)
    ne = {}
    ne[0] = np.array([1], dtype=np.float32)
    ne[1] = np.array([1], dtype=np.float32)
    game.add_nasheq(epoch, ne)

    # save a copy of game data
    game_path = os.getcwd() + '/game_data/game.pkl'
    fp.save_pkl(game, game_path)

    # sys.stdout.flush()
    return env, game

def DO_EGTA(env, game, retrain=False, epoch = 1, game_path = os.getcwd() + '/game_data/game.pkl', MPI_flag = False):
    #TODO: check length of str_set mismatch

    print("=======================================================")
    print("===============Begin Running DO-EGTA===================")
    print("=======================================================")

    count = 2
    # while count != 0:
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

        if retrain:
            print("Begin retraining attacker......")
            training.training_hado_att(game, epoch)
            print("Attacker retraining done......")

        print("Begin training defender......")
        training.training_def(game, mix_str_att, epoch)
        print("Defender training done......")

        if retrain:
            print("Begin retraining defender......")
            training.training_hado_def(game, epoch)
            print("Defender retraining done......")


        # Judge beneficial deviation
        # one plays nn and another plays ne strategy
        print("Simulating attacker payoff. New strategy vs. mixed opponent strategy.")
        nn_att = "att_str_epoch" + str(epoch) + ".pkl"
        nn_def = mix_str_def
        if MPI_flag:
            a_BD, _ = do_MPI_sim(nn_att, nn_def)
        else:
            a_BD, _ = series_sim(env, game, nn_att, nn_def, game.num_episodes)
        print("Simulation done for a_BD.")

        print("Simulating defender's payoff. New strategy vs. mixed opponent strategy.")
        nn_att = mix_str_att
        nn_def = "def_str_epoch" + str(epoch) + ".pkl"
        if MPI_flag:
            _, d_BD = do_MPI_sim(nn_att, nn_def)
        else:
            _, d_BD = series_sim(env, game, nn_att, nn_def, game.num_episodes)
        print("Simulation done for d_BD.")
        #
        # #TODO: This may lead to early stop.
        # if a_BD - aPayoff < game.threshold and d_BD - dPayoff < game.threshold:
        #     print("*************************")
        #     print("aPayoff=", aPayoff, " ", "dPayoff=", dPayoff)
        #     print("a_BD=", a_BD, " ", "d_BD=", d_BD)
        #     print("*************************")
        #     break
        #
        game.add_att_str("att_str_epoch" + str(epoch) + ".pkl")
        game.add_def_str("def_str_epoch" + str(epoch) + ".pkl")

        # simulate and extend the payoff matrix.
        game = sim_Series.sim_and_modifiy_Series_with_game(game)

        #
        # find nash equilibrium using gambit analysis
        payoffmatrix_def = game.payoffmatrix_def
        payoffmatrix_att = game.payoffmatrix_att
        print("Begin Gambit analysis.")
        nash_att, nash_def = ga.do_gambit_analysis(payoffmatrix_def, payoffmatrix_att)
        ga.add_new_NE(game, nash_att, nash_def, epoch)
        fp.save_pkl(game, game_path)
        print("Round_" + str(epoch) + " has done and game was saved.")
        print("=======================================================")
        break
        # count -= 1

    #     sys.stdout.flush() #TODO: make sure this is correct.
    #
    # os._exit(os.EX_OK)


if __name__ == '__main__':
    env, game = initialize(env_name='test_env')
    DO_EGTA(env, game)








