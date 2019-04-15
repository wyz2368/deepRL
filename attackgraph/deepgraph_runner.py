# Packages import
import numpy as np
import os
import time

# Modules import
from attackgraph import DagGenerator as dag
from attackgraph import file_op as fp
from attackgraph import json_op as jp
from attackgraph import sim_Series
from attackgraph import training
from attackgraph import util
from attackgraph import game_data
from attackgraph import gambit_analysis as ga
from attackgraph.simulation import series_sim
from attackgraph.sim_MPI import do_MPI_sim
from attackgraph.sim_retrain import sim_retrain

# load_env: the name of env to be loaded.
# env_name: the name of env to be generated.
# MPI_flag: if running simulation in parallel or not.

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

    # load param
    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    # initialize game data
    game = game_data.Game_data(env, num_episodes=param['num_episodes'], threshold=param['threshold'])
    game.set_hado_param(param=param['hado_param'])
    game.set_hado_time_step(param['retrain_timesteps'])
    game.env.defender.set_env_belong_to(game.env)
    game.env.attacker.set_env_belong_to(game.env)

    env.defender.set_env_belong_to(env)
    env.attacker.set_env_belong_to(env)

    # uniform strategy has been produced ahead of time
    epoch = 1

    act_att = 'att_str_epoch1.pkl'
    act_def = 'def_str_epoch1.pkl'

    game.add_att_str(act_att)
    game.add_def_str(act_def)

    # simulate using random strategies and initialize payoff matrix
    if MPI_flag:
        aReward, dReward = do_MPI_sim(act_att, act_def)
    else:
        aReward, dReward = series_sim(env, game, act_att, act_def, game.num_episodes)

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

def EGTA(env, game, start_hado = 3, retrain=False, epoch = 1, game_path = os.getcwd() + '/game_data/game.pkl', MPI_flag = False):
    #TODO: check length of str_set mismatch

    if retrain:
        print("=======================================================")
        print("==============Begin Running HADO-EGTA==================")
        print("=======================================================")
    else:
        print("=======================================================")
        print("===============Begin Running DO-EGTA===================")
        print("=======================================================")

    retrain_start = False

    count = 7
    while count != 0:
    # while True:
        # fix opponent strategy
        mix_str_def = game.nasheq[epoch][0]
        mix_str_att = game.nasheq[epoch][1]
        aPayoff, dPayoff = util.payoff_mixed_NE(game, epoch)

        # increase epoch
        epoch += 1
        print("Current epoch is " + str(epoch))

        # train and save RL agents

        if retrain and epoch > start_hado:
            retrain_start = True

        print("Begin training attacker......")
        training.training_att(game, mix_str_def, epoch, retrain=retrain_start)
        print("Attacker training done......")

        print("Begin training defender......")
        training.training_def(game, mix_str_att, epoch, retrain=retrain_start)
        print("Defender training done......")

        if retrain and epoch > start_hado:
            print("Begin retraining attacker......")
            training.training_hado_att(game)
            print("Attacker retraining done......")

            print("Begin retraining defender......")
            training.training_hado_def(game)
            print("Defender retraining done......")

            # Simulation for retrained strategies and choose the best one as player's strategy.
            print('Begin retrained sim......')
            a_BD, d_BD = sim_retrain(env, game, mix_str_att, mix_str_def, MPI_flag, epoch)
            print('Done retrained sim......')

        else:

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
        # break
        count -= 1

    #     sys.stdout.flush() #TODO: make sure this is correct.
    #
    # os._exit(os.EX_OK)



if __name__ == '__main__':
    env, game = initialize(env_name='test_env')
    EGTA(env, game, retrain=True)
    # EGTA(env, game, retrain=False)








