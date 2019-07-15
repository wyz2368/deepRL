import numpy as np
from attackgraph import file_op as fp
import os
import warnings
from attackgraph.simulation import series_sim_combined
from attackgraph.gambit_analysis import do_gambit_analysis
from attackgraph.deepgraph_runner import initialize
import sys
# sys.path.append('/home/wangyzh/combined')

def whole_payoff_matrix(num_str):
    print('Begin simulating payoff matrix of combined game.')
    # path = os.getcwd() + '/game_data/game.pkl'
    game = initialize(load_env='run_env_B', env_name=None)

    env = game.env
    num_episodes = game.num_episodes
    payoff_matrix_att = np.zeros((num_str, num_str))
    payoff_matrix_def = np.zeros((num_str, num_str))

    second_start = 42
    third_start = 85

    for i in np.arange(1,43):
        def_str = 'def_str_epoch' + str(i) + '.pkl'
        if i < second_start:
            def_scope = 'def_str_epoch' + str(i) + '.pkl'
        elif i >= second_start and i < third_start:
            def_scope = 'def_str_epoch' + str(i-second_start+2) + '.pkl'
        else:
            def_scope = 'def_str_epoch' + str(i-third_start+2) + '.pkl'

        for j in np.arange(1,num_str+1):
            att_str = 'att_str_epoch' + str(j) + '.pkl'
            if j < second_start:
                att_scope = 'att_str_epoch' + str(j) + '.pkl'
            elif j >= second_start and j < third_start:
                att_scope = 'att_str_epoch' + str(j - second_start + 2) + '.pkl'
            else:
                att_scope = 'att_str_epoch' + str(j - third_start + 2) + '.pkl'

            print('Current position:', i,j)
            sys.stdout.flush()

            aReward, dReward = series_sim_combined(env, game, att_str, att_scope, def_str, def_scope, num_episodes)
            payoff_matrix_att[i-1,j-1] = aReward
            payoff_matrix_def[i-1,j-1] = dReward

        save_path = os.getcwd() + '/combined_game/'
        fp.save_pkl(payoff_matrix_att, save_path + 'payoff_matrix_att.pkl')
        fp.save_pkl(payoff_matrix_def, save_path + 'payoff_matrix_def.pkl')

    print('Done simulating payoff matrix of combined game.')
    return payoff_matrix_att, payoff_matrix_def


def regret(nash_att, nash_def, payoffmatrix_att, payoffmatrix_def):
    num_str = len(nash_att)
    x1, y1 = np.shape(payoffmatrix_def)
    x2, y2 = np.shape(payoffmatrix_att)
    if x1 != y1 or x1 != x2 or x2 != y2 or x1 != num_str:
        raise ValueError("Dim of NE does not match payoff matrix.")

    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.sum(nash_def * payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * payoffmatrix_att * nash_att), decimals=2)

    utils_def = np.round(np.sum(payoffmatrix_def * nash_att, axis=1), decimals=2)
    utils_att = np.round(np.sum(nash_def * payoffmatrix_att, axis=0), decimals=2)

    regret_def = utils_def - dPayoff
    regret_att = utils_att - aPayoff

    regret_def = np.reshape(regret_def, newshape=np.shape(regret_att))

    regret_att = -regret_att
    regret_def = -regret_def

    return regret_att, regret_def

def mean_regret(regret_att, regret_def):
    mean_reg_att = []
    mean_reg_def = []
    mean_reg_att.append(np.round(np.mean(regret_att[1:41]), decimals=2))
    mean_reg_def.append(np.round(np.mean(regret_def[1:41]), decimals=2))
    mean_reg_att.append(np.round(np.mean(regret_att[41:85]), decimals=2))
    mean_reg_def.append(np.round(np.mean(regret_def[41:85]), decimals=2))
    mean_reg_att.append(np.round(np.mean(regret_att[85:129]), decimals=2))
    mean_reg_def.append(np.round(np.mean(regret_def[85:129]), decimals=2))
    return mean_reg_att, mean_reg_def

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    num_str = 125

    payoff_matrix_att, payoff_matrix_def = whole_payoff_matrix(num_str)
    # nash_att, nash_def = do_gambit_analysis(payoff_matrix_def, payoff_matrix_att)
    # regret_att, regret_def = regret(nash_att, nash_def, payoff_matrix_att, payoff_matrix_def)
    # mean_reg_att, mean_reg_def = mean_regret(regret_att, regret_def)
    #
    # # Saving
    # save_path = os.getcwd() + '/combined_game/'
    # data_dic = {}
    # data_dic['payoff_matrix_att'] = payoff_matrix_att
    # data_dic['payoff_matrix_def'] = payoff_matrix_def
    # data_dic['nash_att'] = nash_att
    # data_dic['nash_def'] = nash_def
    # data_dic['regret_att'] = regret_att
    # data_dic['regret_def'] = regret_def
    # data_dic['mean_reg_att'] = mean_reg_att
    # data_dic['mean_reg_def'] = mean_reg_def
    # fp.save_pkl(data_dic, save_path + 'data_dic.pkl')

    # Printing
    # print("############### Summary of Combined Game ####################")
    # print("Attacker's Nash Equilibrium Strategy: ", nash_att)
    # print("Defender's Nash Equilibrium Strategy: ", nash_def)
    # print('Regret of attacker:', regret_att)
    # print('Regret of defender:', regret_def)
    # print('Mean regret of attacker: ', mean_reg_att)
    # print('Mean regret of defender: ', mean_reg_def)
    # print("#################### End of Summary ########################")

