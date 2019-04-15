from attackgraph.sim_MPI_retrain import do_MPI_sim_retrain
from attackgraph.simulation import series_sim_retrain
from attackgraph import file_op as fp
import os
import numpy as np


#TODO: sim_MPI may cause error since name==main os.exit
def sim_retrain(env, game, mix_str_att, mix_str_def, MPI_flag, epoch):
    # sim for retained attacker
    a_BD = sim_retrain_att(env, game, mix_str_def, MPI_flag, epoch)
    # sim for retained defender
    d_BD = sim_retrain_def(env, game, mix_str_att, MPI_flag, epoch)

    return a_BD, d_BD


def sim_retrain_att(env, game, mix_str_def, MPI_flag, epoch):
    rewards_att = fp.load_pkl(os.getcwd() + '/retrained_rew/' + 'rewards_att.pkl') # reward is np.array([1,2,3,4])
    k, gamma, alpha = game.param
    DIR = os.getcwd() + '/retrain_att/'
    str_list = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and '.pkl' in name]
    num_str = len(str_list)
    util = []
    for i in range(num_str):
        nn_att = 'att_str_retrain' + str(i) + ".pkl"
        nn_def = mix_str_def
        if MPI_flag:
            a_BD, _ = do_MPI_sim_retrain(nn_att, nn_def)
        else:
            a_BD, _ = series_sim_retrain(env, game, nn_att, nn_def, 100)

        util.append(alpha*a_BD+(1-alpha)*rewards_att[i])

    best_idx = np.argmax(np.array(util))
    os.rename(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(best_idx) + ".pkl", os.getcwd() + "/attacker_strategies/" + 'att_str_epoch' + str(epoch) + '.pkl')
    return np.max(np.array(util))



def sim_retrain_def(env, game, mix_str_att, MPI_flag, epoch):
    rewards_def = fp.load_pkl(os.getcwd() + '/retrained_rew/' + 'rewards_def.pkl')
    k, gamma, alpha = game.param
    DIR = os.getcwd() + '/retrain_def/'
    str_list = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and '.pkl' in name]
    num_str = len(str_list)
    util = []
    for i in range(num_str):
        nn_att = mix_str_att
        nn_def = "def_str_epoch" + str(i) + ".pkl"
        if MPI_flag:
            _, d_BD = do_MPI_sim_retrain(nn_att, nn_def)
        else:
            _, d_BD = series_sim_retrain(env, game, nn_att, nn_def, 100)

        util.append(alpha * d_BD + (1 - alpha) * rewards_def[i])

    best_idx = np.argmax(np.array(util))
    os.rename(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(best_idx) + ".pkl", os.getcwd() + "/defender_strategies/" + 'def_str_epoch' + str(epoch) + '.pkl')
    return np.max(np.array(util))