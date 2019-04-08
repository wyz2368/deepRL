import numpy as np
from attackgraph import file_op as fp
from baselines.common import models
import os
from baselines.deepq.deepq import learn_multi_nets, Learner
from attackgraph import uniform_str_init

DIR = os.getcwd() + '/'
DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'
# print(DIR_att)


#TODO: str in str_set should include .pkl
def sample_strategy_from_mixed(env, str_set, mix_str, identity):

    if not isinstance(mix_str,np.ndarray):
        raise ValueError("mix_str in sample func is not a numpy array.")

    if not len(str_set) == len(mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies.")

    picked_str = np.random.choice(str_set, p=mix_str)
    if not fp.isInName('.pkl', name = picked_str):
        raise ValueError('The strategy picked is not a pickle file.')

    if identity == 0: # pick a defender's strategy
        path = DIR + 'defender_strategies/'
    elif identity == 1:
        path = DIR + 'attacker_strategies/'
    else:
        raise ValueError("identity is neither 0 or 1!")

    # print(path + picked_str)
    if not fp.isExist(path + picked_str):
        raise ValueError('The strategy picked does not exist!')

    if "epoch1" in picked_str:
        act = fp.load_pkl(path + picked_str)
        return act

    flag = env.training_flag
    env.set_training_flag(identity)

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act = learn_multi_nets(
                env,
                network=models.mlp(num_hidden=256, num_layers=1), #TODO: hard coding.
                total_timesteps=0,
                load_path= path + picked_str
            )

    env.set_training_flag(flag)

    return act

def sample_both_strategies(env, att_str_set, att_mix_str, def_str_set, def_mix_str):

    if not len(att_str_set) == len(att_mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies for the attacker.")
    if not len(def_str_set) == len(def_mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies for the defender.")

    att_picked_str = np.random.choice(att_str_set, p=att_mix_str)
    def_picked_str = np.random.choice(def_str_set, p=def_mix_str)

    if not fp.isInName('.pkl', name=def_picked_str):
        raise ValueError('The strategy picked is not a pickle file for the defender.')
    if not fp.isInName('.pkl', name=att_picked_str):
        raise ValueError('The strategy picked is not a pickle file for the attacker.')

    path_def = DIR + 'defender_strategies/'
    path_att = DIR + 'attacker_strategies/'

    if not fp.isExist(path_def + def_picked_str):
        raise ValueError('The strategy picked does not exist for the defender!')
    if not fp.isExist(path_att + att_picked_str):
        raise ValueError('The strategy picked does not exist for the attacker!')

    # TODO: assign nn info from game
    env.set_training_flag(0)
    act_att = learn_multi_nets(
        env,
        network=models.mlp(num_hidden=256, num_layers=1),
        total_timesteps=0,
        load_path=path_att + att_picked_str
    )

    env.set_training_flag(1)
    act_def = learn_multi_nets(
        env,
        network=models.mlp(num_hidden=256, num_layers=1),
        total_timesteps=0,
        load_path= path_def + def_picked_str
    )

    return act_att, act_def

#TODO: check the input dim of nn and check if this could initialize nn.
#TODO: check when to set training flag
def rand_att_str_generator(env, game):
    # Generate random nn for attacker.
    num_layers = game.num_layers
    num_hidden = game.num_hidden

    env.set_training_flag(1)
    act_att = learn_multi_nets(
        env,
        network=models.mlp(num_hidden=num_hidden, num_layers=num_layers-3),
        total_timesteps=0
    )

    print("Saving attacker's model to pickle. Epoch name is equal to 1.")
    act_att.save(DIR_att + "att_str_epoch" + str(1) + ".pkl")
    # game.att_str.append("att_str_epoch" + str(1) + ".pkl")
    game.add_att_str("att_str_epoch" + str(1) + ".pkl")


def rand_def_str_generator(env, game):
    # Generate random nn for attacker.
    num_layers = game.num_layers
    num_hidden = game.num_hidden

    env.set_training_flag(0)
    act_def = learn_multi_nets(
        env,
        network=models.mlp(num_hidden=num_hidden, num_layers=num_layers-3),
        total_timesteps=0
    )

    print("Saving defender's model to pickle. Epoch in name is equal to 1.")
    act_def.save(DIR_def + "def_str_epoch" + str(1) + ".pkl")
    # game.def_str.append("def_str_epoch" + str(1) + ".pkl")
    game.add_def_str("def_str_epoch" + str(1) + ".pkl")



# def rand_att_str_generator_uniform(game):
#     def act_att(ob, mask, training_flag, stochastic=True, update_eps=-1):
#         if training_flag != 1:
#             raise ValueError("training flag for uniform att str is not 1")
#         legal_action = np.where(mask == 0)[0]
#         return np.random.choice(legal_action)
#
#     # return act_att
#     fp.save_pkl(act_att, DIR_att + "att_str_epoch" + str(1) + ".pkl")
#     game.add_att_str("att_str_epoch" + str(1) + ".pkl")
#
#
# def rand_def_str_generator_uniform(game):
#     def act_def(ob, mask, training_flag, stochastic=True, update_eps=-1):
#         if training_flag != 0:
#             raise ValueError("training flag for uniform def str is not 0")
#         legal_action = np.where(mask == 0)[0]
#         return np.random.choice(legal_action)
#
#     # return act_def
#     fp.save_pkl(act_def, DIR_def + "def_str_epoch" + str(1) + ".pkl")
#     game.add_def_str("def_str_epoch" + str(1) + ".pkl")










