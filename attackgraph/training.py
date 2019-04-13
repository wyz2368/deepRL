from attackgraph import json_op as jp
from baselines.common import models
from baselines.deepq.deepq import learn_multi_nets, Learner
import os
import copy
#TODO: improvement can be done by not including all RL strategies.

DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'

#TODO: pick a strategy from a mixed strategy.
#TODO: add strategy name to strategy name list.
#TODO: extend payoff matrix.
#TODO: network model should be rechecked.
#TODO: make all params able to set outside. Not hard coding.
def training_att(game, mix_str_def, epoch):
    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while training")

    env = copy.deepcopy(game.env)
    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def) #TODO: Can mix_str_def be expressed by game and epoch?
    env.defender.set_str_set(game.def_str)

    # num_layers = game.num_layers
    # num_hidden = game.num_hidden

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr =param['lr'],
                total_timesteps=param['total_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'att_str_epoch' + str(epoch) + '.pkl' + '/'
            )
            print("Saving attacker's model to pickle.")
            act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()




def training_def(game, mix_str_att, epoch):
    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while training")

    env = copy.deepcopy(game.env)
    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    # num_layers = game.num_layers
    # num_hidden = game.num_hidden

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=param['lr'],
                total_timesteps=param['total_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = "def_str_epoch" + str(epoch) + '.pkl' + '/'
            )
            print("Saving defender's model to pickle.")
            act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()


def training_hado_att(game, epoch):
    param = game.param
    mix_str_def = game.hado_str(identity=0, param=param)

    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while training")

    env = copy.deepcopy(game.env)
    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def) #TODO: Can mix_str_def be expressed by game and epoch?
    env.defender.set_str_set(game.def_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner(retrain=True)
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr =param['lr'],
                total_timesteps=param['retrain_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'att_str_epoch' + str(epoch) + '.pkl' + '/',
                load_path=DIR_att + "att_str_epoch" + str(epoch) + ".pkl"

            )
            # print("Saving attacker's model to pickle.")
            # act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()




def training_hado_def(game, epoch):
    param = game.param
    mix_str_att = game.hado_str(identity=1, param=param)

    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while training")

    env = copy.deepcopy(game.env)
    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    learner = Learner(retrain=True)
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=param['lr'],
                total_timesteps=param['retrain_timesteps'],
                exploration_fraction=param['exploration_fraction'],
                exploration_final_eps=param['exploration_final_eps'],
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = "def_str_epoch" + str(epoch) + '.pkl' + '/',
                load_path = DIR_def + "def_str_epoch" + str(epoch) + ".pkl"
            )
            # print("Saving defender's model to pickle.")
            # act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()