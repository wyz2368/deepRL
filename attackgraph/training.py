from baselines import deepq
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

    num_layers = game.num_layers
    num_hidden = game.num_hidden

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=num_hidden, num_layers=num_layers-3),
                lr = 5e-5,
                total_timesteps=1000,
                exploration_fraction=0.5,
                exploration_final_eps=0.03,
                print_freq=250,
                param_noise=False,
                gamma=0.99,
                prioritized_replay=True,
                checkpoint_freq=30000,
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

    num_layers = game.num_layers
    num_hidden = game.num_hidden

    learner = Learner()
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=num_hidden,num_layers=num_layers-3),
                lr = 5e-5,
                total_timesteps=1000,
                exploration_fraction=0.5,
                exploration_final_eps=0.03,
                print_freq=250,
                param_noise=False,
                gamma=0.99,
                prioritized_replay=True,
                checkpoint_freq=30000,
                scope = "def_str_epoch" + str(epoch) + '.pkl' + '/'
            )
            print("Saving defender's model to pickle.")
            act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()


