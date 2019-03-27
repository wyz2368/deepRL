from baselines import deepq
from baselines.common import models
from baselines.deepq.deepq import learn_multi_nets

#TODO: make sure the path is correct
def load_action(path, game, training_flag):

    env = game.env
    env.set_training_flag(training_flag)
    num_layers = game.num_layers
    num_hidden = game.num_hidden
    act = learn_multi_nets(
        env,
        network=models.mlp(num_layers=num_layers-3, num_hidden=num_hidden),
        total_timesteps=0,
        load_path=path
    )
    return act

