from baselines import deepq
from baselines.common import models

#TODO: make sure the path is correct
def load_action(path, game):

    env = game.env
    num_layers = game.num_layers
    num_hidden = game.num_hidden
    act = deepq.learn(
        env,
        network=models.mlp(num_layers=num_layers-3, num_hidden=num_hidden),
        total_timesteps=0,
        load_path=path
    )
    return act

