from mpi4py import MPI
from attackgraph.parallel_sim import parallel_sim
from baselines.deepq import load_action
from attackgraph import file_op as fp
import os


#TODO: assign epoch
def sim_and_modifiy_MPI():
    #TODO: load game
    path = os.getcwd() + '/game_data/game.pkl'
    game = fp.load_pkl(path)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size



    if rank == 0:
        fp.save_pkl(newData,path='./attackgraph/data/newdata_' + str(epoch) + ".pkl")

    newData = comm.gather(data, root=0)






if __name__ == '__main__':
    sim_and_modifiy_MPI()