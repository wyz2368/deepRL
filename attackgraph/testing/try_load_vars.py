import tensorflow as tf
import joblib
import os

sess = tf.Session()
sess.__enter__()

load_path = os.getcwd() + '/retrain_att/att_str_retrain0.pkl'
loaded_params = joblib.load(os.path.expanduser(load_path))
print(loaded_params.keys())