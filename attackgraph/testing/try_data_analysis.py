from attackgraph.data_analysis import expected_payoff as ep
from attackgraph.data_analysis import learning_curve, learning_curve_many

# data = 'att_data3'
# learning_curve(data)

data_att = ['att_data2', 'att_data3']
learning_curve_many(data_att)

# data_def_old = ['def_data2','def_data3']
# learning_curve_many(data_def_old)

