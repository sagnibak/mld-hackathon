from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense

from load_data import get_train_data

x_train = get_train_data(mode='mlp')