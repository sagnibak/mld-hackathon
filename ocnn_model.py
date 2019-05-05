from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import l2

import numpy as np

from load_data import get_train_data
from losses import make_ocnn_obj

# hyperparameters
L1_UNITS = 80
OUT_UNITS = 1
NUM_EPOCHS_MODEL = 1000
NUM_EPOCHS_R = 1000
BATCH_SIZE = 2048
NU = 0.50

x_train = get_train_data(mode='mlp')

def make_model():
    img_in = Input(shape=(x_train.shape[1:]), dtype='float32')
    layer1 = Dense(L1_UNITS, activation='sigmoid', use_bias=True,
                kernel_regularizer=l2())(img_in)
    # linear activation for the output
    model_out = Dense(OUT_UNITS, use_bias=True,
                      kernel_regularizer=l2())(layer1)
    return Model(img_in, model_out)

def train_model():
    # initialize r
    r = np.random.normal(0, 1)

    for i in range(NUM_EPOCHS_R):
        print(f'Epoch: {i}, r: {r}')
        early_stopping = EarlyStopping(monitor='loss', patience=5,
                                       restore_best_weights=True)
        tensorboard = TensorBoard(log_dir=f'logdir3/run{i}')

        # optimize the model for the given value of `r`
        model = make_model()
        # model.summary()
        model.compile('adam', loss=make_ocnn_obj(NU, r))
        model.fit(x=x_train, y=None,
                batch_size=BATCH_SIZE,
                epochs=NUM_EPOCHS_MODEL,
                verbose=0,
                callbacks=[early_stopping, tensorboard])
        
        # calculate the NU-th quantile on the train set and
        # store that as r
        y_hat = model.predict(x_train)
        print('y_hat shape: ', y_hat.shape)

        # save r
        with open('r3.csv', 'a+') as f:
            f.write(f'{i}, {r}\n')

        r = np.quantile(y_hat, NU)
        
        if (i + 1) % 10 == 0:
            model.save(f'models/ckpt_model_3_{i}.h5')

    # return the last model
    return model

if __name__ == "__main__":
    final_model = train_model()
    final_model.save('models/final_model3.h5')
