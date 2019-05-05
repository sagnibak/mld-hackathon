from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import l2

from load_data import get_train_data
from losses import make_ocnn_obj

# hyperparameters
L1_UNITS = 40
OUT_UNITS = 1
NUM_EPOCHS_MODEL = 1000
NUM_EPOCHS_R = 1000
BATCH_SIZE = 256
NU = 0.002

x_train = get_train_data(mode='mlp')

def make_model():
    img_in = Input(shape=(x_train.shape[1:]), dtype='float32')
    layer1 = Dense(L1_UNITS, activation='sigmoid', use_bias=False,
                kernel_regularizer=l2())
    # linear activation for the output
    model_out = Dense(OUT_UNITS, use_bias=False,
                      kernel_regularizer=l2())
    return Model(img_in, model_out)

def train_model():
    # initialize r
    r = np.random.normal(0, 1)

    for i in range(NUM_EPOCHS_R):
        print(f'Epoch: {i}, r: {r}')
        early_stopping = EarlyStopping(monitor='loss', patience=5,
                                       restore_best_weights=True)
        tensorboard = TensorBoard(log_dir=f'logdir/run{i}')

        # optimize the model for the given value of `r`
        model = make_model()
        model.compile('adam', loss=make_ocnn_obj(NU, r))
        model.fit(x=x_train, y=None,
                batch_size=BATCH_SIZE,
                epochs=NUM_EPOCHS_MODEL,
                callbacks=[early_stopping, tensorboard])
        
        # calculate the NU-th quantile on the train set and
        # store that as r
        y_hat = model.predict(x_train)
        print('y_hat shape: ', y_hat.shape)
        r = np.quantile(y_hat, NU)

    # return the last model
    return model

if __name__ == "__main__":
    final_model = train_model()
    final_model.save('models/model0.h5')
