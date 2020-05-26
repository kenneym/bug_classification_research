import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, PReLU, LeakyReLU,ThresholdedReLU

from hyperopt import Trials, STATUS_OK, tpe
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, quniform


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    random_state = 101
    features = pd.read_pickle('../data/pickles/tfidf_features.pkl')
    labels = pd.read_pickle('../data/pickles/backlog_labels.pkl')
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,random_state=random_state)
    
    return x_train, y_train, x_test, y_test


def tfidf_models(x_train, y_train, x_test, y_test):
    """
    Tests different model permutaions
    """
    
    # dropout_rate = {{quniform(0, 0.5, 0.1)}}
    
    # Define the paramater options to be tested:
    dropout_rate = 0.3
    activation = {{choice(['relu', 'tanh', 'prelu'])}}
    first_drop = {{choice([True, False])}}
    optimizer = {{choice(['adam', 'nadam', 'adamax', 'rmsprop'])}}
    chosen_layers = {{choice(['one', 'two', 'three'])}}
    layer_1 = {{choice([4096, 2048, 1024])}}
    layer_2 = -1
    layer_3 = -1
    
    # Build the Model:
    model = Sequential()
    model.add(Input(shape=[len(x_train.keys())], name="TFIDF_Features"))

    if first_drop:
        model.add(Dropout(dropout_rate, trainable=True))

    model.add(Dense(layer_1, input_shape=[len(x_train.keys())]))
    
    if activation == "prelu":
        model.add(PReLU())
    else:
        model.add(Activation(activation))
        
    layers = 1
    if chosen_layers =='two':
        layers = 2
    if chosen_layers =='three':
        layers = 3
        
    if layers >= 2:
        layer_2 = {{choice([2048, 1024, 512])}}
        model.add(Dropout(dropout_rate, trainable=True))
        model.add(Dense(layer_2))
        
        if activation == "prelu":
            model.add(PReLU())
        else:
            model.add(Activation(activation))

    
    if layers >= 3:
        layer_3 = {{choice([512, 256, 128])}}
        model.add(Dropout(dropout_rate, trainable=True))
        model.add(Dense(layer_3))
        
        if activation == "prelu":
            model.add(PReLU())
        else:
            model.add(Activation(activation))
    
    model.add(Dropout(dropout_rate, trainable=True))
    model.add(Dense(len(y_train.keys()), activation='softmax', name="softmax_output"))
    
    # Parameters

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    elif optimizer == 'adamax':
        optimizer = tf.keras.optimizers.Adamax()
    elif optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam()
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop()
    
    model.compile(loss='kullback_leibler_divergence', # originally, we tested a variety of different loss fns, 
                                                      # but we quickly found KL-Divergence to be the best and stopped wasting 
                                                      # permutations with other loss fns
                optimizer=optimizer,
                metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

    result = model.fit(x_train, y_train,
              batch_size=10, 
              epochs=100,
              verbose=2,
              validation_split=0.1,
              callbacks = [early_stop])
    
    validation_acc = np.amax(result.history['val_accuracy'])
    
    
    # Reccord the paramaters for the best performing models:
    if validation_acc > 0.79:

        model_file = open('model-trials.txt', 'a') 
        print('Best validation acc of epoch:', validation_acc, file=model_file)
        print('Optimizer: ', optimizer, 'activation:', activation, 
              'chosen_layers', chosen_layers, 'layer_1', layer_1, 
              'layer_2', layer_2, 'layer_3', layer_3, 
               file=model_file)
    
        model_file.close()

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

    best_run, best_model = optim.minimize(model=tfidf_models,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=500,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    model_file = open('best-model.txt', 'a') 

    print("Evalutation of best performing model:", file=model_file)
    print(best_model.evaluate(X_test, Y_test), file=model_file)
    print("Best performing model chosen hyper-parameters:", file=model_file)
    print(best_run, file=model_file) 

    model_file.close() 

