
# coding: utf-8

from tensorflow.python.lib.io import file_io
import argparse
import tensorflow as tf
import numpy as np 
import pandas as pd
from pandas.compat import StringIO
print(tf.__version__) 


def initialise_training_set():
    path_train = 'gs://rakshas_gc/Data/x_train.csv'
    print('downloading x_train.csv file from', path_train)     
    file_stream = file_io.FileIO(path_train, mode='r')
    x_train_data = pd.read_csv(StringIO(file_stream.read()))
    x_train = x_train_data.values 
    return x_train


def initialise_test_set():
    path_test = 'gs://rakshas_gc/Data/x_test.csv'
    print('downloading x_test.csv file from', path_test)     
    file_stream = file_io.FileIO(path_test, mode='r')
    x_test_data = pd.read_csv(StringIO(file_stream.read()))
    x_test = x_test_data.values 
    return x_test


def load_labels():
    path_y_train = 'gs://rakshas_gc/Data/y_train.csv'
    print('downloading y_train.csv file from', path_y_train)     
    file_stream = file_io.FileIO(path_y_train, mode='r')
    y_train_data = pd.read_csv(StringIO(file_stream.read()))
    y_train = y_train_data.values 
    path_y_test = 'gs://rakshas_gc/Data/y_test.csv'
    print('downloading y_test.csv file from', path_y_test)     
    file_stream = file_io.FileIO(path_y_test, mode='r')
    y_test_data = pd.read_csv(StringIO(file_stream.read()))
    y_test = y_test_data.values
    print(y_train.shape)
    print(y_test.shape)
    return y_train, y_test


def train_input_fn(x_train, y_train):
    return tf.estimator.inputs.numpy_input_fn(
      x= x_train,
      y= y_train,
      batch_size= 32,
      num_epochs= 64,
      shuffle= False,
      queue_capacity= 8000)
    


def test_input_fn(x_test, y_test):
    return tf.estimator.inputs.numpy_input_fn(
      x= x_test,
      y= y_test,
      batch_size= 32,
      num_epochs= 1,
      shuffle= False,
      queue_capacity= 5000)

#def model(x_train, x_test, batch_size):
    #source = Input(shape=x_train.shape[1:], batch_shape=(32,x_train.shape[1],x_train.shape[2]), dtype=tf.float32, name='Input')
    #lstm = tf.keras.layers.CuDNNLSTM(128, kernel_initializer='glorot_uniform', activation = 'relu', name='LSTM')(source)
    #Dense_1 = Dense(64,kernel_initializer='glorot_uniform', activation = 'relu', name='Dense_1')(lstm)
    #dropout = Dropout(0.3, name='Dropout')(Dense_1)
    #Dense_2 = Dense(8, kernel_initializer='glorot_uniform', activation = 'relu', name='Dense_2')(dropout)
    #predicted = Dense(1, kernel_initializer='glorot_uniform', activation = 'sigmoid',name='Output')(Dense_2)
    #model = Model(inputs=[source], outputs=[predicted])
    #return model

def model(x_train,x_test,y_train,y_test):
    model = tf.keras.Sequential() 
    model.add(tf.keras.layers.CuDNNLSTM(256, input_shape=(x_train.shape[1:]), kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dense(128, kernel_initializer='glorot_uniform', activation = 'relu'))
    model.add(tf.keras.layers.Dense(64, kernel_initializer='glorot_uniform', activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, kernel_initializer='glorot_uniform', activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))
    return model

def main(job_dir):
    x_train = initialise_training_set()
    x_test = initialise_test_set()
    y_train, y_test = load_labels()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    print("training with GPU clusters...")
    Model = model(x_train, x_test, y_train, y_test)
    optimizer = tf.train.AdamOptimizer(epsilon=0.0001)
    Model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy','binary_accuracy'])
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=job_dir)
    estimator = tf.keras.estimator.model_to_estimator(keras_model=Model, config=run_config)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn(x_train,y_train),max_steps=20000)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn(x_test, y_test), steps=1)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("reached here lol")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')

    
    main(job_dir)
