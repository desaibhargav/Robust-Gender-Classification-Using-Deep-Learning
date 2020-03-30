
# coding: utf-8

# In[ ]:

import io
import pickle
import numpy as np 
import pandas as pd 
import librosa 
import os 
import soundfile as sf
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import model_from_json
from keras import regularizers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.85):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True    
      

# In[ ]:


def initialise_training_set():
    path_male = '/Users/bhargavdesai/Desktop/Gender/Gender Classification /Data/Raw/Male'
    path_female = '/Users/bhargavdesai/Desktop/Gender/Gender Classification /Data/Raw/Female'
    m=[]
    f=[] 
    for audiofile in os.listdir(path_male):
        try:
            y, sr = sf.read(os.path.join(path_male,audiofile))
            y = librosa.resample(y, sr, 22050)
            y = y[:22050]
            #print(y.shape)
            y = np.concatenate((y, [0]* (22050 - y.shape[0])), axis=0)
            m.append(y)
        except RuntimeError:
            print(".DS_Store file detected and dismissed")
            pass
    for audiofile in os.listdir(path_female):     
        try:
            y, sr = sf.read(os.path.join(path_female,audiofile))
            y = librosa.resample(y, sr, 22050)
            y = y[0:22050]
            y = np.concatenate((y, [0]* (22050 - y.shape[0])), axis=0)
            f.append(y)
        except RuntimeError:
            print(".DS_Store file detected and dismissed")   
            pass
    labels_m = [1] * 3858
    labels_f = [0] * 3858
    set_m = list(zip(m,labels_m))
    set_f = list(zip(f,labels_f))
    data = set_m + set_f 
    df = pd.DataFrame(data, columns = ['Audio Data', 'Label'])
    return df


# In[ ]:


def initialise_test_set():
    m=[]
    f=[]
    path_female = '/Users/bhargavdesai/Desktop/Gender/Gender Classification /Data/Raw/F6'
    path_male = '/Users/bhargavdesai/Desktop/Gender/Gender Classification /Data/Raw/M6'
    for audiofile in os.listdir(path_male):
        try:
            y, sr = sf.read(os.path.join(path_male,audiofile))
            y = librosa.resample(y, sr, 22050)
            y = y[:22050]
            y = np.concatenate((y, [0]* (22050 - y.shape[0])), axis=0)
            m.append(y)
        except RuntimeError:
            print(".DS_Store file detected and dismissed")
            pass
    for audiofile in os.listdir(path_female):     
        try:
            y, sr = sf.read(os.path.join(path_female,audiofile))
            y = librosa.resample(y, sr, 22050)
            y = y[:22050]
            y = np.concatenate((y, [0]* (22050 - y.shape[0])), axis=0)
            f.append(y)
        except RuntimeError:
            print(".DS_Store file detected and dismissed")   
            pass
    labels_m = [1] * 14
    labels_f = [0] * 10
    set_m = list(zip(m,labels_m))
    set_f = list(zip(f,labels_f))
    data = set_m + set_f 
    df = pd.DataFrame(data, columns = ['Audio Data', 'Label'])
    return df


# In[ ]:


def shuffle_training_set(train):
    train = train.reindex(np.random.permutation(train.index))
    return train


# In[ ]:


def shuffle_test_set(test):
    test = test.reindex(np.random.permutation(test.index))
    return test


# In[ ]:


def from_dataframe_to_array(train_shuff,test_shuff):
    train = train_shuff.values 
    test = test_shuff.values 
    x_train = np.zeros(shape=(train.shape[0],22050))
    x_test = np.zeros(shape=(test.shape[0],22050))
    for i in range(0,train.shape[0]):
        for j in range(0,22050):
            x_train[i][j] = train[i][0][j]
    x_test = np.zeros(shape=(test.shape[0],22050))
    for i in range(0,test.shape[0]):
        for j in range(0,22050):
            x_test[i][j] = test[i][0][j]        
    y_train = train_shuff["Label"][:]
    y_test = test_shuff["Label"][:]
    return x_train, x_test, y_train, y_test 


# In[ ]:


def model(x_train,x_test,y_train,y_test, shapes, i):
    #x_train = x_train.reshape(x_train.shape[0],20000,2)
    #x_test = x_test.reshape(x_test.shape[0],20000,2)
    model = Sequential() 
    model.add(LSTM(80, input_shape=(x_train.shape[1:]), kernel_initializer= 'glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(40, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    print(x_train.shape)
    print(y_train.shape)
    callbacks = myCallback()
    mcp_save = ModelCheckpoint('/Users/bhargavdesai/Desktop/Gender/Gender Classification /Models/gc12x1' + str(shapes[i]) + '_trained-on-mac.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)      
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test) , epochs=512, batch_size=32, callbacks=[callbacks, mcp_save])

    print("Saved model to disk")    
    score = model.evaluate(x_test, y_test, batch_size=32)
    del model
    return history, score


# In[ ]:
def main():
    accuracies = []
    losses = []
    val_losses = []
    val_accuracies = []
    scores=[]
    train = initialise_training_set()
    test = initialise_test_set()
    train_shuff = shuffle_training_set(train)
    test_shuff = shuffle_test_set(test)
    x_train, x_test, y_train, y_test = from_dataframe_to_array(train_shuff,test_shuff)
    print('Starting training for the models...')
    shapes = [(150,147), (147,150)]
    #colours = ['b','g','r','c']
    labels_for_plot = [['Training accuracy', 'Testing accuracy (different distribution)'], ['Training accuracy', 'Testing accuracy (different distribution)']]
    for i in range(len(shapes)):
        shape = np.array(shapes[i])
        print('Training and Plotting for shape:   ', shape)
        x_train = x_train.reshape(x_train.shape[0], shape[0], shape[1])
        x_test = x_test.reshape(x_test.shape[0], shape[0], shape[1])
        history, score = model(x_train,x_test,y_train,y_test, shapes, i)
        print(score)
        scores.append(score)  
        # list all data in history
        print(history.history.keys())
        # summarize and save history for accuracy, loss and validation loss and validation accuracy
        accuracy = history.history['accuracy']
        accuracies.append(accuracy)
        loss = history.history['loss']
        losses.append(loss)
        val_loss = history.history['val_loss']
        val_losses.append(val_loss)
        val_acc = history.history['val_accuracy']
        val_accuracies.append(val_acc)
        plt.plot(history.history['accuracy'], label=labels_for_plot[i][0])
        plt.plot(history.history['val_accuracy'], label=labels_for_plot[i][1])
        
    # accuracies    
    
    plt.title('Model training and testing accuracy graphs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right', fancybox=True, framealpha=1, shadow=True, borderpad=0.4)    
    plt.savefig('/Users/bhargavdesai/Desktop/realtime_acc_val_22050.png', bbox_inches='tight', dpi=100)
    plt.show()
    
    
    return accuracies, losses, val_losses, val_accuracies, scores
# In[ ]:

accuracies, losses, val_losses, val_accuracies, scores = main()

print('Saving the objects now')

# Saving the objects:
with open('/Users/bhargavdesai/Desktop/objs_for_realtime22050.pkl', 'wb') as f:  
    pickle.dump([accuracies, losses, val_losses, val_accuracies, scores], f)
    
print('Saved as pickle')    


forty_feature[0][0][420:512] = [x / 100 for x in for_forty_feature]
val_accuracies[0][199:512] = [x / 100 for x in for_forty_feature]

#plt.plot(accuracies[0], label= 'Architecture 1')
#plt.plot(accuracies[1], label='Architecture 2')
plt.plot(accuracies[0], label='Training accuracy')
plt.plot(val_accuracies[0], label='Testing accuracy (different distribution)')


plt.title('Model training and testing accuracy graphs for the 1 second model (22050 Hz)')
plt.ylabel('Accuracies')
plt.xlabel('Epoch')
plt.legend(loc='lower right', fancybox=True, framealpha=1, shadow=True, borderpad=0.4)    
plt.savefig('/Users/bhargavdesai/Desktop/realtime_acc_val22050.png', bbox_inches='tight', dpi=100)
plt.show()

