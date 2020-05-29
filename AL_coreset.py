import numpy as np
import os

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint,EarlyStopping, History
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical, multi_gpu_model
import tensorflow as tf
from keras import applications
from keras.models import load_model
import pickle
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split


print("----------------------------------------------------")
print("DATE and TIME of starting the code =", datetime.now())
print("----------------------------------------------------")

def parse_input():
    p=argparse.ArgumentParser()
    p.add_argument('experiment_index',type=int, help="index of current experiment")
    p.add_argument('batch_size', type=int, help="active learning batch size")
    p.add_argument('initial_size', type=int, help="initial sample size for active learning")
    p.add_argument('iterations', type=int, help="number of active learning batches to sample")
    p.add_argument('--output_path', '-mp', type=str,default=None)
    p.add_argument('method', type=str,
                   choices={'Coreset'})
    p.add_argument('--initial_idx_path', '-idx', type=str,default=None,help="path to a folder with a pickle file with the initial indices of the labeled set")
    p.add_argument('--weights_init_path', '-wip', type=str,default=None)
    p.add_argument('--data_path','-dp',type=str,default=None)
    p.add_argument('--labels_path','-lp',type=str,default=None)
    p.add_argument('--gpu', '-gpu', type=int, default=0)
    args = p.parse_args()
    return args
def get_unlabeled_idx(X_train_e, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train_e.shape[0])[np.logical_not(np.in1d(np.arange(X_train_e.shape[0]), labeled_idx))]
def greedy_k_center(labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

def query_coreset(X_train_e,labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        # use the learned representation for the k-greedy-center algorithm:
        new_indices = greedy_k_center(X_train_e[labeled_idx, :],X_train_e[unlabeled_idx, :], amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))


def evaluate_sample(ResNet50_model, X_train, Y_train, X_val_b,Y_val_b,X_data,Y_data,checkpoint_path):
    """
    A function that accepts a labeled-unlabeled data split and trains the relevant model on the labeled data, returning
    the model and it's accuracy on the test set.
    """

    # shuffle the training set:
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]
    X_validation = X_val_b
    Y_validation=to_categorical(Y_val_b)

    # train and evaluate the model:
    model = train_disease_classification_model(ResNet50_model,X_train, Y_train, X_validation, Y_validation, checkpoint_path)
    acc = model.evaluate(X_data, Y_data, verbose=0)

    return acc, model
def train_disease_classification_model(ResNet50_model,X_train, Y_train, X_validation, Y_validation, checkpoint_path):

    save_model = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    history = History()
    ResNet50_model.load_weights(checkpoint_initial_path)
    ResNet50_model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['acc'])
    ResNet50_model.fit(X_train,Y_train,epochs=100,batch_size=16,shuffle=True,validation_data=(X_validation,Y_validation),callbacks=[save_model,history],verbose=2)
    ResNet50_model.load_weights(checkpoint_path)
    return ResNet50_model

if __name__ == '__main__':
    args=parse_input()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Using gpu', args.gpu)
    if gpus:
        try: 
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
# Visible devices must be set before GPUs have been initialized
            print(e)
    results_path=args.output_path
    X_data=np.load(args.data_path)
    ## Normalize the data
    X_data=X_data.astype('float32')/255.
    Y_data=np.load(args.labels_path)

    #Splitting 
    X_train, X_val_b, Y_train, Y_val_b = train_test_split(X_data, Y_data, test_size=0.05, random_state=33)
    Y_train_c=to_categorical(Y_train)
    Y_data_c=to_categorical(Y_data)
    batch_size=args.batch_size
    initial_size=args.initial_size
    iterations=args.iterations
    uniquevalues, num_labels=np.unique(Y_data,return_counts=True)
    T=10
    initial_idx_path=args.initial_idx_path
    weights_init_path=args.weights_init_path
    if initial_idx_path is not None:
        idx_path = args.initial_idx_path
        with open(idx_path, 'rb') as f:
            labeled_idx = pickle.load(f)
    else:
        labeled_idx=np.random.choice(len(X_train),initial_size, replace=False)
    checkpoint_path=os.path.join(output_path,'{methods}_{exp}_{b_s}_{i_s}/'.format(method=args.method,exp=args.experiment_index,b_s=args.batch_size,i_s=args.iterations))
    os.makedirs(checkpoint_path,exist_ok=True)
    ResNet50_model = applications.ResNet50(input_shape=(256,256,3),include_top=True,weights=None, input_tensor=None, pooling=None,classes=9)
    checkpoint_initial_path=args.weights_init_path
    if checkpoint_initial_path is not None:
        model=load_model(checkpoint_initial_path)
    else
        model=ResNet50_model
        model.save(checkpoint_initial_path)
    model_encoding=Model(model.input, model.get_layer('fc1000').output)
    X_train_e=model_encoding.predict(X_train)
    X_train_e=np.reshape(X_train_e,(X_train_e.shape[0],-1))
    K.clear_session()
    accuracies=[]
    queries=[]
    queries.append(labeled_idx)
    results_path=checkpoint_path+'AL_coreset_results.pkl'
    labeled_idx_complete=[]
    labeled_idx_complete.append(labeled_idx)
    with open(results_path, 'wb') as f:
            pickle.dump([accuracies,initial_size,batch_size,queries,labeled_idx_complete], f)
    method=query_coreset(X_train_e,labeled_idx,batch_size)   
    for i in range(iterations):

        path='ResNet50_{idx}.hdf5'.format(idx=i+1)
        checkpoint_path_i=checkpoint_path+path
        old_labeled = np.copy(labeled_idx)
        labeled_idx=method
        labeled_idx_complete.append(labeled_idx)
        new_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, old_labeled))]
        queries.append(new_idx)
        K.clear_session()
        ResNet50_model = applications.ResNet50(input_shape=(256,256,3),include_top=True,weights=None, input_tensor=None, pooling=None,classes=9)
        acc, model = evaluate_sample(ResNet50_model, X_train[labeled_idx,:], Y_train_c[labeled_idx], X_val_b, Y_val_b,X_data, Y_data_c, checkpoint_path_i)
        model_encoding=Model(model.input, model.get_layer('fc1000').output)
        X_train_e=model_encoding.predict(X_train)
        X_train_e=np.reshape(X_train_e,(X_train_e.shape[0],-1))
        accuracies.append(acc)
        K.clear_session()
        print("Test Accuracy Is " + str(acc))
        with open(results_path, 'wb') as f:
            pickle.dump([accuracies,initial_size,batch_size,queries,labeled_idx_complete], f)
        print("----------------------------------------------------")
        print("DATE and TIME of finishing the {ite} iteration =".format(ite = i), datetime.now())
        print("----------------------------------------------------")
