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
    p.add_argument('--output_path', '-op', type=str,default=None)
    p.add_argument('method', type=str,
                   choices={'Random','LC','Entropy','BALD'},
                   help="sampling method ('Random','LC','Entropy','BALD')")
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
def query_random(X_train_e,labeled_idx, amount):
    """
    Returns randomly selected samples
    """
        unlabeled_idx = get_unlabeled_idx(X_train_e, labeled_idx)
        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))
def query_LC(X_train,labeled_idx,amount,ResNet50_model):
    """
    Returns samples based on Least Confidence
    """

     unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
     predictions = ResNet50_model.predict(X_train[unlabeled_idx, :])
     unlabeled_predictions = np.amax(predictions, axis=1)

     selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
     return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
def query_uncertainityentropy(X_train,labeled_idx,amount,ResNet50_model):
    """
    Returns samples with maximum predictive entropy
    """
     unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
     predictions = ResNet50_model.predict(X_train[unlabeled_idx, :])
     unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)

     selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
     return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
def dropout_predict(data):

    f = K.function([model.layers[0].input, K.learning_phase()],
                   [model.layers[-1].output])
    predictions = np.zeros((T, data.shape[0], num_labels))
    for t in range(T):
        predictions[t,:,:] = f([data, 1])[0]
    expected_entropy = - np.mean(np.sum(predictions * np.log(predictions + 1e-10), axis=-1), axis=0)  # [batch size]
    expected_p = np.mean(predictions, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    BALD_acq = entropy_expected_p - expected_entropy

    final_prediction = np.mean(predictions, axis=0)
    prediction_uncertainty = BALD_acq

    return final_prediction, prediction_uncertainty

def query_bald(X_train,labeled_idx, amount):
    """
    Returns samples based on BALD acquisition
    """

    unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

    predictions = np.zeros((unlabeled_idx.shape[0], num_labels))
    uncertainties = np.zeros((unlabeled_idx.shape[0]))
    i = 0
    split = 128  # split into iterations of 128 due to memory constraints
    while i < unlabeled_idx.shape[0]:

        if i+split > unlabeled_idx.shape[0]:
            preds, unc = dropout_predict(X_train[unlabeled_idx[i:], :])
            predictions[i:] = preds
            uncertainties[i:] = unc
        else:
            preds, unc = dropout_predict(X_train[unlabeled_idx[i:i+split], :])
            predictions[i:i+split] = preds
            uncertainties[i:i+split] = unc
        i += split

    
    selected_indices = np.argpartition(-uncertainties, amount)[:amount]
    return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
    
    return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))

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
    accuracies=[]
    queries=[]
    queries.append(labeled_idx)
    results_path=checkpoint_path+'AL_results.pkl'
    labeled_idx_complete=[]
    labeled_idx_complete.append(labeled_idx)
    with open(results_path, 'wb') as f:
            pickle.dump([accuracies,initial_size,batch_size,queries,labeled_idx_complete], f)
    if args.method == 'Random':
        method = query_random(X_train_e,labeled_idx,batch_size)
    elif args.method == 'LC':
        method = query_LC(X_train,labeled_idx,batch_size,model)
    elif args.method == 'Entropy':
        method = query_uncertainityentropy(X_train,labeled_idx,batch_size,model)
    elif args.method == 'BALD':
        method = query_bald(X_train,labeled_idx,batch_size)
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
        
        accuracies.append(acc)
        
        print("Test Accuracy Is " + str(acc))
        with open(results_path, 'wb') as f:
            pickle.dump([accuracies,initial_size,batch_size,queries,labeled_idx_complete], f)
        print("----------------------------------------------------")
        print("DATE and TIME of finishing the {ite} iteration =".format(ite = i), datetime.now())
        print("----------------------------------------------------")
