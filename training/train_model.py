#!/usr/bin/env python

# Data Manipulation:
import pandas as pd
import numpy as np

# Feature Extraction
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

# Models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Flatten, Input, Activation, PReLU, LeakyReLU, ThresholdedReLU

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tabulate import tabulate

# Utilities
import configparser
import pickle
import multiprocessing
import argparse, sys
from pathlib import Path
from tqdm import tqdm

# Logging
import logging


# Custom Libraries
sys.path.insert(1, '../nlp_engine')
from MLFunctions import plot_history, clear_memory, test_with_uncertainty, \
                        predict_with_uncertainty, get_monte_carlo_accuracy, graph_confidence, \
                        acc_to_uncertainty
from Preprocessing import preprocess_training_text


description = '''

TFIDF Trainer
------------
Script to train our tfidf classifier with a csv as input.


Author: Matt Kenney
Contact: mtnkenney2@gmail.com
         matthew.kenney@pega.com (will be terminated after internship)

Description
-----------

o As input, this script takes a csv file that includes at least 5 columns: 'Work ID', 'Label', 'Description', and 'Backlog ID', and 'Team ID'. Columns must be named exactly as described. Pass this file in using the `-f` specifier.

o This script will save the TFIDF Vectorizer used to extract features from the data, the keras bug classifier itself, 
  and a dictionary that converts prediction indicies to actual backlog id names (allowing us to understand the model's predictions
  in production) to the directory '../saved_models/tfidf-model/' (unless another directory is specified)

o This script will preprocess the bugs contained in the passed csv using our NLP preproccessing library, and save a dataframe of these 
  processed bugs in the file '../data/pickles/preprocessed-bugs.pkl'. If you've preprocessed the bugs previously, use the option --pickle to 
  load your existing preprocessed bugs pickle file. Note that you must preprocess your bugs over again as soon as the CSV file changes.

o If you'd prefer to save the model to some other directory, use the option --model-dir to choose your directory.

o If you'd prefer to save the preprocessed data to some other directory, use the option --data-dir to choose your directory.


Example Runs:
------------

    To train with a csv file: 
        `python train-model.py -f bugs.csv`
    To train the model with a pickled dataframe containing text that has already been preprocessed:
    `python train-model.py -p bugs.csv`

'''


def process_data(csv: str, data_dir: str) -> pd.DataFrame:

    """ Preprocesses the csv using our Preprocessing library in nlp_engine.
        Saves preprocessed data for later use & returns a dataframe containing
        the data.
    """

    # Read in Data and rename to snake case:
    email_bugs = pd.read_csv(csv)
    email_bugs = email_bugs.rename(columns={"Work ID" : "work_id", 
                                            "Label" : "label", 
                                            "Description" : "description", 
                                            "Team ID" : "team_id", 
                                            "Backlog ID" : "backlog_id"})

    ## DATA CLEANING ##
    # 1. Eliminate:
    assigned_bugs = email_bugs[email_bugs['backlog_id'].notnull()]   # bugs that were not assigned to a backlog
    assigned_bugs = assigned_bugs[assigned_bugs['team_id'].notnull()]  # bugs that lack a team
    assigned_bugs = assigned_bugs[assigned_bugs['description'].notnull()]  # bugs that lack a description
    assigned_bugs = assigned_bugs[assigned_bugs['label'].notnull()]        # or label
    
    # 2. Drop Duplicates:
    assigned_bugs.drop_duplicates(subset = "work_id", keep = False, inplace = True) # duplicate bugs

    # 3. remove backlogs that have been assigned 40 or less training examples:
    former_length = len(assigned_bugs)
    bugs_per_id = assigned_bugs['backlog_id'].value_counts()
    valid_id_list = bugs_per_id[bugs_per_id > 40].index.tolist()
    assigned_bugs = assigned_bugs[assigned_bugs['backlog_id'].isin(valid_id_list)]


    # TEXT PREPROCESSING ##
    tqdm.pandas(desc="preproccessing descriptions")
    assigned_bugs['description'] = assigned_bugs['description'].progress_apply(preprocess_training_text)
    tqdm.pandas(desc="preproccessing labels")
    assigned_bugs['label'] = assigned_bugs['label'].progress_apply(preprocess_training_text)

    # 5. Concatenate bug labels and descriptions, and create a `combined` data column in our `assigned_bugs` dataframe.
    assigned_bugs['combined'] = assigned_bugs['label'].map(str) + ' ' + assigned_bugs['description']
    category='combined'

    # Additional Step: Remove all bugs that contain 10 or less words in the description:
    assigned_bugs = assigned_bugs[assigned_bugs['description'].str.split().apply(len) > 10]
    print("Number of training samples remaining: " , len(assigned_bugs))

    # Save the preprocessed bugs
    data_file = data_dir + "preprocessed_bugs.pkl"
    pd.to_pickle(assigned_bugs, data_file)

    return assigned_bugs


def load_data(pickle: str):
    """Loads the data from the specified pickle file."""

    return pd.read_pickle(pickle)


def fit_tfidf(assigned_bugs: pd.DataFrame, model_dir: str):
    """ Generates a TFIDF representation of the bug desciption + label fields,
        and returns a train test split of the resulting data
    """

    # 1. Get the one-hot-encoded representation of the backlog ids to which each bug belongs.
    backlog_labels = pd.get_dummies(assigned_bugs['backlog_id'])
    
    # 2. Perform a train test split on our text and labels to seperate out train data and test data.
    category='combined'
    train_features, test_features, train_labels, test_labels = train_test_split(assigned_bugs[category], backlog_labels, test_size=0.2)
    
    # 3. Use a TF/IDF Vectorizer to convert plain text descprtions into TF/IDF vectors.
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,
                            binary=False,
                            min_df=3,
                            max_df=0.5, 
                            norm='l2', 
                            ngram_range=(1, 2),
                            lowercase=True)
    
    train_features = pd.DataFrame(tfidf_vectorizer.fit_transform(train_features).toarray()) # Fit the Vectorizer to the train data
    test_features = pd.DataFrame(tfidf_vectorizer.transform(test_features).toarray()) # Only transform (don't fit) the test data to emulate real-world predictions
    
    # 4. Convert all datatypes to float32
    train_features = train_features.astype('float32')
    test_features = test_features.astype('float32')
    train_labels = train_labels.astype('float32')
    test_labels = test_labels.astype('float32')

    # Notify user of new tfidf size:
    logging.warning("TFIDF Vectors may grow to an astronomical size over time "
                   "as more data is pulled from Agile Studio. If the vectors "
                   "grow too large, they may overload system memory. If this "
                   "ever occurs, you should consider discarding older bugs from the "
                   "training dataset, and attempt to retrain the model using this "
                   "reduced dataset.")
    logging.info("Current TFIDF Vector Size: "+ str(train_features.shape[1]))
    logging.info("Current number of backlogs this classifier will classify into :"+ str(train_labels.shape[1]))
                
    
    # Save tfidf_vectorizer
    tfidf_file = model_dir + "tfidf_vectorizer.pkl"
    pickle.dump(tfidf_vectorizer, open(tfidf_file, "wb"))

    return train_features, test_features, train_labels, test_labels


# Function to build our TF/IDF model:
def build_tfidf_model(features, labels, optimizer, activations, drop_rate, lr, layer1_size, 
                      layer2_size=None, layer3_size=None, layer4_size=None):
    
    model = Sequential()
    model.add(Input(shape=[len(features.keys())], name="TFIDF_Features"))
    model.add(Dense(layer1_size, input_shape=[len(features.keys())]))
    
    if activations[0] == "leaky":
        model.add(LeakyReLU())
    elif activations[0] == "prelu":
        model.add(PReLU())
    else:
        model.add(Activation(activations[0]))     
        
    if layer2_size:
        model.add(Dropout(drop_rate, trainable=True))
        model.add(Dense(layer2_size))
        
        if activations[1] == "leaky":
            model.add(LeakyReLU())
        elif activations[0] == "prelu":
            model.add(PReLU())
        else:
            model.add(Activation(activations[1]))      
            
    if layer3_size:
        model.add(Dropout(drop_rate, trainable=True))
        model.add(Dense(layer3_size))
        
        if activations[2] == "leaky":
            model.add(LeakyReLU())
        elif activations[0] == "prelu":
            model.add(PReLU())
        else:
            model.add(Activation(activations[2]))   
            
    if layer4_size:
        model.add(Dropout(drop_rate, trainable=True))
        model.add(Dense(layer4_size))
        
        if activations[3] == "leaky":
            model.add(LeakyReLU())
        elif activations[0] == "prelu":
            model.add(PReLU())
        else:
            model.add(Activation(activations[3]))
            
            
    model.add(Dropout(drop_rate, trainable=True))
    model.add(Dense(len(labels.keys()), activation='softmax', name="softmax_output"))
    
    # Parameters
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizer == 'adamax':
        optimizer = tf.keras.optimizers.Adamax() # Use default learning rate for adamax
    elif optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(lr)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr )
    else:
        print("ERROR: No valid optimizer passed")
        return None
    
    
    model.compile(loss='kullback_leibler_divergence',
                optimizer=optimizer,
                metrics=['accuracy'])
    
    return model


def train_model(train_features, test_features, train_labels, test_labels, 
                model_dir, num_trials, desired_acc):
    """ Train the model & save it as an HD5. Return it so that this file can test
        the model performance.

        returns: The threshold uncertainty value that will produce the desired accuracy
                 listed in the config file... or 1, if no desired accuracy is stated.
                 (all predictions fall under uncertainty threshold of 1)
    """
    # Generate a dictionary that maps prediction indicies 
    # (indicies of one-hot-encoded labels) to actual backlog ID names 
    label_to_id = {}
    for i, col in enumerate(test_labels.columns):
        label_to_id[i] = col
    
    # Save:
    dict_file = model_dir + "label_to_id.pkl"
    pickle.dump(label_to_id, open(dict_file, 'wb'))
    
    # Train multiple models and take the best of that trial:
    # - if there is no desired acc, we take the most accurate model overall
    # - if there is a desired acc, we take the model that can achieve that accuracy
    #   by throwing away the LEAST ammount of data
    max_test_acc = 0
    max_val_acc = 0
    max_data_retained = 0
    final_threshold = 1.0

    for i in range(num_trials):

        # Construct the model itself:
        assignment_model = build_tfidf_model(features=train_features, 
                                             labels=train_labels, 
                                             optimizer='adam',
                                             activations=['prelu', 'prelu'],
                                             drop_rate=0.3,
                                             lr=0.0001, 
                                             layer1_size=2048,
                                             layer2_size=512, 
                                             layer3_size=None,
                                             layer4_size=None)


        # Define Params for this model:
        workers = multiprocessing.cpu_count()
        use_multiprocessing = workers > 1
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=30)
        
        # Train
        validation_acc = 0 # initialize
        result = assignment_model.fit(
                 train_features, train_labels,
                 batch_size=128, # default size
                 epochs=1000, validation_split=0.1, verbose=1,
                 callbacks=[early_stop],
                 workers=workers, use_multiprocessing=use_multiprocessing) 

        
        # Report Results:
        mc_accuracies = './results/mc-accuracy.png'
        history = './results/history.png'

        # Get Metrics:
        validation_acc = np.amax(result.history['val_accuracy'])
        preds_df = test_with_uncertainty(assignment_model, test_features, test_labels, label_to_id, n_iter=100)
        _, test_accuracy = get_monte_carlo_accuracy(preds_df=preds_df, threshold=None) # get overall acc

        # Get Plots
        plot_history(result, "Model Training Record", save=True, fname=history)
        graph_confidence(preds_df, save=True, fname=mc_accuracies)


        # Determine if this model is the 'best' so far:

        # Select model that can achieve the desired accuracy by throwing away the least ammt of data:
        if desired_acc is not None:
            threshold = acc_to_uncertainty(preds_df, desired_acc) # get the uncertainty threshold that produces 
                                                                  # the desired acc
            data_retained, _ = get_monte_carlo_accuracy(preds_df=preds_df, threshold=threshold)
            best_model = data_retained > max_data_retained

            if best_model:
                # Reset max data retained:
                max_data_retained = data_retained
                final_threshold = threshold
                print('New max data retained', max_data_retained)

                # Record these values as well for record keeping
                max_test_acc = test_accuracy
                max_val_acc = validation_acc


        # If no desired acc is found, select most accurate model overall:
        else:
            best_model = test_accuracy > max_test_acc
            if best_model:
                max_test_acc = test_accuracy
                max_val_acc = validation_acc # record val acc also for record keeping purposes
                print('New max test acc', max_test_acc)


        if best_model:
            print("Better Model Found")

            # Generate Markdown Rendering of Results:
            results = open('./results/results.md', 'w') 

            print('# Model Performance Results', file=results)

            print('\n## Metrics:', file=results)
            print('- Validation Accuracy: %', validation_acc, file=results)
            print('- Test accuracy (no confidence bounds): %', test_accuracy, file=results)

            print('\n## Example Predictions:', file=results)
            print(tabulate(preds_df[:10], tablefmt="pipe", headers="keys"), file=results)

            print('\n## Training History:', file=results)
            print('![training history](' + './history.png' + ')', file=results)

            print('\n## Efficiency/ Accuracy Trade-Off (Confidence Bounding):', file=results)
            print('![accuracy plot](' + './mc-accuracy.png' + ')', file=results)

            results.close()

            # Save Model:
            model_file = model_dir + "tfidf_classifier.h5"
            assignment_model.save(model_file)


    # Log Performance once entire training process is complete:
    logging.info("Best model's performance on the validation set: " + str(max_val_acc))
    logging.info("Best model's performance on the test set: " + str(max_test_acc))
    logging.info("Threshold uncertainty value to be used in production: " + str(final_threshold))
    
    if desired_acc is not None:
        logging.info("% of emails the model could confidently route using the desired accuracy: " + \
                     str(max_data_retained))


    return final_threshold


def training_process(training_csv):
    """ 
    This function allows a user to import this python script and call
    it directly without commmandline args

    :param training_csv: the name of the csv to train the model on
    """
    # Load data
    model_dir = "../saved_models/tfidf_model/"
    data_dir = "../data/pickles/"
    results_dir = "./results/"
    assigned_bugs = process_data(training_csv, data_dir)

    # Generate TFIDF Representation:
    train_features, test_features, train_labels, test_labels = fit_tfidf(assigned_bugs, model_dir)

    # Select number of training trials
    config = configparser.ConfigParser()
    config.read('../config.ini')

    num_trials = 1 # train the model only once (for speed) in the production environment
    if config['APP']['ENVIRONMENT'] == 'production':
        num_trials = 10

    desired_acc = None
    if config['APP']['DESIRED_MODEL_ACC'] != 'None':
        desired_acc = config.getint('APP', 'DESIRED_MODEL_ACC')


    # Train and save the model itself. Return uncertainty threshold to use in production:
    return train_model(train_features, test_features, train_labels, test_labels, model_dir, num_trials, desired_acc)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(prog="Automated Training File for TFIDF Email Classifier",
                                     description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-f', '--file', type=str,
                        help="The name of csv file containing bug descriptions and assignments", dest="csv")
    parser.add_argument('-p', '--pickle', type=str,
                        help="The name of pickle file containing preprocessed bug descriptions and their assignments", dest="pickle")
    parser.add_argument('-md', '--model-dir', type=str,
                        help="Use this option if you'd like to specify a model directory other than the default", dest="model_dir")
    parser.add_argument('-dd', '--data-dir', type=str,
                        help="Use this option if you'd like to specify a data directory other than the default ", dest="data_dir")
    

    # Check and parse command line args:
    args = parser.parse_args()
    model_dir = "../saved_models/tfidf_model/"
    data_dir = "../data/pickles/"
    results_dir = "./results/"
    assigned_bugs = None
    
    if not args.csv and not args.pickle:
        parser.print_help()
        print("No CSV or Pickle File Found. Please pass a csv or pickle file to train the model.")
        sys.exit(1)

    elif args.csv and args.pickle:
        parser.print_help()
        print("Passed Both CSV and Pickle. Please pass only one file to this training script")
        sys.exit(1)

    else:
        if args.model_dir:
            model_dir = args.model_dir
            
        if args.data_dir:
            data_dir = args.data_dir
    
    # Format Directories and Notify User
    if model_dir[-1] != '/':
        model_dir += '/'

    if data_dir[-1] != '/':
        data_dir += '/'

    print("Model Dir is", model_dir)
    print("Data Dir is", data_dir)

    # Create directories if they don't yet exist:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path('./results/').mkdir(parents=True, exist_ok=True)

    # Show User Tensorflow Information:
    tf.compat.v1.disable_eager_execution()
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
    
    # Load data
    if args.csv:
        assigned_bugs = process_data(args.csv, data_dir)
    else:
        assigned_bugs = load_data(args.pickle)

    # Send log messages to stdout, since this script was called via cmd line:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Generate TFIDF Representation:
    train_features, test_features, train_labels, test_labels = fit_tfidf(assigned_bugs, model_dir)

    # Select number of training trials
    config = configparser.ConfigParser()
    config.read('../config.ini')

    num_trials = 1 # train the model only once (for speed) in the development environment
    if config['APP']['ENVIRONMENT'] == 'production':
        num_trials = 10

    
    desired_acc = None
    if config['APP']['DESIRED_MODEL_ACC'] != 'None':
        desired_acc = config.getint('APP', 'DESIRED_MODEL_ACC')

    # Train and save the model itself:
    train_model(train_features, test_features, train_labels, test_labels, model_dir, num_trials, desired_acc)

