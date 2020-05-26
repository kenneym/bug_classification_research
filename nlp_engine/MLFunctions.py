##########################################
# Machine Learning Functions and Classes #
##########################################

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_history(history, title, save=False, fname=None):

    """
    Pass in a history created by training a keras model,
    and a simple string to represent the title of the figure
    you'd like to display. Note that this function displays an
    accuracy graph and is intended for use with classification models.
    """
    # Clear former plots
    plt.clf()
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [Backlog Assignment] (%)')
    plt.plot(hist['epoch'], hist['accuracy'] * 100,
             label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'] * 100,
             label = 'Validation Accuracy')
    plt.ylim([0,100])
    plt.legend()
    
    
    if save:
        if fname is None:
            fname = './history.png'
        
        plt.savefig(fname)
        

    
def clear_memory(model=None):
    
    """
    This simple clear memory function removes all existing keras models 
    from system (RAM) and/or GPU memeory (VRAM) to prevent memory leaks 
    and to ensure you can load another model without error. You should run
    this clear_memory function every time you want to discard a currently loaded
    model and build a new one. 
    
    If you are running a GPU instance, use the command
    `nvidia-smi` from the bash terminal to determine the ammount of occupied VRAM.
    If you see high memory usage, try using this function to clear the memory. If 
    the problem still persists, a memory leak is likely the culprit, so you will need
    to kill all existing python/ jupyter kernals. If all else fails, a system restart
    will always succeed in clearing the VRAM.
    """
    
    K.clear_session()
    
    if model:
        del model
        
def test_with_uncertainty(model, test_features: pd.DataFrame, test_labels: pd.DataFrame, label_to_id: dict, n_iter=10) -> pd.DataFrame:
    
    """
    USE FOR TESTING
    ---------------

    Parameters:

        - model: a keras model trained with Monte Carlo Dropout

        - test_features: one or multiple samples (1 sample = 1 vector of 
                         test_features).

        - test_labels: the ground-truth labels corresponding to the samples. 
                       (one-hot-encoded)

        - label_to_id: A dictionary that maps prediction indicies (indicies of
                       the softmax outputs of the model) to label names. 

       - n_iter: the number of stochastic forward passes you'd like your model 
                  to perform for each given test sample. i.e. for n_iter=10, the
                  function applies 10 distinct dropout schemas and generates
                  10 different softmax outputs from the model. It then takes the final or 'master'
                  prediction to be the maximum softmax index with the highest overall probability
                  (i.e. final prediction = average prediction over all 10 trials) and the uncertainty 
                  to be the Standard Deviation of the set of 10 softmax outputs (more variable softmax output 
                  = more uncertainty)
    
    Returns: preds_df, a dataframe of labels predicted by your model, ground truth lables, and prediction
             uncertainty values.
    """
    
    # Define the testing function to use dropout (testing function is just a forward pass through training function)
    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    
    num_classes = len(label_to_id)
    result = np.zeros((n_iter,) + (test_features.shape[0], num_classes) )
    preds_df = pd.DataFrame(columns = ['Prediction', 'Label', 'Uncertainty'])
    
   # Generate n_iter different softmax outputs by altering dropout each pass through
    for i in range(n_iter):
        result[i,:, :] = f((test_features, 1))[0]

    predictions = result.mean(axis=0) # “ultimate prediction”
    uncertainties = result.std(axis=0) # “STD as proxy for uncertainty”
    
    for i in range(len(predictions)):
        
        one_hot_label = test_labels.iloc[i] # Get the ground truth one-hot-encoded label for this prediction
        predicted_index = np.argmax(predictions[i]) # Predicted index is maximum softmax ouput probability
        predicted_value = label_to_id[predicted_index] # convert predicted index to label name
        actual_value = one_hot_label.idxmax()
        uncertainty = uncertainties[i][predicted_index]
        preds_df = preds_df.append({'Prediction' : predicted_value , 'Label' : actual_value, 'Uncertainty' : uncertainty} , ignore_index=True)

    return preds_df


def predict_with_uncertainty(model, test_features: pd.DataFrame, label_to_id: dict, n_iter=10) -> tuple:
    
    """
    USE FOR DEPLOYMENT
    ------------------

    Wheras the above function is used during the testing phase (testing monte 
    carlo accuracies), this function should be used for deployment. Note that
    there are no ground-truth labels passed into the function.

    Parameters:

        - model: a keras model trained with Monte Carlo Dropout

        - test_features: a SINGLE sample (1 sample = 1 vector of 
                         test_features).

        - label_to_id: A dictionary that maps prediction indicies (indicies of
                       the softmax outputs of the model) to label names. 

       - n_iter: the number of stochastic forward passes you'd like your model 
                  to perform for each given test sample. i.e. for n_iter=10, the
                  function applies 10 distinct dropout schemas and generates
                  10 different softmax outputs from the model. It then takes the final or 'master'
                  prediction to be the maximum softmax index with the highest overall probability
                  (i.e. final prediction = average prediction over all 10 trials) and the uncertainty 
                  to be the Standard Deviation of the set of 10 softmax outputs (more variable softmax output 
                  = more uncertainty)
   
    Returns: a tuple, including the predicted label value and an uncertainty value
             ... i.e. return  (prediction id, uncertainty)
    """
    # from tensorflow.keras.models import load_model
    # tf.compat.v1.disable_eager_execution()
    # 
    # model = load_model("../saved_models/tfidf-model/tfidf-classifier.h5")
    
    # Define the testing function to use dropout (testing function is just a forward pass through training function)
    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    
    num_classes = len(label_to_id)
    result = np.zeros((n_iter,) + (test_features.shape[0], num_classes) )
    
    test_features = test_features.values.reshape(1,-1) # reshape single input sample to matrix so model can parse it
    
   # Generate n_iter different softmax outputs by altering dropout each pass through
    for i in range(n_iter):
        result[i,:, :] = f((test_features, 1))[0]

    predictions = result.mean(axis=0) # “ultimate prediction”
    uncertainties = result.std(axis=0) # “STD as proxy for uncertainty”
    
    predicted_index = np.argmax(predictions[0])
    predicted_value = label_to_id[predicted_index] # convert predicted index to label name
    uncertainty = uncertainties[0][predicted_index]

    return predicted_value, uncertainty



def get_monte_carlo_accuracy(preds_df: pd.DataFrame, threshold=None) -> tuple:
    """
    Takes as input a preds_df (Dataframe) generated by the predict_with_uncertainty
    function.This function will assess the models accuracy only on test examples 
    below `threshold` uncertainty.

    Returns: accuracy and percentage of the orignal dataset that is retained once
             threshold is applied.
    """
    orig_length = len(preds_df)
    retained = 100 # If no data is lost, we've retained 100% of testing data

    # Evaluate only confident predictions
    if threshold:
        preds_df = preds_df[preds_df['Uncertainty'] < threshold]
        
        # Avoid Divide-by-Zero error:
        if len(preds_df) == 0:
            retained = 0
        else:
            retained = len(preds_df) / orig_length * 100
        
    good_preds = preds_df[preds_df['Prediction'] == preds_df['Label']]
    accuracy = len(good_preds) / len(preds_df) * 100
    
    return retained, accuracy


def graph_confidence(preds_df, save=False, fname=None):
    """
    Takes as input a preds_df (Dataframe) generated by the predict_with_uncertainty
    function. This functions plugs in a large range of threshold values to the 
    `get_monte_carlo_accuracy` function in order to generate and plot a graph of 
    the trade-off between accuracy and effeciency.
    
    Returns the accuracy of the model when it discards about 50% of the test data.
    This information can be useful for evaluating how 'good' a given model is at
    confidence bounding.
    
    """
    # Clear former plots
    plt.clf()
    
    uncertanties = np.logspace(0, 10, 10000, base=0.5).tolist() # Logspace generates a better distribution of values since
                                                               # uncertainy values are based on STD, which is distributed more logarithmically
                                                               # than linearly - in short, this distribution produces a much better graph.

    accuracies = []
    proportions = []
    for uncertainty in uncertanties:
        retained, accuracy = (get_monte_carlo_accuracy(preds_df, uncertainty))
        proportions.append(retained)
        accuracies.append(accuracy)
        
    proportions[0] = 100 # Show 100% accuracy on 0% of the training data
    
    
    plt.scatter(proportions, accuracies,s=5, c='c')
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Percentage of Testing Data Retained")
    plt.title("Monte Carlo Accuracies")
        
    # Save for later use:
    if save:
        if fname is None:
            fname = '../figures/monte-carlo-accuracies.png'
            
        plt.savefig(fname) 
        
        
    # Return accuracy on 50% of training data:
    for i, proportion in enumerate(proportions):
        
        # Return as soon as we dip below 50
        if proportion < 50:
            return accuracies[i]
        

def acc_to_uncertainty(preds_df, target_acc):
    
    """ 
        Finds the uncertainty value that most closely approximates the desired accuracy ('target_acc')
        (i.e. returns the unvertainty value that throws away the minimum ammount of
         test data while still producing the desired accuracy)
         
        **PASS IN value for 'target_acc' out of 100 (true percent)!
    """
    uncertainties = np.logspace(0, 10, 10000, base=0.5).tolist()
    
    for uncertainty in uncertainties:
        
        retained, accuracy = (get_monte_carlo_accuracy(preds_df, uncertainty))
        
        # Return as soon as we reach the desired accuracy
        if accuracy > target_acc:
            print('% of test data retained', retained)
            print('Exact accuracy', accuracy)
            return uncertainty
    
    
def proportion_to_uncertainty(preds_df, target_prop):
    
    """ 
        Finds the maximum uncertainty value that results in the target proprotion ('target_prop')
        of the test data being retained.
        
        Ex: target_prop = 70%
        Returns: The uncertainty value that causes the model to discard approx. 30% of the data.
        
        **PASS IN value for 'target_prop' out of 100 (true percent)!

    """
    uncertainties = np.logspace(0, 10, 10000, base=0.5).tolist()
    
    for uncertainty in uncertainties:
        
        retained, accuracy = (get_monte_carlo_accuracy(preds_df, uncertainty))
        
        # Return as soon as we drop barely below the target proportion
        if retained < target_prop:
            print('% of test data retained', retained)
            print('Exact accuracy', accuracy)
            return uncertainty
    
    
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
       
