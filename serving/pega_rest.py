#!/usr/bin/env python
#--------------------------- RESTful API for Bug Classification Inference ------------------------------------------------
# The following code loads the pretrained tf-idf model from email bodies and makes it usable standard RESTful API
# to perform classification inference. When running, a user can use a standard POST call and send a single plaintext 
# concatenated bug description and label field. Returns inference, confidence, and a boolean variable indicating success.

# Load Relevant libraries, uniquely the flask web framework used for establishing RESTful API
from waitress import serve
import numpy as np
import pandas as pd
from tensorflow import keras
import sys
import tensorflow as tf
import flask
import pickle
import io

sys.path.insert(1, '../nlp_engine')
from Preprocessing import preprocess_training_text
from MLFunctions import predict_with_uncertainty


#initialize flask application and the keras model
app = flask.Flask(__name__)
model = None

# function to load and return the saved .h5 model (tdidf- classifier)
def load_model():
    # load the pre-trained Keras model
    tf.compat.v1.disable_eager_execution()
    return keras.models.load_model("../saved_models/tfidf_model/tfidf_classifier.h5")

# function to perform standard text preprocessing on submitted text
def prepare_text(text):
    # feed in description + label field and prepare text for inference by preprocessing it
    return preprocess_training_text(text, accented_chars=True, contractions=True, convert_num=False, extra_whitespace=True,
            lemmatization=True, lowercase=True, punctuations=True, remove_html=True, remove_num=True, special_chars=True,
            stop_words=True)

# function to convert text to it's tfidf vector
def tfidf(text):
    # vectorize to tf-idf vectors
    with open("../saved_models/tfidf_model/tfidf_vectorizer.pkl", "rb") as handle:
        tfidf_vectorizer = pickle.load(handle)
    return tfidf_vectorizer.transform(text).toarray()

# function to run prediction. This is called at POST
@app.route("/predict", methods=["POST"])
def predict():
    
    data = {}
    response = {}
    response['success'] =  False

    if flask.request.method == "POST":
        if flask.request.get_data():
            
            tf.compat.v1.disable_eager_execution()
            # read text from a file
            req_data = flask.request.get_json(force=True)
            body = req_data['body']
            subject = req_data['subject']
            text = body + ' ' + subject
            print("Response Recieved: ", text)
            # preprocess text
            text = prepare_text(text)
            # make text into an iterable (a list)
            text = [text]
            # convert text to tf-idf vector
            tfidf_vector = tfidf(text)
            # covert to a data frome
            tfidf_vector = pd.DataFrame(tfidf_vector)
            # use dictionary that maps indices to backlog_ids to list relevant predictions
            with open("../saved_models/tfidf_model/label_to_id.pkl", "rb") as handle:
                label_to_id = pickle.load(handle)
            
            model = load_model()
            # predict result and output:
            data2 = predict_with_uncertainty(model, tfidf_vector, label_to_id, n_iter=100)
            del model
            data['prediction'] = data2[0]
            data['uncertainty'] = data2[1]
            response['data'] = data
            # indicate the request was successful
            response['success'] = True
            
    # return data
    return flask.jsonify(response)

# if this is the main thread of execution start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    tf.compat.v1.disable_eager_execution() 
    serve(app, host='0.0.0.0', port=5000)
