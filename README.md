# <span style="color: darkturquoise;">Automated Bug Assignment for Agile Studio using Natural Language Processing </span>
**Authors: Matt Kenney and Jake Epstein**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kenneym/bug_classification_research/master)

## <span style="color: orange;">Preface</span>

### Project Significance:
The NLP model we built processes bug reports submitted to Pegasystems Inc. It reads in said reports in email (HTML) format, processes them and analyzes them, and attempts to identify specifically which engineering team would be best suited to solve the bug at hand. In order to accomplish this task, our model must identify words and phrases within the email that can help identify where exactly the customer's problem may be originating with respect to Pegasystems' technology stack. If a bug report contained a significant number of terms related to cloud technology, for example, our classifier may classify that bug as being best addressed by the cloud engineering team at Pegasystems. Likewise, if a bug report contained references to UI elements ('button', 'sidebar', etc.), our classifier may determine that the bug is best addressed by the UI engineering team at Pega. To accomplish this task, a deep learning model was constructed to assign any given bug report to 1of 47 different engineering teams.

Prior to the creation of this machine learning model, Pegasystems employees relied on manually assigning bug reports to the relevant engineering teams so that bugs could be investigated and resolved. This machine learning model will help to free up human work-hours for other tasks at Pegasystems, to prevent the accumulation of unresolved bugs (by assigning bugs immediately after they are recieved), and to address potential issues in Pega's technology stack more quickly and effectively.

Because assigning a bug report to the incorrect engineering team would result in additional work and inconvenience for Pegasystems engineers, we constructed a system for confidence bounding that allows the model to notify its users about how confident it is in making a classification. Using this system for confidence bounding, we allow our model to assign bugs directly to the relevant engineering team when it is confident in making a prediction, and pass bugs for which it is unsure of the correct classification to a queue for human review. This confidence bounding system allows for an ideal balance between machine-driven automation and human intuition.

Although natural language processing algorithms abound in this day and age, we believe that our algorithm is unique for a variety of reasons. Firstly, it operates on a domain-specific corpora that is specific to a single company (i.e. the model must learn associations between pegasystems-specific technology terms). Creating a model that works on such an usual set of terminology was a formidable challenge. Secondly, and more importantly, our model utilizes a novel confidence bounding approach termed [“Monte Carlo Dropout”](https://arxiv.org/pdf/1506.02142.pdf), a technique introduced in 2016 which relies on measuring the variance in the softmax output of a multiple-classification deep learning algorithm to determine the model’s confidence about any given prediction. Monte-Carlo dropout has yet to be incorporated into any major machine learning frameworks (Tensorflow, Caffe, PyTorch, etc.), and although there are several discussions on Github “issues” pages, and a few limited mentions of utilizing Monte Carlo Dropout in the Tensorflow framework, we believe that our code represents one of the first implementations of Monte-Carlo dropout in the Tensorflow language and using a production-quality model. Because confidence bounding is still a relatively obscure technique (surprisingly, most production models have no methodology to control for erroneous prediction) and because there is such limited documentation on how to use Monte Carlo dropout in Tensorflow, we built a variety of functions into a python library that make using MC dropout in Tensorflow straightforward and intelligible. We plan to publish these libraries to the wider public to allow other Tensorflow users to more easily build confidence-bounding into their models. 

Especially in cases in which model accuracy is of utmost importance, or in cases in which bias is a factor, closely monitoring model confidence is of serious importance. That being said, we believe the research community is in desperate need of accessible and straightforward methodologies for model confidence bounding… as well as other methods for model monitoring (ex: explainability metrics, bias metrics, etc.). We hope that our project and the library we will be publishing shortly to work with Monte Carlo Dropout in Tensorflow will help the machine learning research community to better address these issues.


### A Note on the Codebase:

The model we built is currently being placed into production at pegasystems with the help of a devops team. In order to prepare this model for production, we wrote significantly more code than is shown in this repository to allow for the following features. 

- Allow the model to be retrained periodically on new bug report data collected from consumers. 
- Automation scripts to obtain this data and retrain the model on a specified schedule without any human intervention necessary.
- Allow the model to take on new backlog IDs (representing new technology teams), or remove obsolete ones as needed.
- Production grade REST API that allows the model to be utilized by client applications in production.

Because these aspects of the code would uncover specific details about the deployment of our code on Pegasystems servers, we redacted these elements of our code base and returned our codebase to represent just our research efforts. Production-ready code is not included for security reasons. In addition, we redacted any elements of the code that were deemed to be proprietary and unsuitable for outside viewing.

Finally, the data that was used to construct the machine learning model itself is proprietary (it consists of actual bug reports sent from pega users and is marked as confidential) and therefore cannot be shared online. 

We've added a 'launch binder' button that allows you to open up this codebase and view all of the included jupyter notebooks in a cloud-hosted jupyter environment for your convenience.


# README:

## <span style="color: orange;">Summary</span>

The following project uses a Natural Language Processing model to read in bug descriptions submitted by the Global Customer Support Desk, internal Pega employees, and Pega customers, and assign these bugs to the relevant backlogs in Agile management tool so that scrum teams automatically receive only bugs that are relevant to them.

To perform bug assignment, we first pre-process bug descriptions and subject lines. Then, we convert each bug description to a TF/IDF vector. Finally, we use a Deep Learning model to extract insights from the TF/IDF scores and train that model to perform accurate classifications.

At the prediction phase, we apply confidence bounding using [Monte Carlo Dropout](https://arxiv.org/pdf/1506.02142.pdf). Using this approach, we can choose the threshold confidence score that best suits our needs - the model will automatically route bugs to the appropriate backlogs if it can generate a prediction above the confidence threshold (a high confidence prediction) and will route bugs that it cannot confidently place into a queue for human review. Stakeholders at Pega can choose a proper trade off between accuracy and efficiency by tweaking the confidence threshold to the desired value. **Monte Carlo Dropout is a relatively new technique for confidence bounding model predictions, and this code represents one of the first implementations of Monte-Carlo dropout in the Tensorflow language and using a production-quality model.**

If you'd like to learn more about how we built the model, or want to gain the necessary background to make tweaks of your own, we've documented our entire model design process in the [TF/IDF model ipython notebook](./training/tfidf_model.ipynb). **We recommend reading this file to better understand our work.** The most important considerations outlined in this notebook come at the end of the file, where we discuss our confidence bounding approach.

---

For a broader summary of our work, and to learn about the various approaches we took prior to settling on the TFIDF/ Deep Learning approach, check out the PDF of our [PowerPoint presentation](bug_triage_presentation.pdf)- which covers our design and development process in depth, or watch our 30 minute [recorded presentation](./bug_classification_presentation.mp4).

### Embedding Models:
If you're interested in the embedding approaches we've taken, check out the `./embedding_models` directory. We build the corpora and Word2Vec models needed for embedding-based models in [`buildw2vs`](./embedding_models/build_w2vs.ipynb), and construct deep learning models using those W2V models as input layers in [`embedding_models`](./embedding_models/embedding_models.ipynb). Finally, we built a [`hybrid model`](./embedding_models/embedding_models.ipynb) which takes both TFIDF vectors and embedding vectors as input.

These notebooks may prove helpful if you're trying to build an embedding model of your own. Unfortunately, for this specific use case, we found that TFIDF outperformed our embedding models.

## <span style="color: orange;">Directory Structure </span>

Data for the project is not tracked on GitHub, but all Jupyter Notebooks and python scripts are present.
- Find notebooks we've used to analyze email body data using TF/IDF and Hashing approaches, as well as our automated training script in the [`training`](./training)  directory.
- Find notebooks we've used to analyze email body data using embedding approaches in the [`embedding_models`](./embedding_models) directory.
- All python packages and virtual environment configurations for this work are stored in [`nlp_engine`](./nlp_engine).
- RESTful API application for deployment of this project is stored in [`serving`](./serving).
- There are several empty folders in this project, including `saved_models` and `data`. These folders are kept in the repository so that our automation scripts and jupyter notebook files can safely save and load data and models to these locations without encountering 'directory not found' errors.


## <span style="color: orange;">Getting Set Up:</span>

**You may open and run the jupyter notebooks or python files in this repository at the Binder link above (see icon at top of github page)**. Binder is a service that builds Docker containers out of github repositories, and hosts them for free online, installing all neccessary dependencies. The binder machine is intended for you to easily explore the code, but should not be used in production.

In order to set up your machine, you'll need to install conda, setup an environment, and install the relevant packages. We have scripts that make each of these processes easy. Before we lay them out, some considerations:

- For training the model, you may want to consider using a GPU machine loaded with the NVIDIA CUDA library. You will only need to train the model once, so the cost of using a GPU machine would be trivial.
- After you finish training, you'll have a set of reload-able files that you can use to reload the model anywhere, so the training machine need not be the same machine as the deployment machine.
- For the deployment machine, you should be just fine with a standard CPU machine, and likely won't even need much memory. When selecting a deployment machine, you only need to ensure that
	1. The machine can stay running
	2. You can open ports on that machine to accept REST requests (typically port 5000)


### Set Up Procedure:

1. Install the conda environment by running `bash ./nlp_engine/install_conda.sh`. This script installs [miniconda](https://docs.conda.io/en/latest/miniconda.html) and sets up an environment called `nlp-workspace`. You can learn more about conda environments [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). The environment essentially acts as a sand-boxed virtual machine where we keep all of the code dependencies. As you can see by inspecting the install script, we use python 3.7 as our base.
	- Note that if you already have conda installed on your machine, and prefer to use the version of conda that is already installed, simply create a new environment using conda named `nlp-workspace` and activate it. You should then be able to follow the next steps without issue.

2. Run `source ~/.bashrc` in order to make sure conda is registered by your shell. You should now see the word `(base)` to the left of your command line, indicating that conda has activated the `base` environment.

3. Run `conda activate nlp-workspace` to enable your shell to access all python libraries and configuration that we will install into this `nlp-workspace` environment.

4. Run `bash ./nlp_engine/install_packages.sh` to install the relevant packages into this environment. 
    - Note that if you are on a GPU Machine, you will need to check your cuda version using the command `nvcc --version` on the command line. If your cuda version is 10.0, then you're all set. If not, however, you'll need to edit the `install_packages.sh` file, and change the spacy cuda version to the version of cuda that you are running. We've set the spacy version to `spacy[cuda100]` by default, representing cuda version 10.0. You should change this line to represent whatever version of cuda you have (i.e. cuda 10.1 -> `spacy[cuda101]`) All packages other than spacy automatically detect cuda version and thus do not require additional configuration.

5. To run the jupyter notebooks in this repository, activate the nlp-workspace (`conda activate nlp-workspace`), and then run `jupyter lab` from the terminal. Because data is not tracked in this repository, you'll need to add a bug data csv file to the `./data/csvs/` directory you just creted in order for the notebooks in this repository to work correctly. Name the bug file `bug_emails.csv` and ensure it includes at least 5 columns: `Work ID`, `Label`, `Description`, `Backlog ID`, and `Team ID`. **Columns must be named exactly as described, and the CSV file must be UTF-8 Encoded.** If you're working with an excel file, export it to a CSV (UTF-8) format rather than a standard CSV.

6. To run the python files in this repository, simply `conda activate nlp-workspace` and run them directly `python file.py ... (cli args)`

7. If you want to install additional packages, you may not be able to do so from within a jupyter notebook. Instead, simply open up a terminal, activate the environment, and then `pip install` whichever package(s) you need. You can check whether or not you're using the right `pip` package by typing `which pip`. You'll want to make sure you're using the `pip` package that is specific to the `nlp-workspace` environment, which should look something like `HOME/miniconda3/envs/nlp-workspace/bin/pip`


## <span style="color: orange;">Running the Training Script</span>

To train the model and save it in a reloadable format to be used later in deployment, use the `train-model.py` script located in `./training`. To run the training script, simply pass it a CSV of bug descriptions and assigned labels. As input, this script takes a CSV file that includes at least 5 columns: `Work ID`, `Label`, `Description`, `Backlog ID`, and `Team ID`. **Columns must be named exactly as described, and the CSV file must be UTF-8 Encoded.** If you're working with an excel file, export it to a CSV (UTF-8) format rather than a standard CSV.

This script will re-initialize and train the same model architecture 10 times in a row. We train multiple times because the training process in machine learning is non-deterministic, and can yield varying accuracy and confidence results depending on model initialization and training. We evaluate model quality based on model accuracy when the uncertainty threshold is set in the config file `config.ini`. If the user does not set model accuracy in this config file, it is defaulted to 80\% and the model discards whatever amount of the test set is needed to achieve this accuracy. **We are making the assessment that models that are better at bounding their own confidence, even at the expense of overall accuracy, are superior for a production environment.** We noticed that models tend to have very little variability in overall confidence each training trial, but have significant variability in their ability to bound their own confidence - this is the main observation underpinning our assessment of how to best evaluate 'model quality'.

This script will save the TFIDF Vectorizer used to extract features from the data, the (best) keras bug classifier itself, and a dictionary that converts prediction indices to actual backlog id names (allowing us to understand the model's predictions in production) to the directory '../saved_models/tfidf_model/' (unless another directory is specified). 

The RESTful API script described below will load this trained model from the saved model directory to generate predictions in production.

**IMPORTANT NOTES:**
1. To use the script with defaults, run `python train-model.py -f PATH_TO_CSV`
2. Run `python train-model.py -h` to view the script description and see the available command line options.
3. You *must* run this file from within the `training` directory for imports to work correctly.
4. **Upon completion, the script will create a `results` directory as a child of the directory you ran the script in. Open up the `results.md` file in that directory to view model performance on the validation and test set and view useful figures summarizing the model's performance.**
5. To determine the best threshold uncertainty score to use in production, we reccomend loading the model in a jupyter notebook, and using the `acc_to_uncertainty` or `proportion_to_uncertainty` functions defined in our [`MLFunctions` library](./nlp_engine/MLFunctions.py) to select an uncertainty threshold based on a target accuracy, or a target ammount of data we want to retain in production (the percentage of the bug classification that will be automated by the model). You can then play around with different threshold scores and determine the right balance between efficiency and accuracy. Although we'd like a more programatic way to select the threshold, uncertainty scores are based on standard deviations in model predictions, and are thus unitless and difficult to interpret. *You can learn more about how to find the optimal uncertainty score and see how to utilize the `get_monte_carlo_accuracy` function by checking out the "Testing Model + Generating Predictions" section of our [TF/IDF model notebook](./bug_classification/tfidf_model.ipynb).* The model is defaulted to train using the \% of the dataset that achieves 80\% accuracy. To change the target accuracy, edit DESIRED_MODEL_ACC in config.ini


## <span style="color: orange;">RESTful API</span>

The RESTful API is to be used as a means of deploying the bug classification model in production. Using a POST, a user can send a single description + label plaintext. The API preprocesses the text from the request and vectorizes it using TF-IDF. It then loads the pre-trained model saved as a .h5 (tfidf-classifier.h5 in 'saved_models/tfidf-model/tfidf_classifier.h5'). It attempts to perform inference by passing the loaded model and other parameters to the function `predict_with_uncertainty`, located in our [MLFunctions](./nlp_engine/MLFunctions.py) module. A dictionary is returned as the response. On failure, the response is `"success":false`. When successful, an example response is `{"prediction":"BL-14","success":true,"uncertainty":0.07847238542679846}` where "prediction" corresponds to the predicted backlog id and "uncertainty" corresponds to the standard-deviation-based uncertainty metric.

### Example

*POST request:*
`curl -X POST -H "Content-Type: text/plain" --data "the cloud is malfunctioning" http://localhost:5000/predict`

*Response:*
`{"prediction":"BL-3339","success":true,"uncertainty":0.020733406494500232}`

*Note:*
BL-3339 -> Cloud Backlog

### How To Use

The REST API script can be found at `'serving/pega_rest.py'`. To use it, simply...
- set up a server to host the prediction model
- open up port 5000 on that server
- run the `pega_rest.py` script on that server (`python pega_rest.py` from the terminal)
- send plaintext POST requests to the REST API endpoint URL. This can easily be automated using a simple python script that passes a request payload and returns the prediction in whatever format is most convenient. An example POST request in python:

```python
KERAS_REST_API_URL = http://remote-server-domain-name:5000/predict`
payload = "the cloud is malfunctioning"
r = request.post(KERAS_REST_API_URL, payload).json()
```
