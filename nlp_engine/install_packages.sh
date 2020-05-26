#!/bin/bash
pip=$HOME/miniconda3/envs/nlp-workspace/bin/pip
python=$HOME/miniconda3/envs/nlp-workspace/bin/python

# Uncomment the following to install pydot and graphviz -- neccessary for generating figures
# of keras models. 

# Make sure sudo is installed to ensure that we can perform proper installs regardless of whether or not user is root
#if sudo
#then
#	echo sudo already installed
#else
#	echo installing sudo
#	apt install -y sudo
#fi

#sudo apt update 
#sudo apt install -y python-pydot python-pydot-ng graphviz
# ---


# If the machine has a CUDA compatable GPU (and CUDA has been installed):
if nvcc --version
then
        echo Installing Packages for GPU Machine
        yes | $pip install tensorflow-gpu==2.0.1 \
                           tensorflow-hub \
                           tensorflow_datasets \
                           spacy[cuda100] \
                           gensim \
                           pandas \
                           numpy \
                           matplotlib \
                           seaborn \
                           sklearn \
                           bs4 \
                           text-unidecode \
                           word2number \
                           graphviz \
                           pydot \
                           keras-mxnet \
                           hyperas \
                           jupyterlab \
                           validator_collection \
                           nltk \
                           langdetect \
                           flask \
			   waitress \
			   tabulate \
			   schedule

# If the machine is CPU-only
else
        echo Installing Packages for CPU-only Machine
        yes | $pip install tensorflow==2.0.1 \
                           tensorflow-hub \
                           tensorflow_datasets \
                           spacy \
                           gensim \
                           pandas \
                           numpy \
                           matplotlib \
                           seaborn \
                           sklearn \
                           bs4 \
                           text-unidecode \
                           word2number \
                           graphviz \
                           pydot \
                           keras-mxnet \
                           hyperas \
                           jupyterlab \
                           validator_collection \
                           nltk \
                           langdetect \
                           flask \
			   waitress \
                           tabulate \
			   schedule
fi

$python -m spacy download en_core_web_sm
