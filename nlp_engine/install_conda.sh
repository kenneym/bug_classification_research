#!/bin/bash
curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -bu
rm -rf /tmp/miniconda.sh

conda=$HOME/miniconda3/bin/conda
MYENV=nlp-workspace
ipython=$HOME/miniconda3/envs/nlp-workspace/bin/ipython

# Update and initialize env
$conda update -n base -c defaults conda -y
$conda create -n "$MYENV" python=3.7 -y 
$conda init bash

# Install Ipykernel to access this env in Jupyter notebook
$conda activate "$MYENV"
$conda install -n "$MYENV" ipykernel -y
$ipython kernel install --user --name="$MYENV"
