# Tutorial
To run this tutorial you need to run jupyter.
If you run jupyter remotely, you can use jupyter notebook --ip=0.0.0.0
The tutorial should be run on a Linux machine.
Please contact mmalinow@mpi-inf.mpg.de if you encounter any problems.

# Kraino - Keras-based RNN for Visual Turing Test
Keras implementation of the 'Ask Your Neurons'.
 * Free software: MIT license
 * If you use this library, please cite our "Ask Your Neurons" paper [1]
 * Note that we use a simplified version of Kraino for the purpose of the
 Tutorial

## Installation
Requirements:
 * Theano
 * Keras (fchollet)
 * Agnez (EderSantana)
 * toolz
 * h5py
 * Bokeh (0.10.0)
 * sklearn
 * nltk (required by WUPS metrics)
 * pydot
 * spacy
 * seaborn (required by Agnez)
 * sklearn (required by Agnez)
 * matplotlib (required by Agnez) 

Additional:
 * VQA (VT-vision-lab/VQA) for Visual Question Answering 
  * vqaEvaluation for the evaluation metrics
  * vqaTools for the dataset providers
  * both should be placed in the kraino/utils folder


## Folders structure
data/

    daquar/

    vqa/

    ...

kraino/

    local/

        logs/

        weights/

        model-*.pkl

    kraino/

        __init__.py

        core/

        utils/


data 
 * store all datasets

kraino
 * source code and local ouput
 * local
    * stores logs (e.g. predictions) in the 'logs' folder
    * stores weights of different models in the 'weights' folder
    * stores model topologies as '.pkl' files
 * kraino
    * stores the models in the 'core' folder
    * stores functions (dataset providers or callbacks) in the 'utils' folder

## Eras
It counts a computational cycle in eras (not epochs).
Every era ends when "MAX EPOCH" is reached, then the training proceeds to
the next era. Before and after each era the (callback) actions are executed.

## Warning
The framework is under the continous development, and hence it is not warranted 
that API won't change in the future. To avoid adaptations to new API, I
encourage to use a fork from a fixed time stamp.

## Tested on 
 * Python 2.7.3
 * Theano:0.8.0.dev0.dev-709c944030a713d0dd7e1d16d10d99a192a1f716
 * Keras:f2443de96d71b4328d840a6e7e77958025055529

## Bibliography
    @inproceedings{malinowski2015ask,

        title={Ask your neurons: A neural-based approach to answering questions about images},

        author={Malinowski, Mateusz and Rohrbach, Marcus and Fritz, Mario},

        booktitle={Proceedings of the IEEE International Conference on Computer Vision},

        pages={1--9},

        year={2015}

    }
