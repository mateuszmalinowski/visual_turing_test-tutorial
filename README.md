# Tutorial
To run this tutorial you need to run jupyter.

If you run jupyter remotely, you can use jupyter notebook --ip=0.0.0.0

Main file: visual_turing_test.ipynb
The reader is however encouraged to download the notebook together
with the associated files and go through the tutorial on his own.

The tutorial should be run on a Linux machine.
Please also make sure that all Installation requirements are fullfiled
and you have similar versions of Theano and Keras (see 'Tested on').

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
 * toolz
 * h5py
 * Bokeh (0.10.0)
 * nltk (required by WUPS metrics)
 * pydot
 * spacy

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
that API won't change in the future. To avoid adaptations to new API, you can 
clone from a specific commit hash.

## Tested on 
 * Python 2.7.3
 * Theano:0.8.0.dev0.dev-63990436c98f107cf120f3578021a5d259ecf352
 * Keras:b587aeee1c1be3633a56b945af3e7c2c303369ca

## Bibliography

   @article{malinowski2016ask,
   
       title={Ask Your Neurons: A Deep Learning Approach to Visual Question Answering},
       
       author={Malinowski, Mateusz and Rohrbach, Marcus and Fritz, Mario},
  
       journal={arXiv preprint arXiv:1605.02697},
       
       year={2016}
       
    }

    @inproceedings{malinowski2015ask,

        title={Ask your neurons: A neural-based approach to answering questions about images},

        author={Malinowski, Mateusz and Rohrbach, Marcus and Fritz, Mario},

        booktitle={Proceedings of the IEEE International Conference on Computer Vision},

        pages={1--9},

        year={2015}

    }

    @inproceedings{malinowski2014multi,
    
      title={A multi-world approach to question answering about real-world scenes based on uncertain input},
      
      author={Malinowski, Mateusz and Fritz, Mario},
      
      booktitle={Advances in Neural Information Processing Systems},
      
      pages={1682--1690},
      
      year={2014}
      
    }
    
    @article{malinowski2016tutorial,
    
      title={Tutorial on Answering Questions about Images with Deep Learning},
      
      author={Malinowski, Mateusz and Fritz, Mario},
      
      journal={arXiv preprint arXiv:1610.01076},
      
      year={2016}
      
    }