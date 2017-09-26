"""Finetuning example.

Trains the DeepMoji model on the SCv2-GEN sarcasm dataset, using the 'last'
finetuning method and the class average F1 metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.

The class average F1 metric does the following:
1) For each class, relabel the dataset into binary classification
   (belongs to/does not belong to this class).
2) Calculate F1 score for each class.
3) Compute the average of all F1 scores.
"""

from __future__ import print_function
import example_helper
import json
import os
from deepmoji.finetuning import load_benchmark, finetune
from deepmoji.class_avg_finetuning import class_avg_finetune
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH

import numpy as np

DATASET_PATH = '../data/'

GLOBAL_SAVE_PATH = '../model/{}_weights.hdf5'

CLASSES_NB = {'kaggle-insults': 2, 'Olympic': 4, 'PsychExp': 7,
              'SCv1': 2, 'SCv2-GEN': 2, 'SE0714': 3,
              'SS-Twitter': 2, 'SS-Youtube': 2}

MEASURE = {'kaggle-insults': 'acc', 'Olympic': 'f1', 'PsychExp': 'f1',
              'SCv1': 'f1', 'SCv2-GEN': 'f1', 'SE0714': 'f1',
              'SS-Twitter': 'acc', 'SS-Youtube': 'acc'}

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

data_files = []
for dirpath, _, filenames in os.walk(DATASET_PATH):
    dataset_name = dirpath.split('/')[-1]
    fn = filenames[0]
    if dataset_name != '' and fn == 'raw.pickle' and dataset_name in CLASSES_NB:
        data_files.append((dataset_name, dirpath+'/'+fn, CLASSES_NB[dataset_name], MEASURE[dataset_name]))

for (dataset_name, data_file, nb_classes, measure) in data_files:
    save_path = GLOBAL_SAVE_PATH.format(dataset_name)
    print("Training model on", dataset_name, "located in", data_file,
          "with", nb_classes, "classes, measuring", measure, "and saving in", save_path)

    # Load dataset. Don't extend the existing vocabulary (we want a signle voc for all tasks)
    data = load_benchmark(data_file, vocab, extend_with=0)

    # Add singleton dim if needed
    for i in range(len(data['labels'])):
        if len(data['labels'][i].shape) == 1:
            print("Adding axis")
            data['labels'][i] = data['labels'][i][:, np.newaxis]

    for ls in data['labels']:
        print("ls.shape", ls.shape)

    if measure == 'f1':
        # Set up model and finetune. Note that we have to extend the embedding layer
        # with the number of tokens added to the vocabulary.
        #
        # Also note that when using class average F1 to evaluate, the model has to be
        # defined with two classes, since the model will be trained for each class
        # separately.
        model = deepmoji_transfer(2, data['maxlen'], PRETRAINED_PATH,
                                extend_embedding=data['added'])
        model.summary()

        # For finetuning however, pass in the actual number of classes.
        model, f1 = class_avg_finetune(model, data['texts'], data['labels'],
                                       nb_classes, data['batch_size'], method='last')
        print("Training finished, final class average F1:", f1)
    else:
        model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH,
                                extend_embedding=data['added'])
        model.summary()

        # Finetuning and measuring result on accuracy
        model, acc = finetune(model, data['texts'], data['labels'],
                     nb_classes, data['batch_size'], method='last')
        print("Training finished, final accuracy:", acc)

    model.save_weights(save_path)
    print("Saved model in", save_path)
