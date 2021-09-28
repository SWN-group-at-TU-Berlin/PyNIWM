#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Riccardo Taormina
"""
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

def build_ann(n_classes, neurons_per_layer, act_fun, dropout_rate):
  model = Sequential()  
  for l,nH in enumerate(neurons_per_layer):
    if l == 0:
      # first layer
      model.add(Dense(nH,activation=act_fun))
    else:
      # other layers
      model.add(Dropout(dropout_rate))
      model.add(Dense(nH,activation=act_fun))
  model.add(Dropout(dropout_rate))
  model.add(Dense(n_classes, activation='softmax'))
  model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
  
  return model



class ANNClassifier(KerasClassifier):
  """ 
    Modification of implementation of the scikit-learn classifier API for Keras.
  """
  def __init__(self, **sk_params):
    super().__init__(build_ann, **sk_params)

  def summary(self):
    print(self.model.summary())
    
  # def predict(self):
  #  temp = super().predict()
  #  return 