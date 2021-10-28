# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:26:08 2019

@author: 70609
"""

import tensorflow as tf
import os
import random
import numpy as np


def mean_absolute_error(y_true, y_pred):

    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mean_square_error(y_true, y_pred):

    return tf.reduce_mean(tf.square(y_true - y_pred))
    
def cross_entropy_loss(logits, labels):
        
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    
mse = MSE = mean_square_error
mae = MAE = mean_absolute_error
