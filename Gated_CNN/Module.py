# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:33:46 2019

@author: 70609
"""

import tensorflow as tf 

def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer

def pixel_shuffler(inputs, shuffle_size = 2, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

def dense_layer(
    inputs,
    units,
    activation = None,
    kernel_initializer = None,
    name = None):
    
    
#    dense_layer = tf.keras.layers.Dense(
#        units = units,
#        activation = activation,
#        kernel_initializer = kernel_initializer,
#        name = name)

    dense_layer = tf.layers.dense(
        inputs = inputs,
        units = units,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)
    
    return dense_layer

def gru_layer(
    inputs,
    units,
    activation = None,
    kernel_initializer = None,
    name = None):
    
    gru_layer = tf.keras.layers.GRU(
    units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    recurrent_constraint=None, bias_constraint=None, dropout=0.0,
    recurrent_dropout=0.0, implementation=1, return_sequences=False,
    return_state=False, go_backwards=False, stateful=False, unroll=False,
    reset_after=False, **kwargs
)


    return gru_layer

def conv1d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def conv2d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer
    
def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)

    return activation

def downsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def downsample2d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def residual1d_block(
    inputs, 
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residule_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3

def upsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    shuffle_size = 2,
    name_prefix = 'upsample1d_block_'):
    
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def generator_gatedcnn(inputs, out_features, reuse = False, scope_name = 'generator_gatedcnn'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 3, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 3, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = 3, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 3, strides = 2, name_prefix = 'downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # Upsample
        u1 = upsample1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 512, kernel_size = 3, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs = u2, filters = out_features, kernel_size = 3, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2
    


def generator_dnn(inputs, out_features, reuse = False, scope_name = 'generator_dnn'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size(frame_per_sample), num_features] for Dnn
    inputs = tf.transpose(inputs, perm = [0, 1], name = 'inputs_transpose')
    
#    batch_shape=inputs.get_shape().as_list()
#    batch_shape[batch_shape.index(None)]=-1
#    batch_shape[batch_shape.index(None)]=-1
    
#    inputs_reshape = tf.reshape(inputs, [-1,batch_shape[batch_shape.index(num_features)]],name = 'inputs_reshape')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

#        h1 = dense_layer(inputs = inputs_reshape, units = 1024, activation = tf.sigmoid, name = 'h1_dense')
        h1 = dense_layer(inputs = inputs, units = 512, activation = tf.sigmoid, name = 'h1_dense')
        h2 = dense_layer(inputs = h1, units = 512, activation = tf.sigmoid, name = 'h2_dense')
        h3 = dense_layer(inputs = h2, units = 512, activation = tf.sigmoid, name = 'h3_dense')
        h4 = dense_layer(inputs = h3, units = 512, activation = tf.sigmoid, name = 'h4_dense')
        h5 = dense_layer(inputs = h4, units = 512, activation = tf.sigmoid, name = 'h5_dense')
        
        # Output
        o1 = dense_layer(inputs = h5, units = out_features, activation = None, name = 'o1_dense')
        outputs = tf.transpose(o1, perm = [0, 1], name = 'outputs_transpose')        
#        outputs_reshape = tf.reshape(o1, batch_shape,name = 'outputs_reshape')
#        outputs = tf.transpose(outputs_reshape, perm = [0, 1], name = 'outputs_transpose')
        
    return outputs

def discriminator_1d(inputs, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size(frame_per_sample), num_features]
    # we need to add channel for 1D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample1d_block2_')
        d3 = downsample1d_block(inputs = d2, filters = 1024, kernel_size = [6, 3], strides = [1, 2], name_prefix = 'downsample1d_block3_')

        # Output
        o1 = tf.layers.dense(inputs = d3, units = 1, activation = tf.sigmoid)

        return o1

def discriminator_2d(inputs, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        d3 = downsample2d_block(inputs = d2, filters = 1024, kernel_size = [6, 3], strides = [1, 2], name_prefix = 'downsample2d_block3_')

        # Output
        o1 = tf.layers.dense(inputs = d3, units = 1, activation = tf.sigmoid)

        return o1