# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:07:45 2019

@author: 70609
"""
import os
 #0: GPU2, 1: GPU3, 2: GPU0, 3: GPU1
import tensorflow as tf
from Module import generator_gatedcnn, generator_dnn, discriminator_1d, discriminator_2d
#from GATECNN_Utils import LossFunc
from Utils import *
from datetime import datetime

Gpu_Control = tf.GPUOptions(allow_growth=True) # G_RAM will auto control it's range
sess = tf.Session(config = tf.ConfigProto(gpu_options=Gpu_Control))

#tf.keras.backend.set_session(sess)

class Generator(object):
    
    
    def __init__(self, in_features, out_features, num_frames, generator = 'generator_gatedcnn',discriminator = None, mode = 'train', ini_weights = None, log_dir = './log'):

        self.in_features = in_features
        self.out_features = out_features
        self.num_frames = num_frames
        self.mode = mode
        
              
        if generator in ['generator_gatedcnn','gatedcnn','Gatedcnn','GatedCnn','GatedCNN']:
            self.generator = generator_gatedcnn
            self.input_shape = [None, in_features, None] # [batch_size, in_features, num_frames]
            self.output_shape = [None, out_features, None] # [batch_size, out_features, num_frames]
        elif generator in ['generator_dnn','dnn','Dnn','DNN']:
            self.generator = generator_dnn
            self.input_shape = [None, in_features] # [batch_size*num_frames, in_features]
            self.output_shape = [None, out_features] # [bbatch_size*num_frames, out_features]
        else:
            raise KeyError("There is no module call: %s" %(generator))
                   
        if discriminator:
            print("Training Cycle-Gan...\nGenerator: %s\nDiscriminator: %s" %(generator,discriminator))
            if discriminator in ['discriminator_2d','discriminator_2D','Conv_2D','Conv_2d','conv_2D','conv_2d']:    
                self.discriminator = discriminator_2d               
            elif discriminator in ['discriminator_1d','discriminator_1D','Conv_1D','Conv_1d','conv_1D','conv_1d']:
                self.discriminator = discriminator_1d            
            else:
                raise KeyError("There is no module call: %s" %(discriminator))   
            
        else:
            print("Training generator only...\nGenerator: %s" %(generator))

        
        
        if tf.get_default_graph().get_operations():
            print('Graph Initializing...')
            tf.reset_default_graph()
        
        self.build_model()
        self.optimizer_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        
        if ini_weights:
            print("Loading Model...")
            assert type(ini_weights) == str          
            self.saver.restore(self.sess, ini_weights)
            print("Model restored Finish!")      
            
        self.sess.run(tf.global_variables_initializer())
#        self.sess.run()
            


        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            if discriminator:
                self.generator_summaries, self.discriminator_summaries = self.summary()
            else:
                self.generator_summaries = self.summary()
            
    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape = self.output_shape, name = 'input_B_real')
        
        self.input_A_test = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_A_test')


        self.generation_A2B = self.generator(
                inputs = self.input_A_real,
                out_features=self.out_features,
                reuse = False,
                scope_name = 'generator_A2B')

        # Generator loss
        # Generator wants to fool discriminator
#        self.generator_loss_A2B = LossFunc().mean_square_error(y_true = self.input_B_real, y_pred = self.generation_A2B)
        self.generator_loss_A2B = mse(y_true = self.input_B_real, y_pred = self.generation_A2B)

        # Merge the two generators and the cycle loss
        self.generator_loss = self.generator_loss_A2B

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        #for var in t_vars: print(var.name)

        # Reserved for test
        self.generationA2B_test = self.generator(inputs = self.input_A_test,
                                                 out_features=self.out_features,
                                                 reuse = True,
                                                 scope_name = 'generator_A2B')
        
    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, None, name = 'generator_learning_rate')
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate = self.generator_learning_rate, beta1 = 0.5).minimize(self.generator_loss_A2B, var_list = self.generator_vars) 
        
    def train(self, input_A, input_B, generator_learning_rate):

        generation_A2B, generator_loss_A2B, _, generator_summaries = self.sess.run(
            [self.generation_A2B, self.generator_loss, self.generator_optimizer, self.generator_summaries], \
            feed_dict = {self.input_A_real: input_A, self.input_B_real: input_B, self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        self.train_step += 1

        return generator_loss_A2B
    
    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generationA2B_test, feed_dict = {self.input_A_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)
        
    def summary(self):

        with tf.name_scope('generator_summaries'):
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
        return generator_loss_A2B_summary
    
if __name__ == '__main__':
    
    # model = Generator(in_features = 24, out_features = 24,num_frames = 128, ini_weights = './model/gatedcnn_wang/gatedcnn_wang.ckpt', generator = 'gatedcnn')
    # model = Generator(in_features = 297, out_features = 24,num_frames = 128, ini_weights = None, generator = 'gatedcnn')
    model = Generator(in_features = 297,out_features = 80,num_frames = 128, ini_weights = './model/mel80_vc/mel80_vc.ckpt', generator = 'gatedcnn')
    # model = Generator(in_features = 297,out_features = 80,num_frames = 128, ini_weights = './model/mel80_vc/mel80_vc.ckpt', generator = 'dnn')
    print('Graph Compile Successeded.')