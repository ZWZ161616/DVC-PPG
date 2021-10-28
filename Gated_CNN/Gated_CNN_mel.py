"""
Created on Fri Jul 26 18:07:45 2019

@author: 70609
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import argparse
import time
#import librosa
from tqdm import tqdm
from natsort import natsorted
import Preprocess_kevin as pre
import glob
#import h5py
from Build_Model import Generator  
import mat73
import scipy.io
#from Tensor_Build_Test import Dnn

def auto_list(directory,nat_sort=False):
    directory_list =[os.path.join(root,file) for root, subdirectories, files in os.walk(directory) for file in files]
    print('\nInput folder:\t{}'.format(directory))
    print('\nTotal number:\t{}\n'.format(len(directory_list)))
    if nat_sort is True:
        return natsorted(directory_list)
    else:
        return directory_list
    
def before_train(train_A_dir, train_B_dir, model_dir, output_dir, tensorboard_log_dir):
    
    sampling_rate = 16000
    ppg_dim = 80
    mel_dim = 80 #24
    frame_period = 5.0

    print('Preprocessing Data...')

    start_time = time.time()
    
    #list_A= pre.load_data_list(train_A_dir)
    #list_B= pre.load_data_list(train_B_dir)
    list_A= pre.load_data_list(train_A_dir)
    list_B= pre.load_data_list(train_B_dir)

#    list_A=glob.glob(train_A_dir)
#    list_B=glob.glob(train_B_dir)
#    print(list_A)
#    print(list_B)
    #scipy.io.savemat('E:\Kevin\org_gatedcnn\mat\Evalu_List_PPG.mat',{'list':list_A})
    #scipy.io.savemat('E:\Kevin\org_gatedcnn\mat\Evalu_List_MEL.mat',{'list':list_B})
    data_ppg = list()
    data_mel = list()
    for i in range(len(list_A)):
        f=mat73.loadmat(list_A[i])
        PPG=f['mel80'][:]#ppgmat
        data_ppg.append(PPG)
    
        #f=np.load(list_B[i])
        #data_mel.append(f)
        f=mat73.loadmat(list_B[i])
        MEL=f['mel80'][:]
        data_mel.append(MEL)
        
    #wavs_A = pre.load_wavs(wav_list = list_A, sr = sampling_rate)
    #wavs_B = pre.load_wavs(wav_list = list_B, sr = sampling_rate)

    # f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = pre.world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    # log_f0s_mean_A, log_f0s_std_A = pre.logf0_statistics(f0s_A)
    # print('Log Pitch A')
    # print('Mean: %f, Std: %f' %(log_f0s_mean_A, log_f0s_std_A))
    
    # f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = pre.world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    # log_f0s_mean_B, log_f0s_std_B = pre.logf0_statistics(f0s_B)
    # print('Log Pitch B')
    # print('Mean: %f, Std: %f' %(log_f0s_mean_B, log_f0s_std_B))


    coded_sps_A_norm = pre.transpose_in_list(lst = data_ppg)#coded_sps_ppg_transposed mel need trans
#    coded_sps_A_norm = data_ppg#coded_sps_ppg_transposed
    coded_sps_B_norm = pre.transpose_in_list(lst = data_mel)#coded_sps_mel_transposed
    
    #coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = pre.coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
    #print("Input data fixed.")
    #coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = pre.coded_sps_normalization_fit_transoform(coded_sps = coded_sps_B_transposed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    #np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A = log_f0s_mean_A, std_A = log_f0s_std_A, mean_B = log_f0s_mean_B, std_B = log_f0s_std_B)
    #np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A = coded_sps_A_mean, std_A = coded_sps_A_std, mean_B = coded_sps_B_mean, std_B = coded_sps_B_std)

    end_time = time.time()
    time_elapsed = end_time - start_time
    
    
    Para_name=['sampling_rate', 'ppg_dim', 'mel_dim', 'frame_period',
               'coded_sps_A_norm', 'coded_sps_B_norm']#, 'coded_sps_A', 'coded_sps_B',
               #'coded_sps_A_mean', 'coded_sps_A_std', 'coded_sps_B_mean', 'coded_sps_B_std',
               #'log_f0s_mean_A', 'log_f0s_std_A', 'log_f0s_mean_B', 'log_f0s_std_B']
    
#    Para_num=len(Para_name) 
    Local_Var=locals()
    Para_data=[Local_Var[para_index] for para_index in Para_name]
    
    Para=dict(zip(Para_name, Para_data))
        
    print('Preprocessing Done.')
    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    return Para





def start_train(model_dir, model_name, initial_model, random_seed, validation_A_dir, output_dir, Para, tensorboard_log_dir, num_epochs = 160):
    
    np.random.seed(random_seed)
    
    globals().update(Para)
    
    num_epochs = num_epochs
    mini_batch_size = 1 # mini_batch_size = 1 is better
    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 200000
    n_frames = 128 #origin 128
    generator = 'generator_gatedcnn'  # gatedcnn dnn
    model = Generator(in_features = ppg_dim, out_features = mel_dim, num_frames = n_frames, ini_weights = None, generator = generator)
    if initial_model:
        model.load(initial_model)
    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)
        '''
        if epoch > 60:
            lambda_identity = 0
        if epoch > 1250:
            generator_learning_rate = max(0, generator_learning_rate - 0.0000002)
            discriminator_learning_rate = max(0, discriminator_learning_rate - 0.0000001)
        '''

        start_time_epoch = time.time()

#        dataset_A2, dataset_B2 = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)
#        dataset_A, dataset_B = pre.sample_parallel_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)
        dataset_A, dataset_B = pre.sample_data(dataset_A = coded_sps_A_norm, 
                                               dataset_B = coded_sps_B_norm,
                                               frames_per_sample = n_frames,
                                               s_type='parallel',
                                               perm=[1,0])

        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):

            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 200000:
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
#                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            if generator == 'generator_dnn':
                sample_A=pre.S_control(dataset_A[start:end],disp=0)
                sample_B=pre.S_control(dataset_B[start:end],disp=0)
                data_A=sample_A.s_transpose(perm=[0,1],disp=0)
                data_B=sample_B.s_transpose(perm=[0,1],disp=0)
            elif generator == 'generator_gatedcnn':
                data_A=dataset_A[start:end]
                data_B=dataset_B[start:end]
                
            generator_loss = model.train(input_A = data_A, input_B = data_B, generator_learning_rate = generator_learning_rate)
#            generator_loss = model.train(input_A = dataset_A[start:end], input_B = dataset_B[start:end], generator_learning_rate = generator_learning_rate)

            if i % 50 == 0:
                #print('Iteration: %d, Generator Loss : %f, Discriminator Loss : %f' % (num_iterations, generator_loss, discriminator_loss))
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Generator Loss : {:.3f}'.format(num_iterations, generator_learning_rate, generator_loss))

        model.save(directory = model_dir, filename = model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch
        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

                
            

        if validation_A_dir is not None:
            validation_A_output_dir = os.path.join(output_dir,model_name[:model_name.find('.ckpt')])
            print(validation_A_output_dir)
            if not os.path.exists(validation_A_output_dir):
                os.makedirs(validation_A_output_dir)
                
            Eva_list_A=pre.load_data_list(validation_A_dir)  
#            tmp=str(validation_A_dir)+'\*.mat'
#            Eva_list_A=glob.glob(tmp)  
            
            if epoch % 50 == 0:
                print('Generating Validation Data B from A...')
                
                for filepath in tqdm(Eva_list_A,desc='Generating'):
                    print(filepath)
                    filedir=os.path.basename(os.path.dirname(filepath))
                    outpath=os.path.join(validation_A_output_dir,filedir)
                    
                    filename = os.path.splitext(os.path.basename(filepath))[0]
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
                        #print("if not")
                        
                     #wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                     #wav = pre.wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                     #f0, timeaxis, sp, ap = pre.world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                     #f0_converted = pre.pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                    #coded_sp = pre.world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    #coded_sp_transposed = coded_sp.T  
                    #coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                    

                    f=mat73.loadmat(filepath)
                    tPPG=f['mel80'][:]#ppgmat
                    coded_sp_norm = tPPG.T
                        
                    
                    if generator == 'generator_dnn':
                        data_Tes_origin = np.array([coded_sp_norm])
                        sample_Tes=pre.S_control(data_Tes_origin)
                        data_Tes=sample_Tes.s_transpose(perm=[0,1])
                        data_Ans = model.test(inputs = data_Tes, direction = 'A2B')
                        _ = sample_Tes.s_update(array_new = data_Ans)
                        coded_sp_converted_norm = sample_Tes.s_transpose(perm=[1,0],disp=0)
                    elif generator == 'generator_gatedcnn':
                        data_Tes=np.array([coded_sp_norm])
                        data_Ans = model.test(inputs = data_Tes, direction = 'A2B')[0]
                        coded_sp_converted_norm = data_Ans
                    #print(coded_sp_converted_norm)
                    coded_sp_converted = coded_sp_converted_norm# * coded_sps_B_std + coded_sps_B_mean
#                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    np.save(outpath+'/'+filename+'.npy',coded_sp_converted)
                    #print(coded_sp_converted)
                    #decoded_sp_converted = pre.world_decode_spectral_envelop(coded_sp = coded_sp_converted.astype('float64'), fs = sampling_rate)
                    #print(decoded_sp_converted)
                    # decoded_list=list()
                    # decoded_list.append(decoded_sp_converted)
#                    A=filepath.find('\Wang')
#                    print(A)
#                    AA=validation_A_output_dir+'/'+filepath[A:-4]
#                    print(AA)
#                    np.save(AA,coded_sp_converted)
                    
                    
                    
                    #wav_transformed = pre.world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    
                    #librosa.output.write_wav(os.path.join(outpath,os.path.basename(filepath)),wav_transformed, sampling_rate)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')
# For Debug default
    train_A_dir_default = './data/Training/Train_mel_List_A.mat'
    train_B_dir_default = './data/Training/Train_mel_List_B.mat'
    model_dir_default = './model/Wang_mel2mel'
    model_name_default = 'Wang_mel2mel.ckpt'
    initial_model_default = None
    random_seed_default = 0
    validation_A_dir_default = r'./data/Evaluating/Evalu_mel_List_out_A.mat'#./mat/Evalu_List_PPG.mat'
    #validation_B_dir_default = './TARGET/Evalu_List_B.mat'
    output_dir_default = './convert_mel/'
    tensorboard_log_dir_default = './log'

#    train_A_dir_default = './data/vcc2016_training/SF1'
#    train_B_dir_default = './data/vcc2016_training/TF2'
#    model_dir_default = './model/sf1_tf2'
#    model_name_default = 'sf1_tf2.ckpt'
#    random_seed_default = 0
#    validation_A_dir_default = './data/evaluation_all/SF1'
#    validation_B_dir_default = './data/evaluation_all/TF2'
#    output_dir_default = './validation_output'
#    tensorboard_log_dir_default = './log'

    parser.add_argument('--train_A_dir', type = str, help = 'Directory for Domain_A.', default = train_A_dir_default)
    parser.add_argument('--train_B_dir', type = str, help = 'Directory for Domain_B.', default = train_B_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--initial_model', type = str, help = 'File name for saving model.', default = initial_model_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type = str, help = 'Convert validation Domain_A after each training epoch. If set none, no conversion would be done during the training.', default = validation_A_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)

    argv = parser.parse_args()
 


#    train_A_dir = argv.train_A_dir
#    train_B_dir = argv.train_B_dir
#    model_dir = argv.model_dir
#    model_name = argv.model_name
#    initial_model = argv.initial_model
#    random_seed = argv.random_seed
#    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
##    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
#    output_dir = argv.output_dir
#    tensorboard_log_dir = argv.tensorboard_log_dir_default
#    
###############################################################################

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    initial_model = argv.initial_model
    num_epochs = 1001
    random_seed = argv.random_seed
    validation_A_dir = argv.validation_A_dir#None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    #validation_B_dir = argv.validation_B_dir#None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir
    
   
    Pre_Data=before_train(train_A_dir = train_A_dir,
                          train_B_dir = train_B_dir,
                          model_dir = model_dir,
                          output_dir = output_dir,
                          tensorboard_log_dir = tensorboard_log_dir)
    
    start_train(model_dir = model_dir, 
                model_name = model_name,
                initial_model = initial_model,
                
                random_seed = random_seed, 
                validation_A_dir = validation_A_dir, 
                output_dir = output_dir, 
                Para=Pre_Data, 
                tensorboard_log_dir = tensorboard_log_dir,
                num_epochs = num_epochs)
