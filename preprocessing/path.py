import pandas as pd
import numpy as np
import math
import tensorflow as tf
from tfrecorder import TFrecorder
import struct
import os
import subprocess
from subprocess import check_output
import scipy.io.wavfile as wav
#import speechpy
PATH = '/USERS/d8182103/'
tfr = TFrecorder()
def create_parser(data_info, retrieve_shape):

    names = data_info['name']
    types = data_info['type']
    shapes = data_info['shape']
    isbytes = data_info['isbyte']
    defaults = data_info['default']
    length_types = data_info['length_type']

    def parser(example_proto):
        def specify_features():
            specified_features = {}
            for i in np.arange(len(names)):
                # which type
                if isbytes[i]:
                    t = tf.string
                    s = ()
                else:
                    if types[i]=='uint8':
                        types[i]=tf.uint8
                    t = types[i]
                    s = shapes[i]
                # has default_value?
                if defaults[i] == np.NaN:
                    d = np.NaN
                else:
                    d = defaults[i]
                # length varies
                if length_types[i] =='fixed':
                    specified_features[names[i]] = tf.FixedLenFeature(s, t)
                elif length_types[i] =='var':
                    specified_features[names[i]] = tf.VarLenFeature(t)
                else:
                    raise TypeError("length_type is not one of 'var', 'fixed'")
            return specified_features


        # decode each parsed feature and reshape
        def decode_reshape():
            # store all decoded&shaped features
            final_features = {}
            for i in np.arange(len(names)):
                # exclude shape info
                if '_shape' not in names[i]:
                    # decode
                    if isbytes[i]:
                        # from byte format
                        decoded_value = tf.decode_raw(parsed_example[names[i]], types[i])
                        decoded_value = tf.cast(decoded_value, tf.float32)
                    else:
                        # Varlen value needs to be converted to dense format
                        if length_types[i] == 'var':
                            decoded_value = tf.sparse_tensor_to_dense(parsed_example[names[i]])
                        else:
                            decoded_value = parsed_example[names[i]]
                    # reshape
                    if '%s_shape' %names[i] in parsed_example.keys():
                        tf_shape = parsed_example['%s_shape' %names[i]]
                        decoded_value = tf.reshape(decoded_value, tf_shape)
                    final_features[names[i]] = decoded_value
                elif retrieve_shape:
                    final_features[names[i]] = parsed_example[names[i]]
            return final_features


        # create a dictionary to specify how to parse each feature 
        specified_features = specify_features()
        # parse all features of an example
        parsed_example = tf.parse_single_example(example_proto, specified_features)
        final_features = decode_reshape()

        return final_features
    return parser


def get_dataset(paths, data_info, retrieve_shape = False, num_parallel_calls=4, prefetch_buffer=2):

    filenames = paths
    
    data_info = pd.read_csv(data_info,dtype={'isbyte':bool})
    data_info['shape']=data_info['shape'].apply(lambda s: [int(i) for i in s[1:-1].split(',') if i !=''])

    #print(data_info)
    dataset = tf.data.TFRecordDataset(filenames)
    parse_function = create_parser(data_info, retrieve_shape)
    dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls).prefetch(prefetch_buffer)
    return dataset
def fb_extractor(file_name):
    datapath = '/USERS/d8182103/'
    modalities = ['images']
    videoshape= [-1,224,224,3]
    videofeature='320x180'
    video_padding_info = {videofeature:videoshape}
    video_padding_value = {videofeature:0.0}
    video_info_path = '%s/firstimpression%s/tfrecord/images/%s.csv' %(datapath,dataset, videofeature)
    video_dataset_path = '%s/firstimpression%s/tfrecord/images/%s/%s.tfrecord' %(datapath, dataset, setname, videofeature)
    video_dataset = get_dataset(paths=video_dataset_path, data_info=video_info_path,  num_parallel_calls=1, prefetch_buffer=1)
    
    return logenergy_feature_cube_cmvn
def path_wav_list(dataset='V2',setname='train',feature_type = '.wav'):
    audio_feature_path = '%s/firstimpression%s/%s/' %(PATH, dataset, setname)
    print(audio_feature_path)
    wav_list = []
    name_list = []
    for folder in os.listdir(audio_feature_path):
        if '80_' in folder:
            for feature in os.listdir('%s/%s/' %(audio_feature_path,folder)):
                if '.wav' in feature:
                    name_list.append(feature.split('.wav')[0])
                    wav_list.append('%s/%s/%s' %(audio_feature_path,folder,feature))
    path_df = pd.DataFrame({'wav':name_list,'path':wav_list},columns=['wav','path'])
    return path_df.set_index('wav')
def path_list(dataset='V2',setname='train',feature_type = 'mfcc'):
    audio_feature_path = '%s/firstimpression%s/%s/%s/' %(PATH, dataset, feature_type, setname)
    print(audio_feature_path)
    video_list = []
    path_list = []
    for folder in os.listdir(audio_feature_path):
        if '80_' in folder:
            for feature in os.listdir('%s/%s/' %(audio_feature_path,folder)):
                if '.htk' in feature:
                    video_list.append(feature.split('.htk')[0])
                    path_list.append('%s/%s/%s' %(audio_feature_path,folder,feature))
    path_df = pd.DataFrame({'video':video_list,'path':path_list},columns=['video','path'])
    return path_df.set_index('video')
def get_dataframe(df,dataset,setname):
    # read file containing the video name that will not be used
    delete_list = PATH+('firstimpression%s/tfrecord/text/%s/%s_videos_without_transcription.txt' %(dataset,setname,setname))
    if dataset =='V2':
        df = df.drop(columns=['text'])
    try:
        with open(delete_list,'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print('firstimpression V1')
    else:
        # get indexes that should be removed from datafram
        remove_indexes = []
        for l in lines:
            remove_indexes.append(df[df['video']==l[:-1]].index[0])
        # drop rows of those indexes
        df = df.drop(np.array(remove_indexes))
    return df
feature_name = 'raw'
def save_audio_feature(dataset,setname):
    # read dataframe
    df = pd.read_csv(PATH+'firstimpression%s/%s/text_and_labels.csv' %(dataset,setname))
    # get correct df
    video_df = get_dataframe(df,dataset, setname)
    path_df = path_wav_list(dataset=dataset,setname=setname)

    num_examples = len( video_df)
    
    data_info = pd.DataFrame({'name':[feature_name,feature_name+'_shape'],
                                 'type':['float32','int64'],
                                 'shape':[(1,),(2,)],
                                 'isbyte':[True,False],
                                 "length_type":['fixed','fixed'],
                                 "default":[np.NaN,np.NaN]})

    data_info_path = PATH+'firstimpression%s/tfrecord/audio/%s_info.csv' %(dataset,feature_name)
    data_info.to_csv(data_info_path,index=False)
    
    tfr = TFrecorder()
    writer = tf.python_io.TFRecordWriter(PATH+'firstimpression%s/tfrecord/audio/%s/%s.tfrecord' %(dataset,setname,feature_name))
    
    for i in np.arange(num_examples):
        features = {}
        
        video = video_df.iloc[i]['video'].split('.mp4')[0]
        wav_path = path_df.loc[video]['path']
        print(wav_path)
        #mfcc_file_name = PATH+'firstimpression%s/mfcc_dir/%s/%s.mfc' %(dataset,setname,video) #path_df.loc[video]['path']
        '''
        #audio_features = read_htk(mfcc_file_name)[0]
        audio_features = fb_extractor(wav_path)
        audio_features = np.array(audio_features,dtype='float32')
        print(audio_features.shape)
        #sentence_based_norm = (audio_features-audio_features.mean())/audio_features.std()
        #sentence_based_norm = np.array(sentence_based_norm,dtype='float32')
        
        
        tfr.feature_writer(data_info.iloc[0], audio_features.reshape(-1), features)
        tfr.feature_writer(data_info.iloc[1], audio_features.shape, features)
        
        tf_features = tf.train.Features(feature= features)
        tf_example = tf.train.Example(features = tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
        print(i)
        '''
        
    writer.close()
for dataset in ['V2']:
    for setname in ['train','vali']:
        save_audio_feature(dataset,setname)
        print(dataset,setname)