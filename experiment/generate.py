import tensorflow as tf
import learning
import numpy as np


#tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("C", 0, "whether continue to train")
tf.app.flags.DEFINE_integer("E", 40, "epoch")
tf.app.flags.DEFINE_string("M", "train", "train, predict, eval")
datapath = '/USERS/d8182103/'
logpath='/USERS/d8182103/best/'
dataset = 'V2'

#*********************************
def main(unused_argv):
    def setparam():
        model_params = tf.contrib.training.HParams(
                                archi = archi,
                                modalities = modalities,
                                num_layers = num_layers,
                                connection = connection,
                                combination = combination,
                                # input params
                                text_dim = text_dim,
                                audio_dim = audio_dim,
                                video_dim = video_dim,
                                audiofeature = audiofeature,
                                videofeature = videofeature,
                                # output params
                                out_dim = out_dim,
                                MEAN = MEAN,
                                STD = STD,
                                # embedding params
                                str2id_path = str2id_path,
                                embed_path = embed_path,
                                UNK_path = UNK_path,
                                num_trainable_words = num_trainable_words,
                                # fnn params
                                denses = denses,
                                # cnn params
                                cksizes = cksizes,
                                wksizes = wksizes,
                                fsizes = fsizes,
                                cstrides = cstrides,
                                wstrides = wstrides,
                                batch_norm = batch_norm,
                                pool_type = pool_type,
                                pool_size = pool_size,
                                # rnn params
                                rnns = rnns,
                                bidirectional = bidirectional,
                                cell_type = cell_type,
                                # activation params,
                                act = act,
                                rnnact = rnnact,
                                gateact = gateact,
                                # pool params
                                globalpool_type = globalpool_type,
                                # learning params
                                L2 = L2,
                                batchsize =batchsize,
                                dropout = dropout,
                                learning_rate = learning_rate,
                                gradient_clipping_norm = gradient_clipping_norm,
                                losses = losses)
        return model_params
    
    MEAN,STD = np.load('%s/firstimpression%s/tfrecord/big5/train/mean_std.npy' %(datapath, dataset))
    MEAN = MEAN[:-1].tolist()
    STD = STD[:-1].tolist()
    archi = 'cp1d'
    modalities = ['video']
    num_layers = 1
    connection = 'stack' # stack, dense
    combination = 'normal' # normal, parallel, residual
    # input params
    text_dim = [300]
    audio_dim = [-1,39]
    video_dim = [-1,45,80,3]
    audiofeature = 'fb'
    videofeature = '80x45'
    # output params
    out_dim = 5
    # embedding params
    str2id_path = '%s/firstimpressionV2/tfrecord/text/word2ID.txt' %(datapath)
    embed_path = '%s/firstimpressionV2/tfrecord/text/embedding_matrix.npy' %(datapath)
    UNK_path = '%s/firstimpressionV2/tfrecord/text/UNK.npy' %(datapath)
    num_trainable_words = 7
    # fnn params
    denses = 128
    # cnn params
    cksizes = 2
    wksizes = 6
    fsizes = 128
    cstrides = 1
    wstrides = 1
    batch_norm = 0

    # rnn params
    rnns = [128]
    bidirectional = 1
    cell_type = 'gru'
    # activation params
    act = 'relu'
    rnnact = 'tanh'
    gateact = 'tanh'
    # pool params
    globalpool_type = 'avg'
    pool_type = 'avg'
    pool_size = 2
    # learning params
    L2 = 0
    batchsize=16
    dropout = 0
    learning_rate = 1e-4
    gradient_clipping_norm = 5.0
    losses = ['mse']   
    
    for con in ['stack']:
        for com in ['normal']:
            for a in ['C_DC']:
                for m in [['text']]:
                    for n in [1]:
                        for gac in ['tanh']:
                            for gp in ['avg']:
                                for d in [0]:
                                    for l2 in [0]:
                                        for ck in [2]:
                                            for f in [256]:
                                                for fea in ['mfcc']:
                                                    for bc in [16]:
                                                        for rnn in [[256]]:
                                                            connection =con
                                                            combination = com
                                                            archi = a
                                                            modalities = m
                                                            num_layers = n
                                                            globalpool_type = gp
                                                            dropout = d
                                                            L2=l2
                                                            fsizes = f
                                                            cksizes = ck
                                                            batchsize = bc
                                                            rnns = rnn
                                                            if fea == 'mfcc':
                                                                audiofeature = 'mfcc'
                                                                audio_dim = [-1,39]
                                                            elif fea == 'raw':
                                                                audiofeature = 'raw'
                                                                audio_dim = [-1,1]
                                                            elif fea == 'fIS13':
                                                                audiofeature = 'framewised_IS13'
                                                                audio_dim = [-1,130]
                                                            elif fea == 'IS13':
                                                                audiofeature = 'IS13'
                                                                audio_dim = [6373]
                                                            else:
                                                                audiofeature = 'fb'
                                                                audio_dim = [-1,120]
                                                            gateact  = gac
                                                            model_params = setparam()
                                                            learning.learning(FLAGS, model_params, logpath)
    '''
    for con in ['stack']:
        for com in ['normal']:
            for a in ['C_DC']:
                for m in [['audio']]:
                    for n in [2]:
                        for gac in ['tanh']:
                            for gp in ['avg']:
                                for d in [0]:
                                    for l2 in [0]:
                                        for ck in [2]:
                                            for f in [512]:
                                                for fea in ['mfcc']:
                                                    for bc in [8]:
                                                        for rnn in [[128]]:
                                                            connection =con
                                                            combination = com
                                                            archi = a
                                                            modalities = m
                                                            num_layers = n
                                                            globalpool_type = gp
                                                            dropout = d
                                                            L2=l2
                                                            fsizes = f
                                                            cksizes = ck
                                                            batchsize = bc
                                                            rnns = rnn
                                                            if fea == 'mfcc':
                                                                audiofeature = 'mfcc'
                                                                audio_dim = [-1,39]
                                                            elif fea == 'raw':
                                                                audiofeature = 'raw'
                                                                audio_dim = [-1,1]
                                                            elif fea == 'fIS13':
                                                                audiofeature = 'framewised_IS13'
                                                                audio_dim = [-1,130]
                                                            elif fea == 'IS13':
                                                                audiofeature = 'IS13'
                                                                audio_dim = [6373]
                                                            else:
                                                                audiofeature = 'fb'
                                                                audio_dim = [-1,120]
                                                            gateact  = gac
                                                            model_params = setparam()
                                                            learning.learning(FLAGS, model_params, logpath)'''
      
if __name__ == '__main__':
    tf.app.run()
