import tensorflow as tf
import learning
import numpy as np


#tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("C", 1, "whether continue to train")
tf.app.flags.DEFINE_integer("E",3, "epoch")
tf.app.flags.DEFINE_string("M", "train", "train, predict, eval")
datapath = '/USERS/d8182103/'
logpath='/USERS/d8182103/merge/'
dataset = 'V2'

#*********************************
def main(unused_argv):
    def setparam():
        model_params = tf.contrib.training.HParams(
                                archi = archi,
                                epoch =epoch,
                                s_trainable =s_trainable,
                                total_loss = total_loss,
                                h_trainable = h_trainable,
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
                                lamda = lamda,
                                learning_rate = learning_rate,
                                gradient_clipping_norm = gradient_clipping_norm,
                                losses = losses)
        return model_params
    
    MEAN,STD = np.load('%s/firstimpression%s/tfrecord/big5/train/mean_std.npy' %(datapath, dataset))
    MEAN = MEAN[:-1].tolist()
    STD = STD[:-1].tolist()
    num_layers = 1
    connection = 'stack' # stack, dense
    combination = 'normal' # normal, parallel, residual
    # input params
    text_dim = [256]
    audio_dim = [512]
    video_dim = [1024]
    audiofeature = '512'
    videofeature = '1024'
    # output params
    out_dim = 6
    # embedding params
    str2id_path = '%s/firstimpressionV2/tfrecord/text/word2ID.txt' %(datapath)
    embed_path = '%s/firstimpressionV2/tfrecord/text/embedding_matrix.npy' %(datapath)
    UNK_path = '%s/firstimpressionV2/tfrecord/text/UNK.npy' %(datapath)
    num_trainable_words = 7
    # cnn params
    cksizes = 2
    wksizes = 6
    fsizes = 128
    cstrides = 1
    wstrides = 1
    batch_norm = 0
    # rnn params
    rnns = [128]
    bidirectional = 0
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
    learning_rate = 1e-4
    gradient_clipping_norm = 5.0 
    lamda = 0.1
    modalities = ['text','audio','video']
    losses =['att']
    
    
    '''for s in archi.split('_'):
        total_loss = int(s[0])
        h_trainable =int(s[1])
        if len(s)==3:
            s_trainable =int(s[2])'''
    dropout = 0.5

    denses = [128]
    batchsize=16
    archi = '3782_3'
    total_loss = 1
    h_trainable = 1
    s_trainable = 1
    epoch = 3
    
    model_params = setparam()
    learning.learning(FLAGS, model_params, logpath)
    total_loss = 2
    h_trainable = 0
    s_trainable = 1
    epoch = 3
    
    model_params = setparam()
    learning.learning(FLAGS, model_params, logpath)
    total_loss = 3
    h_trainable = 0
    s_trainable = 0
    epoch = 1
    
    model_params = setparam()
    learning.learning(FLAGS, model_params, logpath)

    
if __name__ == '__main__':
    tf.app.run()
