import numpy as np
import tensorflow as tf
import layers

class model(object):
    def __init__(self, mode, params):
        # archi
        self.archi = params.archi
        self.modalities = params.modalities
        self.num_layers = params.num_layers
        self.connection = params.connection
        self.combination = params.combination
        # input params
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.video_dim = params.video_dim
        self.audiofeature = params.audiofeature
        self.videofeature = params.videofeature
        # output params
        self.out_dim = params.out_dim
        self.MEAN = params.MEAN
        self.STD = params.STD
        # embedding params
        self.str2id_path = params.str2id_path
        self.embed_path = params.embed_path
        self.UNK_path = params.UNK_path
        self.num_trainable_words = params.num_trainable_words
        # fnn params
        self.denses = params.denses
        # cnn params
        self.cksizes = params.cksizes
        self.wksizes = params.wksizes
        self.fsizes = params.fsizes
        self.cstrides = params.cstrides
        self.wstrides = params.wstrides
        self.batch_norm = params.batch_norm
        self.pool_type = params.pool_type
        self.pool_size = params.pool_size
        # rnn params
        self.rnns = params.rnns
        self.bidirectional = params.bidirectional
        self.cell_type = params.cell_type
        # activation params
        self.act = params.act
        self.rnnact = params.rnnact
        self.gateact = params.gateact
        # pool params
        self.globalpool_type = params.globalpool_type
        # learning params
        self.L2 = params.L2
        self.batchsize = params.batchsize
        self.dropout = params.dropout
        self.learning_rate = params.learning_rate
        self.gradient_clipping_norm = params.gradient_clipping_norm
        self.mode = mode
        self.basic_layer = layers.basic_layer(self.mode)
    
    def dense_layers(self,x, input_size=None, name='dense'):
        return self.basic_layer.dense_layers(x, sizes=self.denses, name=name, L2 = self.L2, dp = self.dropout,
              act=self.act, trainable=True, 
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
    
    def embedding_layer(self,x, input_size=None, name='embedding'):
        return self.basic_layer.embedding_layer(x, str2id_path=self.str2id_path, dim =self.text_dim[-1],
               embed_path=self.embed_path, UNK_path=self.UNK_path,
               name = name, trainable=False, num_trainable_words=self.num_trainable_words)
    
    def rnn_layers(self,x,input_size=None, name='rnn'):
        return self.basic_layer.rnn_layers(x, sizes=self.rnns, name=name, ctype=self.cell_type, act=self.rnnact, dp = self.dropout)
    
    def birnn_layers(self,x,input_size=None, name='birnn'):
        return self.basic_layer.birnn_layers(x, sizes=self.rnns, name=name, ctype=self.cell_type, act=self.rnnact, dp = self.dropout)
    
    
    def conv1d_layer(self,x, input_size=None, ksize=None, fsize=None, stride = None, act=None, padding='same',name='cnn1d'):
        if ksize is None:
            ksize = self.cksizes
        if fsize is None:
            fsize=self.fsizes
        if stride is None:
            stride=self.cstrides
        if act is None:
            act=self.act
        return self.basic_layer.conv1d_layer(x, ksize=ksize, fsize=fsize, name=name, bn=self.batch_norm, padding=padding, 
              strides=stride, L2 = self.L2, dp = self.dropout,
              act=act, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
    
    def atrous_conv1d_layer(self,x, fsize=None, input_size=None, ksize=None, act=None, rate=None,padding='same', name='atrous_conv1d'):
        # need to expand 3D tank tensor to [batch, timestep, 1, channel]
        if fsize is None:
            fsize = self.fsizes
        if act is None:
            act=self.act
        if ksize is None:
            ksize=self.cksizes
        if rate is None:
            myrate = self.rate
        output = self.basic_layer.atrous_conv1d_layer(x, ksize=ksize, fsize=fsize, 
              name=name, bn=self.batch_norm, 
              padding=padding, rate=myrate, L2 = self.L2, dp = self.dropout,
              act=act, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
        if rate is None:
            self.rate = self.rate*2
        return output
    
    def downsampling1d(self,x, input_size=None,name='downsample1d'):
        return self.basic_layer.downsampling1d(x, name=name, ptype=self.pool_type, psize=self.pool_size, strides=self.pool_size)
    
    
    def conv2d_layer(self,x, input_size=None,name='conv2d'):
        return self.basic_layer.conv2d_layer(x, ksize=self.cksizes, fsize=self.fsizes, name=name, bn=self.batch_norm, padding='same', 
              strides=self.cstrides, L2 = self.L2, dp = self.dropout,
              act=self.act, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
    
    
    def downsampling2d(self,x,input_size=None, name='downsample2d'):
        return self.basic_layer.downsampling2d(x, name=name, ptype=self.pool_type, psize=self.pool_size, strides=self.pool_size)
    
    
    def globalpool(self,x, input_size=None,name='globalpool'):
        return self.basic_layer.globalpool(x, name=name, ptype=self.globalpool_type, axis = 1)
        
    
    def cnn1d_weighted_cnn1d_layer(self,x,input_size=None, name='cwc1d'):
        return self.basic_layer.cnn1d_weighted_cnn1d_layer(x, cksize=self.cksizes, wksize=self.wksizes, 
              fsize=self.fsizes, name=name, bn=False, cstrides=self.cstrides, 
              wstrides=self.wstrides, padding='same', L2 = self.L2, dp = self.dropout,
              act=self.act, gateact=self.gateact, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
    
    def cnn1d_inweighted_cnn1d_layer(self,x,input_size=None, name='ciwc1d'):
        return self.basic_layer.cnn1d_inweighted_cnn1d_layer(x, cksize=self.cksizes, wksize=self.wksizes, 
              fsize=self.fsizes, name=name, bn=False, cstrides=self.cstrides, 
              wstrides=self.wstrides, padding='same', L2 = self.L2, dp = self.dropout,
              act=self.act, gateact=self.gateact, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
    
    def cnn2d_weighted_cnn2d_layer(self,x, input_size=None,name='cwc2d'):
        return self.basic_layer.cnn2d_weighted_cnn2d_layer(x, cksize=self.cksizes, wksize=self.wksizes, 
              fsize=self.fsizes, name=name, bn=False, cstrides=self.cstrides, 
              wstrides=self.wstrides, padding='same', L2 = self.L2, dp = self.dropout,
              act=self.act, gateact=self.gateact, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
    
    
    def rnn_weighted_cnn1d_layer(self,x, input_size=None,name='rwc1d'):
        return self.basic_layer.rnn_weighted_cnn1d_layer(x, ksize = self.cksizes, fsize = self.fsizes, 
              name=name, bn=self.batch_norm, strides=self.cstrides, ctype = self.cell_type, 
              bidirectional = self.bidirectional, padding='same',  L2 = self.L2, dp = self.dropout,
              act=self.act, gateact=self.gateact, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
        
    
    def rnn_indweighted_cnn1d_layer(self,x, input_size=None, name='riwc1d'):
        return self.basic_layer.rnn_indweighted_cnn1d_layer(x, ksize = self.cksizes, fsize = self.fsizes, 
              name=name, bn=self.batch_norm, strides=self.cstrides, ctype = self.cell_type, 
              bidirectional = self.bidirectional, padding='same',  L2 = self.L2, dp = self.dropout,
              act=self.act, gateact=self.gateact, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
        
    
    def rnn_weighted_cnn2d_layer(self,x, input_size, name='riwc2d'):
        return self.basic_layer.rnn_weighted_cnn2d_layer(x, ksize = self.cksizes, fsize = self.fsizes, 
              input_size=input_size, name=name, bn=self.batch_norm, strides=self.cstrides, ctype = self.cell_type,
              bidirectional = self.bidirectional,padding='same', L2 = self.L2, dp = self.dropout,
              act=self.act, gateact=self.gateact, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
        
    
    def rnn_indweighted_cnn2d_layer(self,x, input_size, name='riwc2d'):
        return self.basic_layer.rnn_indweighted_cnn2d_layer(x, ksize = self.cksizes, fsize = self.fsizes, 
              input_size=input_size, name=name, bn=self.batch_norm, strides=self.cstrides, ctype = self.cell_type,
              bidirectional = self.bidirectional,padding='same', L2 = self.L2, dp = self.dropout,
              act=self.act, gateact=self.gateact, trainable=True, reuse = None,
              W_init=tf.contrib.layers.xavier_initializer(),
              b_init = tf.constant_initializer(0.0001, tf.float32))
    
    # combination layers
    def cnn1d_pool_layer(self,x, input_size, name='cp'):
        print('cnn1d_pool_layer')
        x = self.conv1d_layer(x,input_size, name+'cp')
        x = self.downsampling1d(x,input_size, name+'cp')
        return x
    def cnn2d_pool_layer(self,x, input_size, name='cp'):
        print('cnn2d_pool_layer')
        x = self.conv2d_layer(x,input_size, name+'cp')
        x = self.downsampling2d(x,input_size, name+'cp')
        return x
    
    def cnn1d_pool_rnn_layers(self,x, input_size, name='cpr'):
        for i in range(self.num_layers):
            x = self.cnn1d_pool_layer(x,input_size, name+'cpr_%s' %i)
            print('cnn1d_pool_rnn_layer')
            
        if self.bidirectional ==0:
            print('rnn_cnn1d_pool_layers')
            x = self.rnn_layers(x,input_size, name+'cpr')
        else:
            print('birnn_cnn1d_pool_layers')
            x = self.birnn_layers(x,input_size, name+'cpr')
        return x
    
    def rnn_cnn1d_pool_layers(self,x, input_size, name='cpr'):
        if self.bidirectional ==0:
            print('rnn_cnn1d_pool_layers')
            x = self.rnn_layers(x,input_size, name+'rcp')
        else:
            print('birnn_cnn1d_pool_layer')
            x = self.birnn_layers(x,input_size, name+'rcp')
        for i in range(self.num_layers):
            print('cnn1d_pool_rnn_layer')
            x = self.cnn1d_pool_layer(x,input_size, name+'rcp_%s' %i)
        return x
    
    def cnn1d_rnn_layers(self,x, input_size, name='cr'):
        for i in range(self.num_layers):
            x = self.conv1d_layer(x,input_size, name+'cr_%s' %i)
            print('cnn1d_rnn_layer')
            
        if self.bidirectional ==0:
            print('rnn_cnn1d_layers')
            x = self.rnn_layers(x,input_size, name+'cr')
        else:
            print('birnn_cnn1d_layers')
            x = self.birnn_layers(x,input_size, name+'cr')
        return x
    
    
    def rnn_cnn1d_layers(self,x, input_size, name='cr'):
        if self.bidirectional ==0:
            print('rnn_cnn1d_layers')
            x = self.rnn_layers(x,input_size, name+'rc')
        else:
            print('birnn_cnn1d_layer')
            x = self.birnn_layers(x,input_size, name+'rc')
        for i in range(self.num_layers):
            print('cnn1d_rnn_layer')
            x = self.conv1d_layer(x,input_size, name+'rc_%s' %i)
        return x
    
    
    def multi_layers(self, x, list_layer_fn, modality, input_size=None):
        for i in range(len(list_layer_fn)):
            x=list_layer_fn[i](x, input_size ,name=modality+'m_%s' %i)
        return x
    
    
    def Normal(self, x, layer_fn, modality, input_size=None):
        x = layer_fn(x , input_size, name=modality)
        return x
    
    def Parallel(self, x, list_layer_fn, modality, input_size=None):
        s = []
        for i in range(len(list_layer_fn)):
            s.append(list_layer_fn[i](x, input_size ,name=modality+'P_%s' %i))
        return tf.concat(s)
    
    def Residual(self, x, layer_fn, modality, input_size=None):
        y1 = layer_fn(x , input_size, name=modality+'R_y1')
        y2 = layer_fn(y1, input_size ,name=modality+'R_y2')
        return x + y2
        
    
    def choose(self, x, modality, input_size=None):
        self.rate = 1
        # set dim
        if 'text' == modality:
            self.dim = self.text_dim
        elif 'audio' == modality:
            self.dim = self.audio_dim
        else:
            self.dim = self.video_dim
            
        # combination
        if self.combination == 'normal':
            COM = self.Normal
        elif self.combination == 'parallel':
            COM = self.Parallel
        elif self.combination == 'residual':
            COM = self.Residual
        # layer
        if self.archi == 'dense':
            L = self.dense_layers
            return COM(x, L, modality)
        elif self.archi == 'conv1d':
            L = self.conv1d_layer
            
        elif self.archi == 'CAC_DS':
            s = 0
            # conv1d k1
            l = self.conv1d_layer(x, ksize =1, fsize = self.denses, name='l0')
            c = self.conv1d_layer(l, ksize =1, fsize = self.denses, name='c0_conv')
            c = self.globalpool(c, name='globalpool')
            s+=c
            
            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.atrous_conv1d_layer(l, name = ('l%s_aconv' %i), fsize = self.fsizes)
                c = self.conv1d_layer(l, ksize =1, fsize = self.denses, name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                s+=c
            return s
        
        elif self.archi == 'CC_DS':
            s = 0
            # conv1d k1
            l = self.conv1d_layer(x, ksize =1, fsize = self.denses, name='l0')
            c = self.conv1d_layer(l, ksize =1, fsize = self.denses, name='c0_conv')
            c = self.globalpool(c, name='globalpool')
            s+=c
            
            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.conv1d_layer(l,name=('l%s_aconv' %i))
                c = self.conv1d_layer(l, ksize =1, fsize = self.denses, name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                s+=c
            return s
        
        elif self.archi == 'AC_DS':
            s = 0
            l = x
            
            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.atrous_conv1d_layer(l, name = ('l%s_aconv' %i), fsize = self.fsizes)
                c = self.conv1d_layer(l, ksize =1, fsize = self.denses, name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                s+=c
            return s
        
        elif self.archi == 'C_DS':
            s = 0
            l = x
           
            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.conv1d_layer(l,name=('l%s_aconv' %i))
                c = self.conv1d_layer(l, ksize =1, fsize = self.denses, name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                s+=c
            return s
        
        elif self.archi == 'CAC_DC':
            lists = []
            # conv1d k1
            l = self.conv1d_layer(x, ksize =1, fsize = self.denses, name='l0')
            c = self.conv1d_layer(l, ksize =1, fsize = self.denses, name='c0_conv')
            c = self.globalpool(c, name='globalpool')
            lists.append(c)
            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.atrous_conv1d_layer(l, name = ('l%s_aconv' %i), fsize = self.fsizes)
                c = self.conv1d_layer(l, ksize =1, name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                lists.append(c)
            
            x = tf.concat(values=lists, axis=-1)

            return x
        elif self.archi == 'CC_DC':
            lists = []
            # conv1d k1
            l = self.conv1d_layer(x, ksize =1, name='l0')
            c = self.conv1d_layer(l, ksize =1, name='c0_conv')
            c = self.globalpool(c, name='globalpool')
            lists.append(c)

            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.conv1d_layer(l,name=('l%s_aconv' %i))
                c = self.conv1d_layer(l, ksize =1,name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                lists.append(c)
            
            x = tf.concat(values=lists, axis=-1)

            return x
        
        elif self.archi == 'AC_DC':
            lists = []

            l = x
            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.atrous_conv1d_layer(l, name = ('l%s_aconv' %i), fsize = self.fsizes)
                c = self.conv1d_layer(l, ksize =1, name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                lists.append(c)
            
            x = tf.concat(values=lists, axis=-1)

            return x
        elif self.archi == 'C_DC':
            lists = []

            l = x

            # aconv1d k2 rate 1
            for i in range(self.num_layers):
                l = self.conv1d_layer(l,name=('l%s_aconv' %i))
                c = self.conv1d_layer(l, ksize =1,name=('c%s_aconv' %i))
                c = self.globalpool(c, input_size=None, name='globalpool_%s' %i)
                lists.append(c)
            
            x = tf.concat(values=lists, axis=-1)

            return x
        
        elif self.archi == 'WaveNet':
            
            input_fsize = self.fsizes
            
            def res_block(tensor, rate, i):
                
                ot = self.atrous_conv1d_layer(tensor, name = 'l%s%s_aconv_tanh' %(i,rate),
                                   fsize = self.fsizes,act='relu',
                                   input_size=None, rate=rate, padding = 'same')
                os = self.atrous_conv1d_layer(tensor, name = 'l%s%s_aconv_sigmoid' %(i,rate),
                                   fsize = self.fsizes,act='tanh',
                                   input_size=None, rate=rate, padding = 'same')
                out = ot*os
                
                out = self.conv1d_layer(out, ksize =1,  fsize = self.fsizes, name='aconv_out%s%s' %(i,rate))
                return out+tensor, out

            
            # conv1d k1
            z = self.conv1d_layer(x, ksize =1, fsize = self.fsizes, name='front')
            
            skip = 0
            
            for i in range(self.num_layers):
                for r in [1,2,4,8]:
                    z,s= res_block(tensor=z, rate=r, i=i)
                    skip += self.globalpool(s, input_size=None, name='globalpool')
                    
            #skip = self.conv1d_layer(skip, ksize =1,  fsize = self.fsizes, name='fout')
            #skip = self.globalpool(skip, input_size=None, name='globalpool')
            return skip

        elif self.archi == 'aconv1d':
            L = self.atrous_conv1d_layers
            
            if self.connection == 'stack':
                print('connection: stack')
                x = COM(x, L, modality, input_size)

            elif self.connection == 'dense':
                print('connection: dense')
                s = x
                for i in range(self.num_layers):
                    x = COM(s, L, modality+'_%s' %i, input_size)
                    s+=x
            x = self.globalpool(x,self.modalities)
            return x
        
        elif self.archi == 'rnn':
            L = self.rnn_layers
        elif self.archi == 'birnn':
            L = self.birnn_layers
        elif self.archi == 'cwc1d':
            L = self.cnn1d_weighted_cnn1d_layer
        elif self.archi == 'ciwc1d':
            L = self.cnn1d_inweighted_cnn1d_layer
        elif self.archi == 'rwc1d':
            L = self.rnn_weighted_cnn1d_layer
        elif self.archi == 'cp1drnn':
            L = self.cnn1d_pool_layer
            if self.connection == 'stack':
                print('connection: stack')
                for i in range(self.num_layers):
                    x = COM(x, L, modality+'_%s' %i, input_size)

            elif self.connection == 'dense':
                print('connection: dense')
                s = x
                for i in range(self.num_layers):
                    x = COM(s, L, modality+'_%s' %i, input_size)
                    s+=x
            L = self.birnn_layers
            x = x = COM(x, L, modality+'_%s' %i, input_size)
            return x
        elif self.archi == 'cwcp1drnn':
            L = self.cnn1d_weighted_cnn1d_layer
            if self.connection == 'stack':
                print('connection: stack')
                for i in range(self.num_layers):
                    x = COM(x, L, modality+'_%s' %i, input_size)
                    x = self.downsampling1d(x,input_size, modality+'_%s' %i)

            elif self.connection == 'dense':
                print('connection: dense')
                s = x
                for i in range(self.num_layers):
                    x = COM(s, L, modality+'_%s' %i, input_size)
                    x = self.downsampling1d(x,input_size, modality+'_%s' %i)
                    s+=x
            L = self.birnn_layers
            x = x = COM(x, L, modality+'_%s' %i, input_size)
            return x
        elif self.archi == 'riwc1d':
            L = self.rnn_indweighted_cnn1d_layer
        elif self.archi == 'rwc2d':
            L = self.rnn_weighted_cnn1d_layer
        elif self.archi == 'riwc2d':
            L = self.rnn_indweighted_cnn1d_layer
        elif self.archi == 'cp1d':
            L = self.cnn1d_pool_layer
        elif self.archi == 'cp2d':
            L = self.cnn2d_pool_layer
        elif self.archi == 'cpr1d':
            L = self.cnn1d_pool_rnn_layers
        elif self.archi == 'rcp1d':
            L = self.rnn_cnn1d_pool_layers
        elif self.archi == 'cr1d':
            L = self.cnn1d_rnn_layers
        elif self.archi == 'rc1d':
            L = self.rnn_cnn1d_layers
        
        if self.connection == 'stack':
            print('connection: stack')
            for i in range(self.num_layers):
                x = COM(x, L, modality+'_%s' %i, input_size)
            
        elif self.connection == 'dense':
            print('connection: dense')
            x = COM(x, L, modality+'_0', input_size)
            s = x
            for i in range(1,self.num_layers):
                x = COM(s, L, modality+'_%s' %i, input_size)
                s+=x
            x=s
        x = self.globalpool(x,self.modalities)
        return x
            
            
            
            
            
            
            
            
            
