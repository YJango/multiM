import model
import layers
import tensorflow as tf
import numpy as np
def model_fn_maker(params):
    def model_fn(features, mode, params):
        
        layer =layers.basic_layer(mode)
        # big5 labels
        MEAN = np.array(params.MEAN+[0.50382286])
        STD = np.array(params.STD+[0.15010303])
        big5 = tf.concat((features['all'][0]['big5'],tf.reshape(features['all'][0]['job'],[-1,1])),axis=-1)
        norm_big5 = (big5-MEAN)/STD
        
        # forward
        a = features['all'][1]['256']
        a0 = tf.zeros_like(a)
        a_dim = 256 #text
        
        b = features['all'][2]['512']
        b0 = tf.zeros_like(b)
        b_dim = 512 #audio
        
        c = features['all'][3]['1024']
        c0 = tf.zeros_like(c)
        c_dim = 1024 #video
        
        denses = params.denses
        def cor(a, b):
            a_mean = tf.reduce_mean(a, axis=0)
            a_centered = a - a_mean
            b_mean = tf.reduce_mean(b, axis=0)
            b_centered = b - b_mean
            corr_nr = tf.reduce_sum(a_centered * b_centered, axis=0)
            corr_dr1 = tf.reduce_sum(a_centered * a_centered, axis=0)
            corr_dr2 = tf.reduce_sum(b_centered * b_centered, axis=0)
            corr_dr = tf.sqrt(corr_dr1 * corr_dr2 + 1e-8)
            corr = corr_nr / corr_dr
            return tf.reduce_mean(corr)
        def cornet(a, b, c, reuse = tf.AUTO_REUSE):
            #attention
            # a --> ha
            a2h = layer.dense_layers(a, [1024], name='auto_a2h', reuse=reuse,trainable=params.h_trainable)
            # b --> hb
            b2h = layer.dense_layers(b, [1024], name='auto_b2h', reuse=reuse,trainable=params.h_trainable)
            # c --> hc
            c2h = layer.dense_layers(c, [1024], name='auto_c2h', reuse=reuse,trainable=params.h_trainable)

            h = a2h + b2h + c2h
            

            if params.dropout!=0:
                h = tf.layers.dropout(inputs=h, rate=params.dropout , training= mode == tf.estimator.ModeKeys.TRAIN)
                print('add dropout %s' %params.dropout)
                
            h2 = layer.dense_layers(h, params.denses, name='h2',dp=params.dropout, reuse=reuse,trainable=params.h_trainable)
            
            # h --> a
            h2a = layer.dense_layers(h, [a_dim], name='auto_h2a', act='linear', reuse=reuse, trainable=params.h_trainable)
            # h --> b
            h2b = layer.dense_layers(h, [b_dim], name='auto_h2b', act='linear', reuse=reuse, trainable=params.h_trainable)
            # h --> c
            h2c = layer.dense_layers(h, [c_dim], name='auto_h2c', act='linear', reuse=reuse, trainable=params.h_trainable)
            

            return {'h2a':h2a, 'h2b':h2b, 'h2c':h2c, 'h':h, 'h2':h2}
        def cornet2(a, b, c, reuse = tf.AUTO_REUSE):

            #attention
            # a --> ha
            a2h = layer.dense_layers(a, [1024], name='auto_a2h',trainable=params.h_trainable,reuse=reuse)
            # b --> hb
            b2h = layer.dense_layers(b, [1024], name='auto_b2h',trainable=params.h_trainable,reuse=reuse)
            # c --> hc
            c2h = layer.dense_layers(c, [1024], name='auto_c2h',trainable=params.h_trainable,reuse=reuse)
            
            ascore = tf.layers.dense(a2h, 1024,use_bias=False,trainable=params.s_trainable,name='ascore',reuse=reuse)
            bscore = tf.layers.dense(b2h, 1024,use_bias=False,trainable=params.s_trainable,name='bscore',reuse=reuse)
            cscore = tf.layers.dense(c2h, 1024,use_bias=False,trainable=params.s_trainable,name='cscore',reuse=reuse)
            
            a2h = tf.expand_dims(a2h,axis=1)
            b2h = tf.expand_dims(b2h,axis=1)
            c2h = tf.expand_dims(c2h,axis=1)
            
            # batch, 3, 1024
            abc2h = tf.concat([a2h,b2h,c2h],1)
           


            acscore = tf.expand_dims(tf.nn.tanh(ascore + cscore),axis=1)
            bcscore = tf.expand_dims(tf.nn.tanh(bscore + cscore),axis=1)
            ccscore = tf.expand_dims(tf.nn.tanh(cscore + cscore),axis=1)
            
            abcscore = tf.concat([acscore,bcscore,ccscore],1)
            attention_weights = tf.nn.softmax(abcscore, axis=1)

            

            if params.h_trainable==1:
                context_vector = (attention_weights/attention_weights) * abc2h
            else:
                context_vector = attention_weights * abc2h
            h = tf.reduce_sum(context_vector, axis=1)
            
            
            if params.dropout!=0:
                h = tf.layers.dropout(inputs=h, rate=params.dropout , training= mode == tf.estimator.ModeKeys.TRAIN)
                print('add dropout %s' %params.dropout)
                
            h2 = layer.dense_layers(h, params.denses, name='h2',dp=params.dropout, reuse=reuse,trainable=~params.h_trainable)
            
            # h --> a
            h2a = layer.dense_layers(h, [a_dim], name='auto_h2a', act='linear', reuse=reuse, trainable=params.h_trainable)
            # h --> b
            h2b = layer.dense_layers(h, [b_dim], name='auto_h2b', act='linear', reuse=reuse, trainable=params.h_trainable)
            # h --> c
            h2c = layer.dense_layers(h, [c_dim], name='auto_h2c', act='linear', reuse=reuse, trainable=params.h_trainable)
            

            return {'h2a':h2a, 'h2b':h2b, 'h2c':h2c, 'h':h, 'h2':h2,'cor_ab':cor(a2h, b2h), 'cor_bc':cor(b2h, c2h), 'cor_ac':cor(a2h, c2h)}
        
        out_a   = cornet(a , b0, c0, tf.AUTO_REUSE) # only input a
        out_b   = cornet(a0, b , c0, tf.AUTO_REUSE) # only input b
        out_c   = cornet(a0, b0, c, tf.AUTO_REUSE) # only input c
        
        out_ab  = cornet(a  ,b , c0, tf.AUTO_REUSE) # input a ,b
        out_bc  = cornet(a0 ,b , c , tf.AUTO_REUSE) # input a ,b
        out_ac  = cornet(a  ,b0, c , tf.AUTO_REUSE) # input a ,b
        
        out_abc = cornet(a  ,b , c , tf.AUTO_REUSE) # input a ,b ,c
        
        z = tf.concat(values=[a,b,c],axis = -1)
        
        # auto3
        predictions_from_abc = tf.concat(values=[out_abc['h2a'],out_abc['h2b'],out_abc['h2c']],axis = -1)
        loss_p3 = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_abc)
        
        # auto2
        predictions_from_ab = tf.concat(values=[out_ab['h2a'],out_ab['h2b'],out_ab['h2c']],axis = -1)
        predictions_from_bc = tf.concat(values=[out_bc['h2a'],out_bc['h2b'],out_bc['h2c']],axis = -1)
        predictions_from_ac = tf.concat(values=[out_ac['h2a'],out_ac['h2b'],out_ac['h2c']],axis = -1)
        
        loss_pab = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_ab)
        loss_pbc = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_bc)
        loss_pac = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_ac)
        
        loss_p2 = loss_pab + loss_pbc + loss_pac
        
        # auto1
        predictions_from_a = tf.concat(values=[out_a['h2a'],out_a['h2b'],out_a['h2c']],axis = -1)
        predictions_from_b = tf.concat(values=[out_b['h2a'],out_b['h2b'],out_b['h2c']],axis = -1)
        predictions_from_c = tf.concat(values=[out_c['h2a'],out_c['h2b'],out_c['h2c']],axis = -1)
        
        loss_pa = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_a)
        loss_pb = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_b)
        loss_pc = tf.losses.mean_squared_error(labels = z, predictions = predictions_from_c)
        
        loss_p1 = loss_pa + loss_pb + loss_pc
        
        
        # prediction using a, b, c
        norm_outputs = tf.layers.dense(inputs = out_abc['h2'], units = params.out_dim, name = 'output%s' %params.out_dim, reuse = tf.AUTO_REUSE)
        # prediction using only ab
        norm_outputs_ab = tf.layers.dense(inputs = out_ab['h2'], units = params.out_dim, name = 'output%s' %params.out_dim, reuse = tf.AUTO_REUSE)
        # prediction using only bc
        norm_outputs_bc = tf.layers.dense(inputs = out_bc['h2'], units = params.out_dim, name = 'output%s' %params.out_dim, reuse = tf.AUTO_REUSE)
        # prediction using only ac
        norm_outputs_ac = tf.layers.dense(inputs = out_ac['h2'], units = params.out_dim, name = 'output%s' %params.out_dim, reuse = tf.AUTO_REUSE)
        # prediction using only a
        norm_outputs_a = tf.layers.dense(inputs = out_a['h2'], units = params.out_dim, name = 'output%s' %params.out_dim, reuse = tf.AUTO_REUSE)
        # prediction using only b
        norm_outputs_b = tf.layers.dense(inputs = out_b['h2'], units = params.out_dim, name = 'output%s' %params.out_dim, reuse = tf.AUTO_REUSE)
        # prediction using only c
        norm_outputs_c = tf.layers.dense(inputs = out_c['h2'], units = params.out_dim, name = 'output%s' %params.out_dim, reuse = tf.AUTO_REUSE)
        
        outputs = norm_outputs*STD+MEAN
        outputs_a = norm_outputs_a*STD+MEAN
        outputs_b = norm_outputs_b*STD+MEAN
        outputs_c = norm_outputs_c*STD+MEAN
        outputs_ab = norm_outputs_ab*STD+MEAN
        outputs_bc = norm_outputs_bc*STD+MEAN
        outputs_ac = norm_outputs_ac*STD+MEAN
        
        
        predictions = {"big5": big5,
                  "outputs": outputs_bc,
                  "outputs_a": outputs_a,
                  "outputs_b": outputs_b,
                  "outputs_c": outputs_c,
                  "outputs_ab": outputs_ab,
                  "outputs_bc": outputs_bc,
                  "outputs_ac": outputs_ac,
                  "h":out_abc['h'],
                  "h2":out_abc['h2']
                   }
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
        # target losses
        mse_loss_3 = tf.losses.mean_squared_error(labels = norm_big5, predictions = norm_outputs, scope='mse_loss')
        
        mse_loss_ab = tf.losses.mean_squared_error(labels = norm_big5, predictions = norm_outputs_ab, scope='mse_loss_ab')
        mse_loss_bc = tf.losses.mean_squared_error(labels = norm_big5, predictions = norm_outputs_bc, scope='mse_loss_bc')
        mse_loss_ac = tf.losses.mean_squared_error(labels = norm_big5, predictions = norm_outputs_ac, scope='mse_loss_ac')
        mse_loss_2 = mse_loss_ab + mse_loss_bc + mse_loss_ac
        
        mse_loss_a = tf.losses.mean_squared_error(labels = norm_big5, predictions = norm_outputs_a, scope='mse_loss_a')
        mse_loss_b = tf.losses.mean_squared_error(labels = norm_big5, predictions = norm_outputs_b, scope='mse_loss_b')
        mse_loss_c = tf.losses.mean_squared_error(labels = norm_big5, predictions = norm_outputs_c, scope='mse_loss_c')
        mse_loss_1 = mse_loss_a + mse_loss_b + mse_loss_c
        #cor_loss= out_ac['cor_ac'] + out_bc['cor_bc']
        # exclude variables
        trainable_variables = tf.trainable_variables()
        print([v.name for v in trainable_variables])

        total_loss1 = (mse_loss_3+mse_loss_2+mse_loss_1)*0.2 + (loss_p3+loss_p2+loss_p1)*(1-params.lamda)#- cor_loss*params.lamda
        total_loss2 =( mse_loss_3)*(1-params.lamda)+(mse_loss_2+mse_loss_1)*params.lamda
        total_loss3 = mse_loss_bc
        
        #if total_loss==0:
            #raise TypeError("must add a loss in params.losses")
        if params.total_loss ==1:
            total_loss = total_loss1
            print('total loss 1')
        elif params.total_loss ==2:
            total_loss = total_loss2
            print('total loss 2')
        elif params.total_loss ==3:
            total_loss = total_loss3
            print('total loss 3')
        # train
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                  loss=total_loss,
                  global_step=tf.train.get_global_step(),
                  learning_rate=params.learning_rate,
                  optimizer="Adam",
                  variables = trainable_variables,
                  # some gradient clipping stabilizes training in the beginning.
                  clip_gradients=params.gradient_clipping_norm,
                  summaries=["learning_rate"])
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

        eval_metric_ops = {
                     "avg_ma": tf.metrics.mean_absolute_error(labels = big5, predictions = outputs),
            
                     "avg_ma_a": tf.metrics.mean_absolute_error(labels = big5, predictions = outputs_a),
                     "avg_ma_b": tf.metrics.mean_absolute_error(labels = big5, predictions = outputs_b),
                     "avg_ma_c": tf.metrics.mean_absolute_error(labels = big5, predictions = outputs_c),
            
                     "avg_ma_ab": tf.metrics.mean_absolute_error(labels = big5, predictions = outputs_ab),
                     "avg_ma_bc": tf.metrics.mean_absolute_error(labels = big5, predictions = outputs_bc),
                     "avg_ma_ac": tf.metrics.mean_absolute_error(labels = big5, predictions = outputs_ac),
            
                     # autoencoder losses
                     "a2z_mse": tf.metrics.mean_squared_error(labels = z, predictions = predictions_from_a),
                     "b2z_mse": tf.metrics.mean_squared_error(labels = z, predictions = predictions_from_b),
                     "c2z_mse": tf.metrics.mean_squared_error(labels = z, predictions = predictions_from_c),
            
                     "ab2z_mse": tf.metrics.mean_squared_error(labels = z, predictions = predictions_from_ab),
                     "bc2z_mse": tf.metrics.mean_squared_error(labels = z, predictions = predictions_from_bc),
                     "ac2z_mse": tf.metrics.mean_squared_error(labels = z, predictions = predictions_from_ac),
            
                     "abc2z_mse": tf.metrics.mean_squared_error(labels = z, predictions = predictions_from_abc),
                     }
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)
    return model_fn