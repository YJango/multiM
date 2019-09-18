import os
import cormodality as modality
import tensorflow as tf
import data
import numpy as np
import subprocess
import json
import sys
def learning(FLAGS, model_params, logpath='/USERS/d8182103/V2/'):
    def print_hparams(hparams):
        """ print hparams to stdout """
        values = hparams.values()

        keys = values.keys()
        keys = sorted(keys)

        print("=================================================")
        for key in keys:
            print("{} : {}".format(key, values[key]))
        print("=================================================")

    def save_hparams(filename, hparams):
        """ save to json file """
        if not os.path.exists(os.path.dirname(filename)):
            raise ValueError('There is no directory to save {}'.format(filename))

        with open(filename, 'w') as f:
            json.dump(hparams.values(), f, sort_keys=True, separators=(',', ': '), indent=4)

    def load_hparams(filename, hparams):
        """ load hparams from json file """
        with open(filename, 'r') as f:
            hparams.parse_json(f.read())
        return hparams
    def clean_folder(path_name):
        # copy
        '''
        command = "mkdir -p %s_log/eval %s_log/train" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/eval/* %s_log/eval/" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/train %s_log/train" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/train.txt %s_log/train.txt" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/vali.txt %s_log/vali.txt" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/model* %s_log/" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/checkpoint %s_log/checkpoint" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/graph.pbtxt %s_log/graph.pbtxt" %(path_name,path_name)
        re = subprocess.call(command, shell=True)
        command = "cp -r %s/HParams.json %s_log/HParams.json" %(path_name,path_name)
        re = subprocess.call(command, shell=True)'''
        # delete
        command = "rm -rf %s/events*" %path_name
        re = subprocess.call(command, shell=True)
        if re ==1:
            print('failed deleting %s' %path_name)
        else:
            print('%s delete' %path_name)
    def writer(path,setname,epoch,logs):
        f = open('%s.txt' %(path+setname),'a')
        f.write('epoch: %s\n' %epoch)
        for k in logs.keys():
            f.write('%s:%s\n' %(k,logs[k]))
        f.write('\n')
        f.close()
        
    def all_acc_and_F1(predicts,my_model_dir):
        np.set_printoptions(precision=2)
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import mean_absolute_error
        def bi(a):
            if a>0:
                return 1
            else:
                return 0
        traits = ['ope', 'con', 'ext', 'agr', 'neu','job']
        # scores
        sbig5s = np.array([p['big5'] for p in predicts])
        spredicts = np.array([p['outputs'] for p in predicts])
        big5s = np.array([list(map(bi,(p['big5']-(model_params.MEAN+[0.50382286]))/(model_params.STD+[0.15010303]))) for p in predicts])
        predicts = np.array([list(map(bi,(p['outputs']-(model_params.MEAN+[0.50382286]))/(model_params.STD+[0.15010303]))) for p in predicts])

        num_example = len(big5s)
        print(traits)
        def re(a):
            if a<0:
                return 1-a
            else:
                return a
        #print(list(map(re,sum(big5s)/num_example)))
        maes = 0
        def acc_and_F1(big5,predict):
            F1 = f1_score(big5,predict, average='macro') 
            acc = accuracy_score(big5,predict)
            return acc, F1
        #def norm_mean_absolute_error(big5,predict):
            
        fw = open(my_model_dir+'/test.txt','w')
        for i,t in enumerate(traits):
            acc,F1 = acc_and_F1(big5s[:,i],predicts[:,i])
            mae = 1-mean_absolute_error(sbig5s[:,i],spredicts[:,i])
            print('%s:  F1:%6.6s | acc:%6.6s | mae:%6.6s' %(t,F1,acc,mae))
            fw.write('%s:  F1:%6.6s | acc:%6.6s | mae:%6.6s\n' %(t,F1,acc,mae))
        maes = 1-mean_absolute_error(sbig5s,spredicts)
        print('MAEs:%6.6s' %(maes))
        fw.write('MAEs:%6.6s\n' %(maes))
        fw.write('\n')
        fw.close()
        
    # folder name maker
    path_name_list = (''.join([r for r in model_params.modalities]),model_params.archi, ''.join([str(r) for r in model_params.denses]),
                 model_params.L2,model_params.dropout,
                 model_params.learning_rate,''.join(model_params.losses))#model_params.batchsize
    path_name = logpath+"/%s_%s_d%s_L2%s_dp%s_lr%s_b16_l%s" %path_name_list

    # create folder
    if os.path.isdir(path_name)&~FLAGS.C:
        print('%s is being trained by other machine' %path_name)
        clean_folder(path_name)
    else:
        print('starts to train %s' %path_name)

        command = "mkdir %s" %path_name
        re = subprocess.call(command, shell=True)
        if re ==1:
            print('%s exists' %command)
        else:
            print('created %s' %command)
        # create model_fn
        my_model_fn = modality.model_fn_maker(model_params)
        # input input_fn
        train_input_fn = data.input_fn_maker(dataset = 'V2', modalities = model_params.modalities, setname='train',
                           audioshape = model_params.audio_dim, videoshape=model_params.video_dim, 
                           audiofeature = model_params.audiofeature, videofeature=model_params.videofeature, 
                           batchsize =model_params.batchsize, shuffle_buffer = 7992, num_parallel_calls=4, prefetch_buffer=2)
        vali_input_fn = data.input_fn_maker(dataset = 'V2', modalities = model_params.modalities, setname='vali',
                           audioshape = model_params.audio_dim, videoshape=model_params.video_dim,
                           audiofeature = model_params.audiofeature, videofeature=model_params.videofeature, 
                           batchsize =model_params.batchsize, shuffle_buffer = 0, num_parallel_calls=4, prefetch_buffer=2)
        test_input_fn = data.input_fn_maker(dataset = 'V2', modalities = model_params.modalities, setname='test',
                           audioshape = model_params.audio_dim, videoshape=model_params.video_dim,
                           audiofeature = model_params.audiofeature, videofeature=model_params.videofeature, 
                           batchsize =model_params.batchsize, shuffle_buffer = 0, num_parallel_calls=4, prefetch_buffer=2)
        train_eval_fn = data.input_fn_maker(dataset = 'V2', modalities = model_params.modalities, setname='train',
                           audioshape = model_params.audio_dim, videoshape=model_params.video_dim,
                           audiofeature = model_params.audiofeature, videofeature=model_params.videofeature, 
                           batchsize =model_params.batchsize, shuffle_buffer = 0, num_parallel_calls=4, prefetch_buffer=2)
        
        myconfig = tf.estimator.RunConfig(
                        model_dir = path_name,
                        save_summary_steps=int(7992/model_params.batchsize),
                        save_checkpoints_steps=int(7992/model_params.batchsize),
                        save_checkpoints_secs=None,
                        session_config=None,
                        keep_checkpoint_max=0,
                        keep_checkpoint_every_n_hours=100000,
                        log_step_count_steps=int(7992/model_params.batchsize))
        # create estimator
        APR_regressor = tf.estimator.Estimator(
            model_fn = my_model_fn,
            config = myconfig,
            params=model_params)
        
        
        # train the model
        if FLAGS.M =='train':
            for i in range(model_params.epoch):
                print_hparams(model_params)
                print('start training\n')
                print(path_name)
                APR_regressor.train(input_fn=train_input_fn)
                print('finished training\n')
                eval_results = APR_regressor.evaluate(input_fn=test_input_fn,name='test')
                #predicts =list(APR_regressor.predict(input_fn=test_input_fn))
                #all_acc_and_F1(predicts,path_name) 
            save_hparams(path_name+'/HParams.json', model_params)
            #print('start predict test\n')
            #predicts =list(APR_regressor.predict(input_fn=test_input_fn))
            #all_acc_and_F1(predicts,path_name) 
            # delete
            clean_folder(path_name)
                
        elif FLAGS.M =='train_eval':
            print_hparams(model_params)
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=int(FLAGS.E*7992/model_params.batchsize))
            eval_spec = tf.estimator.EvalSpec(input_fn=vali_input_fn)
            tf.estimator.train_and_evaluate(APR_regressor, train_spec, eval_spec)
            save_hparams(path_name+'/HParams.json', model_params)
            print_hparams(model_params)
            
            # delete
            clean_folder(path_name)
        elif FLAGS.M =='predict':
            print_hparams(model_params)
            print('start predict test\n')
            predicts =list(APR_regressor.predict(input_fn=test_input_fn))
            #np.save('pppp',predicts)
            #for p in predicts:
            #    print(p)
            all_acc_and_F1(predicts,path_name) 
        elif FLAGS.M =='generate':
            print_hparams(model_params)
            print('start predict test\n')
            predicts =list(APR_regressor.predict(input_fn=vali_input_fn))
            np.save(path_name+'/vali.npy',predicts)
            predicts =list(APR_regressor.predict(input_fn=train_eval_fn))
            np.save(path_name+'/train.npy',predicts)
            predicts =list(APR_regressor.predict(input_fn=test_input_fn))
            np.save(path_name+'/test.npy',predicts)
            
            #for p in predicts:
                #print(p)
            all_acc_and_F1(predicts,path_name) 
        elif FLAGS.M =='eval':
            print_hparams(model_params)
            print('start eval test\n')
            eval_results = APR_regressor.evaluate(input_fn=test_input_fn,name='test')
            writer(path_name,'/testrmse',0, eval_results)
