import os,sys
import subprocess
def get_filenames(path,shuffle=False,extension='.mp4'):
    # get all file names 
    files= os.listdir(path) 
    filepaths = [path+file for file in files if not os.path.isdir(file) and extension in file]
    # shuffle
    if shuffle:
        ri = np.random.permutation(len(filepaths))
        filepaths = np.array(filepaths)[ri]
    #print(filepaths)
    return filepaths
    
def v2a(videopath):
    foldpath = videopath.split('.mp4')[0]
    command = "ffmpeg -i %s.mp4 -ac 1 -ar 16000 -vn %s.wav" %(foldpath,foldpath)
    re = subprocess.call(command, shell=True)
    if re ==1:
        raise TypeError(foldpath)
    else:
        print(foldpath)

def deleteimages(videopath):
    pathparts = videopath.split('/')
    pathparts[-1] = pathparts[-1].split('.mp4')[0]
    imagepath = '/'.join(pathparts)
    c = 'rm -rf %s/' %imagepath
    print(c)
    os.system(c)
def Framewised_IS13(videopath, targetpath='audio_Framewised_IS13'):
    toolpath = '/DB/Tools/opensmile-2.3.0/bin/linux_x64_standalone_libstdc6/SMILExtract'
    configpath = '/DB/Tools/opensmile-2.3.0/config/IS13_ComParE.conf'
    pathparts = videopath.split('/')
    pathparts[-1] = pathparts[-1].split('.mp4')[0]
    audiopath = videopath.split('.mp4')[0]+'.wav'
    csvpath = '/'.join(pathparts[:-3]+[targetpath]+pathparts[-3:])+'.txt'
    command = '%s -C %s -I %s -D %s' %(toolpath,configpath,audiopath,csvpath)
    re = subprocess.call(command, shell=True)
    if re ==1:
        raise TypeError(command)
    else:
        print(command)
def Framewised_IS10(videopath, targetpath='audio_Framewised_IS10'):
    toolpath = '/DB/Tools/opensmile-2.3.0/bin/linux_x64_standalone_libstdc6/SMILExtract'
    configpath = '/DB/Tools/opensmile-2.3.0/config/IS10_paraling.conf'
    pathparts = videopath.split('/')
    pathparts[-1] = pathparts[-1].split('.mp4')[0]
    audiopath = videopath.split('.mp4')[0]+'.wav'
    csvpath = '/'.join(pathparts[:-3]+[targetpath]+pathparts[-3:])+'.txt'
    command = '%s -C %s -I %s -D %s' %(toolpath,configpath,audiopath,csvpath)
    re = subprocess.call(command, shell=True)
    if re ==1:
        raise TypeError(command)
    else:
        print(command)
        
    #opensmiler(f, outfold=PATH+'extraction/test/IS13lld/',config='IS13_ComParE',lld='D')
    #opensmiler(f, outfold=PATH+'extraction/test/IS10lld/',config='IS10_paraling',lld='D')
    
def opensmiler(videopath, targetpath='audio_Framewised_IS13',config='IS13_ComParE',toolfold='/DB/Tools/opensmile-2.3.0/',extension='.txt',lld='D'):
    '''
    infile: single input file to be extracted
    outfold: where to save the extracted file with the same name
    extension: ".txt" or ".csv" 
    
    '''
    pathparts = videopath.split('/')
    pathparts[-1] = pathparts[-1].split('.mp4')[0]
    audiopath = videopath.split('.mp4')[0]+'.wav'
    csvpath = '/'.join(pathparts[:-3]+[targetpath]+pathparts[-3:])+'.txt'
    
    # tool and config
    tool = '%sbin/linux_x64_standalone_libstdc6/SMILExtract' %toolfold
    config = '%sconfig/%s.conf' %(toolfold,config)
    
    # get infile and outfile names
    infilename = infile
    outfilename = '%s/%s%s' %(outfold, infile.split('/')[-1].split('.wav')[0], extension)
    cmd = '%s -C %s -I %s -%s %s' %(tool,config,audiopath,lld,csvpath)
    
    #execute
    if subprocess.call(cmd, shell=True) ==1:
        raise TypeError('something wrong happened')
    else:
        print('Done:',cmd)
        
def v2images(videopath, fps=10, size='320x180',targetpath='image_320x180'):
    pathparts = videopath.split('/')
    pathparts[-1] = pathparts[-1].split('.mp4')[0]
    imagepath = '/'.join(pathparts[:-3]+[targetpath]+pathparts[-3:]+[''])
    c = 'mkdir %s' %imagepath
    os.system(c)
    command = 'ffmpeg -i %s -r %s -q:v 2 -f image2 -s %s %s%s' %(videopath,fps,size,imagepath,size) + '-%01d.jpeg'
    re = subprocess.call(command, shell=True)
    if re ==1:
        raise TypeError(command)
    else:
        print(command)
def make_new_folder(path):
    c = 'mkdir %s' %path
    print(c)
    os.system(c)
    c = 'mkdir %s/train %s/test %s/vali' %(path,path,path)
    print(c)
    os.system(c)
    for i in range(1,76):
        index = str(i).zfill(2)
        c = 'mkdir %s/train/training80_%s' %(path,index)
        print(c)
        os.system(c)
    for i in range(1,26):
        index = str(i).zfill(2)
        c = 'mkdir %s/test/test80_%s' %(path,index)
        print(c)
        os.system(c)
    for i in range(1,26):
        index = str(i).zfill(2)
        c = 'mkdir %s/vali/validation80_%s' %(path,index)
        print(c)
        os.system(c)
#make_new_folder('/USERS/d8182103/firstimpressionV1/image_320x180')
#make_new_folder('/USERS/d8182103/firstimpressionV2/image_320x180')
#make_new_folder('/USERS/d8182103/firstimpressionV1/audio_Framewised_IS13')
#make_new_folder('/USERS/d8182103/firstimpressionV2/audio_Framewised_IS13')
make_new_folder('/USERS/d8182103/firstimpressionV1/audio_Framewised_IS10')
make_new_folder('/USERS/d8182103/firstimpressionV2/audio_Framewised_IS10')

f1 = get_filenames(path='/USERS/d8182103/firstimpressionV2/test/',extension='test80')
f2 = get_filenames(path='/USERS/d8182103/firstimpressionV2/train/',extension='training80')
f3 = get_filenames(path='/USERS/d8182103/firstimpressionV2/vali/',extension='validation80')
f4 = get_filenames(path='/USERS/d8182103/firstimpressionV1/test/',extension='test80')
f5 = get_filenames(path='/USERS/d8182103/firstimpressionV1/train/',extension='training80')
f6 = get_filenames(path='/USERS/d8182103/firstimpressionV1/vali/',extension='validation80')
allfolds =[f1,f2,f3,f4,f5,f6]
for i in range(6):
    for fold in allfolds[i]:
        for videopath in get_filenames(fold+'/'):
            Framewised_IS13(videopath)
            Framewised_IS10(videopath)