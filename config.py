from easydict import EasyDict as edict
import datetime
import os

__C = edict()
cfg = __C

__C.GPU = edict()
__C.GPU.train = '2'
__C.GPU.test = '2'



__C.data = edict()
__C.data.image_size = 256
__C.data.image_channel = 1
__C.data.BATCH_SIZE = 4
__C.data.max_len = 4
__C.data.repeat_num = 1

__C.train = edict()
__C.train.EPOCHS = 500
__C.train.EPOCHS1 = 1000
__C.train.checkpoint_save_step = 50
__C.train.image_save_step = 50
# __C.train.TRAIN = True
__C.train.TRAIN = False
__C.train.tnn = 1   #trainer number 0=trainer 1=trainer1 2=both
__C.train.restore = 1  #0/1
__C.train.seq_length = 4
__C.train.step = 1
__C.train.gen_lr = 1e-4
__C.train.dis_lr = 1e-4
__C.train.imgtr = True
# __C.train.imgtr = False
__C.train.ITnumber = 7  # (1,7)/(1,7)(1-12,14-28,30)





__C.path = edict()
__C.path.input_dir = "./cut_data_set1/"
__C.path.floder = "sketch"
# __C.path.input_dir1 = "./data_set"
# __C.path.floder1 = "sketch/woman/positive_face/young"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
__C.path.train_log_dir = 'logs/' #+ current_time + '/train/'
__C.path.checkpoint_restore = './checkpoint_restore'
__C.path.checkpoint_restore1 = './checkpoint_restore1'
__C.path.test_floder = './test_img'
__C.path.pathIn = os.path.join(__C.path.test_floder, '{:d}.jpg'.format(__C.train.ITnumber))  # ./test_img/1.jpg'
__C.path.pathOut = './it_result'  # './result/result_img01.jpg'
__C.path.model = './model/colorful.t7'
# if happens error:module gast has no attribute 'Num',use  pip install'gast==0.2.2'