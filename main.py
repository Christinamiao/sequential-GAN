from utils import prepare_data,prepare_data0,define_save_dirs,save_images,load_image
from train_test import Trainer,Trainer1
import tensorflow as tf
from config import cfg
import os
from image_transform import style_transfer


TRAIN = cfg.train.TRAIN
STEP = cfg.train.step
IT = cfg.train.imgtr

if TRAIN:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU.train
    train_data_set, max_len= prepare_data(cfg.path.input_dir, cfg.path.floder,
                                                           cfg.data.image_size, cfg.data.image_channel,
                                                           cfg.data.BATCH_SIZE)
    if cfg.train.tnn == 0:
        trainer = Trainer(1)
        trainer.train(train_data_set)
    elif cfg.train.tnn == 1:
        trainer1 = Trainer1(cfg.train.seq_length)
        trainer1.train(train_data_set)
    else:
        trainer = Trainer(1)
        trainer.train(train_data_set)
        trainer1 = Trainer1(cfg.train.seq_length)
        trainer1.train(train_data_set)

else:
    if IT != True:

        if STEP == None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU.test
            max_len = cfg.data.max_len
            trainer = Trainer(1)
            trainer1 = Trainer1(cfg.train.seq_length)
            test_floder = cfg.path.test_floder
            save_dirs = define_save_dirs()
            save_pathes = os.path.join(save_dirs['img'], cfg.path.train_log_dir)
            path = os.listdir(test_floder)
            test_num = 1
            for each in path:
                if '.jpg' in each:
                    path = os.path.join(test_floder,each)
                    gen_image = trainer.test(path,save_pathes)
                    gen_image1 = trainer1.test(path,save_pathes)
                    test_num = test_num + 1
                    gen_images = tf.concat([gen_image,gen_image1],1)
                    test_img = load_image(path, cfg.data.image_size, cfg.data.image_channel)
                    test_img = (test_img - 127.5) / 127.5
                    save_images(test_num, gen_images, test_img, save_pathes)
        else:
            if STEP == 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU.test
                max_len = cfg.data.max_len
                trainer = Trainer(1)
                test_floder = cfg.path.test_floder
                save_dirs = define_save_dirs()
                save_pathes = os.path.join(save_dirs['img'], cfg.path.train_log_dir)
                path = os.listdir(test_floder)
                test_num = 1
                for each in path:
                    if '.jpg' in each:
                        path = os.path.join(test_floder, each)
                        gen_image = trainer.test(path, save_pathes)
                        test_num = test_num + 1
                        test_img = load_image(path, cfg.data.image_size, cfg.data.image_channel)
                        test_img = (test_img - 127.5) / 127.5
                        save_images(test_num, gen_image, test_img, save_pathes)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU.test
                max_len = cfg.data.max_len
                trainer = Trainer1(4)
                test_floder = cfg.path.test_floder
                save_dirs = define_save_dirs()
                save_pathes = os.path.join(save_dirs['img'], cfg.path.train_log_dir)
                path = os.listdir(test_floder)
                test_num = 1
                for each in path:
                    if '.jpg' in each:
                        path = os.path.join(test_floder, each)
                        gen_image = trainer.test(path, save_pathes)
                        test_num = test_num + 1
                        test_img = load_image(path, cfg.data.image_size, cfg.data.image_channel)
                        test_img = (test_img - 127.5) / 127.5
                        save_images(test_num, gen_image[STEP-2], test_img, save_pathes)

    else:
        number = cfg.train.ITnumber  # (1,28)
        pathIn = cfg.path.pathIn # ./test_img/1.jpg'
        pathOut = cfg.path.pathOut #'./it_result'  # './result/result_img01.jpg'
        model = cfg.path.model #'./model/colorful.t7'
        style_transfer(pathIn, pathOut, model, number)