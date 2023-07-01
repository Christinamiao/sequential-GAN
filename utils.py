import os
import time
import numpy as np
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
from config import cfg


def define_save_dirs(root_dir='runs/'):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    timestamp = str(int(time.time()))
    if cfg.train.TRAIN:
        floder1 = 'train'
    else:
        floder1 = 'test'
    root_dir = os.path.join(root_dir,floder1)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if cfg.train.tnn == 0:
        timestamp = timestamp + 'step1'
    elif cfg.train.tnn == 1:
        timestamp = timestamp + 'step234'
    else:
        timestamp = timestamp + 'both'
    if not os.path.exists(os.path.join(root_dir, timestamp)):
        os.makedirs(os.path.join(root_dir, timestamp))
    checkpoint_dir = os.path.join(root_dir, timestamp, 'training_checkpoints')
    checkpoint_dir1 = os.path.join(root_dir, timestamp, 'training_checkpoints1')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint_prefix1 = os.path.join(checkpoint_dir1, 'ckpt')

    save_dirs = {}
    save_dirs['checkpoint_dir'] = checkpoint_dir
    save_dirs['checkpoint_dir1'] = checkpoint_dir1
    save_dirs['checkpoint_prefix'] = checkpoint_prefix
    save_dirs['checkpoint_prefix1'] = checkpoint_prefix1
    save_dirs['img'] = os.path.join(root_dir, timestamp)

    return save_dirs

def load_image(path, imsize, total_depth):
    im = PIL.Image.open(path)
    # img1 = np.array(im)
    # print(np.min(img1), np.max(img1))
    # plt.imshow(im, cmap='gray')
    # plt.axis('off')
    # plt.savefig('image_1.png')
    # plt.close()

    if (total_depth == 1):
        # color to bw image for Ji dataset
        im = im.convert("L")
        # img2 = np.array(im)
        # print(np.min(img2), np.max(img2),img2.shape)
        # plt.imshow(im, cmap='gray')
        # plt.axis('off')
        # plt.savefig('image_2.png')
        # plt.close()

    if im.mode == 'RGBA':
        r, g, b, a = im.split()
        im = PIL.Image.merge("RGB", (r, g, b))
    im = im.resize((imsize, imsize))
    im = np.array(im)

    if (len(im.shape) == 2):
        im = np.expand_dims(im, -1)

    if (total_depth == 3) and (im.shape[-1] == 1):
        im = np.tile(im, [1, 1, 3])
    return im

def prepare_data0(input_dir, floder,image_size,image_channel,batch_size):
    x_train = []
    max_len = 0
    actual_lengths = []
    floder_path = os.path.join(input_dir,floder)
    image_floder = os.listdir(floder_path)
    for each in image_floder:
        x_seq = []
        img_path = os.path.join(floder_path,each)
        if '.jpg' in each:
            img = load_image(img_path, image_size, image_channel).astype('float32')
            img = (img - 127.5) / 127.5
            x_seq.append(img)
            actual_lengths.append(len(x_seq))
            assert ((np.max(x_seq) <= 1.0) and (np.min(x_seq) >= -1.0))
        x_seq = np.array(x_seq)
        x_train.append(x_seq)
        if len(x_seq) > max_len:
            max_len = len(x_seq)

    actual_lengths = np.array(actual_lengths)
    print("length of dataset:", len(x_train))
    print('max_len = ',max_len)

    def gen1():
        for el in x_train:
            yield el

    img_data_set = tf.data.Dataset.from_generator(gen1, output_types=tf.float32)
    lengths = tf.data.Dataset.from_tensor_slices(actual_lengths)

    train_data_set = tf.data.Dataset.zip((img_data_set, lengths))

    train_data_set = train_data_set.repeat(cfg.data.repeat_num).shuffle(
        len(x_train)).padded_batch(
        batch_size,
        padded_shapes=([max_len, cfg.data.image_size,cfg.data.image_size,cfg.data.image_channel], []),
        drop_remainder=True)
    return train_data_set, max_len

def prepare_data(input_dir, floder,image_size,image_channel,batch_size):
    x_train = []
    max_len = 0
    actual_lengths = []
    floder_path = os.path.join(input_dir,floder)
    image_floder = os.listdir(floder_path)
    # print(image_floder)
    for each in image_floder:
        x_seq = []
        path = os.path.join(floder_path,each)
        img_name = os.listdir(path)
        if (len(img_name) == 4) or (len(img_name) == 5):
            for every in img_name:
                if '_0_1.jpg' in every:
                    img_path = os.path.join(path,every)
                    img = load_image(img_path, image_size, image_channel).astype('float32')
                    img = (img - 127.5) / 127.5
                    x_seq.append(img)
            for every in img_name:
                if '_1_1.jpg' in every:
                    img_path = os.path.join(path,every)
                    img = load_image(img_path, image_size, image_channel).astype('float32')
                    img = (img - 127.5) / 127.5
                    x_seq.append(img)
            for every in img_name:
                if '_2_1.jpg' in every:
                    img_path = os.path.join(path,every)
                    img = load_image(img_path, image_size, image_channel).astype('float32')
                    img = (img - 127.5) / 127.5
                    x_seq.append(img)
            for every in img_name:
                if '_3_1.jpg' in every:
                    img_path = os.path.join(path,every)
                    img = load_image(img_path, image_size, image_channel).astype('float32')
                    img = (img - 127.5) / 127.5
                    x_seq.append(img)

            x_seq = np.array(x_seq)
            x_train.append(x_seq)
            if len(x_seq) > max_len:
                max_len = len(x_seq)

            actual_lengths.append(len(x_seq))
            assert ((np.max(x_seq) <= 1.0) and (np.min(x_seq) >= -1.0))

    actual_lengths = np.array(actual_lengths)
    print("length of dataset:", len(x_train))
    print('max_len = ',max_len)

    def gen1():
        for el in x_train:
            yield el

    img_data_set = tf.data.Dataset.from_generator(gen1, output_types=tf.float32)
    lengths = tf.data.Dataset.from_tensor_slices(actual_lengths)

    train_data_set = tf.data.Dataset.zip((img_data_set, lengths))

    train_data_set = train_data_set.repeat(cfg.data.repeat_num).shuffle(
        len(x_train)).padded_batch(
        batch_size,
        padded_shapes=([max_len, cfg.data.image_size,cfg.data.image_size,cfg.data.image_channel], []),
        drop_remainder=True)
    return train_data_set, max_len

def save_images(epoch, gen_images,test_img, save_path,batch_cnt = None):
    batch_num = gen_images.shape[0]
    if batch_cnt == None:
        cnt = 0
    else:
        cnt = (batch_cnt - 1) * batch_num

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(batch_num):
        cnt += 1
        img = gen_images[i]
        if len(test_img.shape) == 4:  # (batch_num,64/256,64/256,channel)
            real_img = test_img[i]
        elif len(test_img.shape) == 3:  # (64/256,64/256,channel)
            real_img = test_img
        else:  # (batch_num,4,256,256,channel)
            real_img = test_img[i]
        if img.shape[-1] == 3:  # channel = 3
            image = np.uint8((img * 127.5 + 127.5).numpy())
            real_img = np.uint8((real_img * 127.5 + 127.5).numpy())
            # image = np.concatenate(image, axis=1)
            # image = np.concatenate((image, real_img), axis=1)
            if len(real_img.shape) == 3:
                image = np.concatenate((image, real_img), axis=1)
            else:
                real_img = np.concatenate(real_img, axis=1)
                image = np.concatenate((image,real_img),axis=0)
            plt.imshow(image)
            plt.axis('off')

        else:  # channel = 1
            if img.shape[0] == 3:
                img = tf.reshape(img, [-1, cfg.data.image_size, cfg.data.image_size])
                image = img.numpy() * 127.5 + 127.5
                image = np.concatenate(image, axis=1)
                if len(real_img.shape) == 3:
                    real_img = tf.reshape(real_img, [cfg.data.image_size, cfg.data.image_size])
                    real_img = real_img.numpy() * 127.5 + 127.5
                    image = np.concatenate((image, real_img), axis=1)
                else:
                    real_img = tf.reshape(real_img, [-1, cfg.data.image_size, cfg.data.image_size])
                    real_img = real_img[1:,:,:]

                    real_img = real_img.numpy() * 127.5 + 127.5
                    real_img = np.concatenate(real_img, axis=1)
                    image = np.concatenate((image, real_img), axis=0)
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            else:
                img = tf.reshape(img, [-1, cfg.data.image_size, cfg.data.image_size])
                image = img.numpy() * 127.5 + 127.5
                image = np.concatenate(image, axis=1)
                if len(real_img.shape) == 3:
                    real_img = tf.reshape(real_img, [cfg.data.image_size, cfg.data.image_size])
                    real_img = real_img.numpy() * 127.5 + 127.5
                    image = np.concatenate((image, real_img), axis=1)
                else:
                    real_img = tf.reshape(real_img, [-1, cfg.data.image_size, cfg.data.image_size])
                    real_img = real_img.numpy() * 127.5 + 127.5
                    real_img = np.concatenate(real_img, axis=1)
                    image = np.concatenate((image, real_img), axis=0)
                plt.imshow(image, cmap='gray')
                plt.axis('off')

        plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}_{:d}.png'.format(epoch, cnt)))
        plt.close()

def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def create_padding_mask(seq):
    # print(seq.shape)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # print(seq.shape)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# class img_encoder(Model):
#     def __init__(self):
#         super(img_encoder, self).__init__()
#         self.model = tf.keras.Sequential()
#         self.model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=[cfg.data.image_size,
#                                                   cfg.data.image_size,
#                                                   cfg.data.image_channel]))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.5))
#         # 64x64x1/3 to 32x32x16 # 256x256x1/3 to 128x128x16
#
#         self.model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.3))
#         # 32x32x16 to 16x16x32 # 128x128x16 to 64x64x32
#
#         self.model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.3))
#         # 16x16x32 to 8x8x64 # 64x64x32 to 32x32x64
#
#         self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.3))
#         # 8x8x64 to 4x4x128 # 32x32x64 to 16x16x128
#
#         self.model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.3))
#         # 4x4x128 to 2x2x256 # 16x16x128 to 8x8x256
#
#         self.model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.3))
#         # 2x2x256 to 1x1x512 # 8x8x256 to 4x4x512
#
#         #add for 256size image
#         self.model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.3))
#         # 4x4x512 to 2x2x1024
#         self.model.add(layers.Conv2D(2048, (5, 5), strides=(2, 2), padding='same'))
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dropout(0.3))
#         # 2x2x1024 to 1x1x2048
#         # self.model.add(layers.Flatten())
#
#         # # self.model.add(layers.Dense(1000, use_bias=False, input_shape=(512,)))
#         self.model.add(layers.Dense(1000, use_bias=False, input_shape=(2048,)))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.LeakyReLU())
#         self.model.add(layers.Dense(500, activation='relu'))
#         self.model.add(layers.Dense(100, activation='relu'))
#
#     def call(self,image,training=False):
#         full_num = cfg.data.max_len - 1
#         if image.shape[0] != 1:
#             img1 = image[0][full_num]
#             img2 = image[1][full_num]
#             img = tf.stack([img1, img2], 0)
#             for i in range(2, image.shape[0]):
#                 img1 = image[i][full_num]
#                 img1 = tf.reshape(img1, [-1, cfg.data.image_size, cfg.data.image_size, cfg.data.image_channel])
#                 img = tf.concat([img, img1], 0)
#             # print('111', img.shape)
#             # print(np.min(img.numpy()), np.max(img.numpy()))
#             img_feature = self.model(img, training= training)
#             img_feature = tf.reshape(img_feature,[-1,100])
#             # print('222', img_feature.shape)
#             # print(np.min(img_feature.numpy()), np.max(img_feature.numpy()))
#             # sys.exit()
#
#         else:
#             img_feature = self.model(image, training= training)
#         return img_feature

def copy_feature(input_tensor,n_copy):
    if n_copy == 1:
        tensor = input_tensor
    else:
        tensors = tf.split(input_tensor, input_tensor.shape[0], 0)
        print(len(tensors))
        tensor1 = tensors[0]
        print(tensor1.shape)
        tensor1 = tf.tile(tensor1, (n_copy, 1, 1, 1))
        print(tensor1.shape)
        tensor2 = tensors[1]
        tensor2 = tf.tile(tensor2, (n_copy, 1, 1, 1))
        tensor = tf.concat([tensor1,tensor2],0)
        for i in range(2,input_tensor.shape[0]):
            tensor1 = tensors[i]
            tensor1 = tf.tile(tensor1, (n_copy, 1, 1, 1))
            tensor = tf.concat([tensor,tensor1],0)
    return tensor

def dislocation(input_tensor):
    batch_size = cfg.data.BATCH_SIZE
    seq_length = cfg.train.seq_length
    dis_tensor = input_tensor[1*seq_length:2*seq_length]
    for i in range(2,batch_size):
        add_tensor = input_tensor[i*seq_length:(i + 1)*seq_length]
        dis_tensor = tf.concat([dis_tensor,add_tensor],0)
    add_tensor = input_tensor[0:seq_length]
    dis_tensor = tf.concat([dis_tensor,add_tensor],0)
    return dis_tensor

def reduce_lr(lr):
    lr = 0.98 * lr
    return lr


