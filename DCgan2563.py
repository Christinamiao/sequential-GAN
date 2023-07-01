import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import utils
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from config import cfg

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
        Argument:
            position: length of input sequence
            d_model: the depth of hidden
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (?, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (?, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (?, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (?, num_heads, seq_len_q, depth)
        # attention_weights.shape == (?, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = utils.scaled_dot_product_attention(
            q, k, v)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (?, seq_len_q, d_model)

        return output, attention_weights

class encoder_block(Model):
    def __init__(self, out_filters):
        super(encoder_block, self).__init__()
        init = tf.initializers.RandomNormal(stddev = 0.02)
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(out_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
    def call(self,input_tensor, training = False):
        output = self.model(input_tensor, training=training)
        return output

class decoder_block(Model):
    def __init__(self, out_filters,dropout=True):
        super(decoder_block, self).__init__()
        init = tf.initializers.RandomNormal(stddev = 0.02)
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2DTranspose(out_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.model.add(layers.BatchNormalization())
        if dropout:
            self.model.add(layers.Dropout(0.5))
    def call(self, input_tensor, skip_in, training = False, step = None):
        g = self.model(input_tensor,training = training)
        if step == None:
            g = layers.Concatenate()([g, skip_in])
        else:
            g = layers.Concatenate()([g, g])
        g = layers.Activation('relu')(g)
        return g


class seq_encoder(tf.keras.Model):
    def __init__(self, seq_length):
        super(seq_encoder, self).__init__()
        self.seq_length = seq_length
        init = tf.initializers.RandomNormal(stddev=0.02)
        shape = (cfg.data.image_size, cfg.data.image_size, cfg.data.image_channel)
        self.encoder1 = tf.keras.Sequential()
        self.encoder1.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                                        kernel_initializer=init, input_shape=shape))
        self.encoder1.add(layers.LeakyReLU(alpha=0.2))
        self.encoder2 = encoder_block(128)  # 64x64x128
        self.encoder3 = encoder_block(256)  # 32x32x256
        self.encoder4 = encoder_block(512)  # 16x16x512
        self.encoder5 = encoder_block(512)  # 8x8x512
        self.encoder6 = encoder_block(512)  # 4x4x512
        self.encoder7 = encoder_block(512)  # 2x2x512

        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
        self.encoder.add(layers.Activation('relu'))
        # 1x1x512
        self.sequence_decoder = tf.keras.Sequential()
        self.sequence_decoder.add(layers.Dense(128 * self.seq_length,
                                               input_shape=(512,)))
        self.sequence_decoder.add(layers.LeakyReLU())
        self.sequence_decoder.add(layers.Reshape((-1, self.seq_length, 128)))
        self.pos_encoding = positional_encoding(self.seq_length, 128)
        self.mha = MultiHeadAttention(d_model=512, num_heads=8)

    def call(self, input_img, training=False):
        e1 = self.encoder1(input_img, training)
        e2 = self.encoder2(e1, training)
        e3 = self.encoder3(e2, training)
        e4 = self.encoder4(e3, training)
        e5 = self.encoder5(e4, training)
        e6 = self.encoder6(e5, training)
        e7 = self.encoder7(e6, training)
        feature = self.encoder(e7, training)

        hidden_encoders = [e1, e2, e3, e4, e5, e6, e7]
        if self.seq_length != 1:
            hidden_features = tf.reshape(feature, (-1, 512))
            hidden_features = self.sequence_decoder(hidden_features, training)
            hidden_features += self.pos_encoding[:, :self.seq_length, :]
            hidden_features, attention_weights = self.mha(hidden_features, hidden_features, hidden_features)
            features = []
            for i in range(self.seq_length):
                feature_i = hidden_features[:, i, :]
                feature_i = tf.reshape(feature_i, [-1, 1, 1, 512])
                features.append(feature_i)  # 4,batch_size,1,1,512
        else:
            features = feature  # batch_size,1,1,512

        return hidden_encoders, features
        
class Generator(tf.keras.Model):
    def __init__(self, seq_length):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        init = tf.initializers.RandomNormal(stddev=0.02)
        shape = (cfg.data.image_size,cfg.data.image_size,cfg.data.image_channel)
        self.encoder1 = tf.keras.Sequential()
        self.encoder1.add(layers.Conv2D(64, (4,4), strides=(2,2),padding='same',
                                               kernel_initializer=init ,input_shape=shape))
        self.encoder1.add(layers.LeakyReLU(alpha=0.2))
        self.encoder2 = encoder_block(128)#64x64x128
        self.encoder3 = encoder_block(256)#32x32x256
        self.encoder4 = encoder_block(512)#16x16x512
        self.encoder5 = encoder_block(512)#8x8x512
        self.encoder6 = encoder_block(512)#4x4x512
        self.encoder7 = encoder_block(512)#2x2x512

        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.encoder.add(layers.Activation('relu'))
        #1x1x512

        # self.sequence_decoder = tf.keras.Sequential()
        # self.sequence_decoder.add(layers.Dense(128*seq_length,
        #                           input_shape=(512,)))
        # self.sequence_decoder.add(layers.LeakyReLU())
        # self.sequence_decoder.add(layers.Reshape((-1, seq_length, 128)))
        # self.pos_encoding = positional_encoding(seq_length, 128)
        # self.mha = MultiHeadAttention(d_model=512, num_heads=8)



        self.decoder1 = decoder_block(512)#2x2x1024
        self.decoder2 = decoder_block(512)#4x4x1024
        self.decoder3 = decoder_block(512)#8x8x1024
        self.decoder4 = decoder_block(512, dropout=False)#16x16x1024
        self.decoder5 = decoder_block(256, dropout=False)#32x32x512
        self.decoder6 = decoder_block(128, dropout=False)#64x64x256
        self.decoder7 = decoder_block(64, dropout=False)#128x128x128

        self.decoder = tf.keras.Sequential()
        self.decoder.add(layers.Conv2DTranspose(cfg.data.image_channel, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))#256x256x1
        self.decoder.add(layers.Activation('tanh'))

        self.conta = layers.Concatenate()

    def call(self,input_img, training=False):
        e1 = self.encoder1(input_img,training)
        e2 = self.encoder2(e1,training)
        e3 = self.encoder3(e2, training)
        e4 = self.encoder4(e3, training)
        e5 = self.encoder5(e4, training)
        e6 = self.encoder6(e5, training)
        e7 = self.encoder7(e6, training)


        feature = self.encoder(e7,training)

        # if self.seq_length != 1:
        #     hidden_features = tf.reshape(feature, (-1, 512))
        #     hidden_features = self.sequence_decoder(hidden_features, training)
        #     hidden_features += self.pos_encoding[:, :self.seq_length, :]
        #     hidden_features, attention_weights = self.mha(hidden_features, hidden_features, hidden_features)
        #     feature = tf.reshape(hidden_features, [-1, 1, 1, 512])

        # e1_ = utils.copy_feature(e1,self.seq_length)
        # e2_ = utils.copy_feature(e2,self.seq_length)
        # e3_ = utils.copy_feature(e3,self.seq_length)
        # e4_ = utils.copy_feature(e4,self.seq_length)
        # e5_ = utils.copy_feature(e5,self.seq_length)
        # e6_ = utils.copy_feature(e6,self.seq_length)
        # e7_ = utils.copy_feature(e7,self.seq_length)

        # d1 = self.decoder1(feature, e7_,training)
        # d2 = self.decoder2(d1, e6_,training)
        # d3 = self.decoder3(d2, e5_,training)
        # d4 = self.decoder4(d3, e4_,training)
        # d5 = self.decoder5(d4, e3_,training)
        # d6 = self.decoder6(d5, e2_,training)
        # d7 = self.decoder7(d6, e1_,training)
        d1 = self.decoder1(feature, e7, training)
        d2 = self.decoder2(d1, e6, training)
        d3 = self.decoder3(d2, e5, training)
        d4 = self.decoder4(d3, e4, training)
        d5 = self.decoder5(d4, e3, training)
        d6 = self.decoder6(d5, e2, training)
        d7 = self.decoder7(d6, e1, training)
        images = self.decoder(d7,training)
        images = tf.reshape(images, [-1, self.seq_length, cfg.data.image_size,cfg.data.image_size,cfg.data.image_channel])

        return images

class Generator1(tf.keras.Model):
    def __init__(self):
        super(Generator1, self).__init__()
        init = tf.initializers.RandomNormal(stddev=0.02)
        self.decoder1 = decoder_block(512)#2x2x1024
        self.decoder2 = decoder_block(512)#4x4x1024
        self.decoder3 = decoder_block(512)#8x8x1024
        self.decoder4 = decoder_block(512, dropout=False)#16x16x1024
        self.decoder5 = decoder_block(256, dropout=False)#32x32x512
        self.decoder6 = decoder_block(128, dropout=False)#64x64x256
        self.decoder7 = decoder_block(64, dropout=False)#128x128x128

        self.decoder = tf.keras.Sequential()
        self.decoder.add(layers.Conv2DTranspose(cfg.data.image_channel, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))#256x256x1
        self.decoder.add(layers.Activation('tanh'))

    def call(self,hidden_encoders,feature, training=False):
        step = 1
        d1 = self.decoder1(feature, hidden_encoders[6],training)
        d2 = self.decoder2(d1, hidden_encoders[5],training)
        d3 = self.decoder3(d2, hidden_encoders[4],training)
        d4 = self.decoder4(d3, hidden_encoders[3],training)
        d5 = self.decoder5(d4, hidden_encoders[2],training)
        d6 = self.decoder6(d5, hidden_encoders[1],training)
        d7 = self.decoder7(d6, hidden_encoders[0],training)
        image = self.decoder(d7,training)#batch_size,256,256,1
        image = tf.expand_dims(image, 1)#batch_size,1,256,256,1

        return image

class Generator2(tf.keras.Model):
    def __init__(self):
        super(Generator2, self).__init__()
        init = tf.initializers.RandomNormal(stddev=0.02)
        self.decoder1 = decoder_block(512)#2x2x1024
        self.decoder2 = decoder_block(512)#4x4x1024
        self.decoder3 = decoder_block(512)#8x8x1024
        self.decoder4 = decoder_block(512, dropout=False)#16x16x1024
        self.decoder5 = decoder_block(256, dropout=False)#32x32x512
        self.decoder6 = decoder_block(128, dropout=False)#64x64x256
        self.decoder7 = decoder_block(64, dropout=False)#128x128x128

        self.decoder = tf.keras.Sequential()
        self.decoder.add(layers.Conv2DTranspose(cfg.data.image_channel, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))#256x256x1
        self.decoder.add(layers.Activation('tanh'))

    def call(self,hidden_encoders,feature, training=False):
        step = 2
        d1 = self.decoder1(feature, hidden_encoders[6],training)
        d2 = self.decoder2(d1, hidden_encoders[5],training)
        d3 = self.decoder3(d2, hidden_encoders[4],training)
        d4 = self.decoder4(d3, hidden_encoders[3],training)
        d5 = self.decoder5(d4, hidden_encoders[2],training)
        d6 = self.decoder6(d5, hidden_encoders[1],training)
        d7 = self.decoder7(d6, hidden_encoders[0],training)
        image = self.decoder(d7,training)#batch_size,256,256,1
        image = tf.expand_dims(image, 1)#batch_size,1,256,256,1

        return image

class Generator3(tf.keras.Model):
    def __init__(self):
        super(Generator3, self).__init__()
        init = tf.initializers.RandomNormal(stddev=0.02)
        self.decoder1 = decoder_block(512)#2x2x1024
        self.decoder2 = decoder_block(512)#4x4x1024
        self.decoder3 = decoder_block(512)#8x8x1024
        self.decoder4 = decoder_block(512, dropout=False)#16x16x1024
        self.decoder5 = decoder_block(256, dropout=False)#32x32x512
        self.decoder6 = decoder_block(128, dropout=False)#64x64x256
        self.decoder7 = decoder_block(64, dropout=False)#128x128x128

        self.decoder = tf.keras.Sequential()
        self.decoder.add(layers.Conv2DTranspose(cfg.data.image_channel, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))#256x256x1
        self.decoder.add(layers.Activation('tanh'))

    def call(self,hidden_encoders,feature, training=False):
        step = 3
        d1 = self.decoder1(feature, hidden_encoders[6],training)
        d2 = self.decoder2(d1, hidden_encoders[5],training)
        d3 = self.decoder3(d2, hidden_encoders[4],training)
        d4 = self.decoder4(d3, hidden_encoders[3],training)
        d5 = self.decoder5(d4, hidden_encoders[2],training)
        d6 = self.decoder6(d5, hidden_encoders[1],training)
        d7 = self.decoder7(d6, hidden_encoders[0],training)
        image = self.decoder(d7,training)#batch_size,256,256,1
        image = tf.expand_dims(image, 1)#batch_size,1,256,256,1

        return image

class Generator4(tf.keras.Model):
    def __init__(self):
        super(Generator4, self).__init__()
        init = tf.initializers.RandomNormal(stddev=0.02)
        self.decoder1 = decoder_block(512)#2x2x1024
        self.decoder2 = decoder_block(512)#4x4x1024
        self.decoder3 = decoder_block(512)#8x8x1024
        self.decoder4 = decoder_block(512, dropout=False)#16x16x1024
        self.decoder5 = decoder_block(256, dropout=False)#32x32x512
        self.decoder6 = decoder_block(128, dropout=False)#64x64x256
        self.decoder7 = decoder_block(64, dropout=False)#128x128x128

        self.decoder = tf.keras.Sequential()
        self.decoder.add(layers.Conv2DTranspose(cfg.data.image_channel, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))#256x256x1
        self.decoder.add(layers.Activation('tanh'))

    def call(self,hidden_encoders,feature, training=False):
        d1 = self.decoder1(feature, hidden_encoders[6],training)
        d2 = self.decoder2(d1, hidden_encoders[5],training)
        d3 = self.decoder3(d2, hidden_encoders[4],training)
        d4 = self.decoder4(d3, hidden_encoders[3],training)
        d5 = self.decoder5(d4, hidden_encoders[2],training)
        d6 = self.decoder6(d5, hidden_encoders[1],training)
        d7 = self.decoder7(d6, hidden_encoders[0],training)
        image = self.decoder(d7,training)#batch_size,256,256,1
        image = tf.expand_dims(image, 1)#batch_size,1,256,256,1

        return image


class Discriminator(tf.keras.Model):
    """
        This class is the discriminator
        seq_lengths: max_len
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        init = tf.initializers.RandomNormal(stddev=0.02)
        shape = (cfg.data.image_size, cfg.data.image_size, cfg.data.image_channel)

        self.d_model = tf.keras.Sequential()
        self.d_model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init,
                                       input_shape = shape))
        self.d_model.add(layers.LeakyReLU(alpha=0.2))
        # 256x256x1 to 128x128x64
        self.d_model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
        self.d_model.add(layers.BatchNormalization())
        self.d_model.add(layers.LeakyReLU(alpha=0.2))
        # 64x64x128
        self.d_model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
        self.d_model.add(layers.BatchNormalization())
        self.d_model.add(layers.LeakyReLU(alpha=0.2))
        # 32x32x256
        self.d_model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
        self.d_model.add(layers.BatchNormalization())
        self.d_model.add(layers.LeakyReLU(alpha=0.2))
        # 16x16x512
        self.d_model.add(layers.Conv2D(512, (4, 4), padding='same', kernel_initializer=init))
        self.d_model.add(layers.BatchNormalization())
        self.d_model.add(layers.LeakyReLU(alpha=0.2))
        # 16x16x512
        self.d_model.add(layers.Conv2D(1, (4, 4),padding='same',kernel_initializer=init))
        self.d_model.add(layers.Activation('tanh'))
        # 16x16x1
        self.d_model.add(layers.Flatten())
        self.d_model.add(layers.Dense(1))

    def call(self, images, training=False):
        images = tf.reshape(images, [-1, cfg.data.image_size,cfg.data.image_size,cfg.data.image_channel])
        fake_score = self.d_model(images,training)# batch_size个score

        return fake_score
