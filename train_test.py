from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import tensorflow as tf
import time
import losses
import utils
import matplotlib.pyplot as plt
from DCgan2563 import seq_encoder, Generator1, Generator2, Generator3, Generator4,Generator, Discriminator
from config import cfg


class Trainer():
    def __init__(self, seq_length):
        self.noise_dim = 100
        self.max_len = cfg.data.max_len
        self.seq_length = seq_length
        self.gen_lr = cfg.train.gen_lr
        self.dis_lr = cfg.train.dis_lr
        self.build_model()


    def build_model(self):
        self.generator = Generator(self.seq_length)
        self.discriminator = Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(self.gen_lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.dis_lr)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    @tf.function
    def train_step(self, real_img,input_img, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_images = self.generator(input_img,training = True)
            real_ind_scores = self.discriminator(real_img,training = True)
            fake_ind_scores = self.discriminator(gen_images,training = True)
            # dl_ind_scores = utils.dislocation(real_ind_scores)


            gen_loss_f = losses.generator_hinge_loss
            disc_loss_f = losses.discriminator_hinge_loss

            gen_ind_loss = gen_loss_f(fake_ind_scores, self.seq_length)
            gen_loss = gen_ind_loss
            disc_ind_loss = disc_loss_f(real_ind_scores, fake_ind_scores, self.seq_length)
            # disc_dl_ind_loss = disc_loss_f(real_ind_scores, dl_ind_scores, self.seq_length)
            disc_loss = disc_ind_loss #+ disc_dl_ind_loss

        gradients_of_generator = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('GAN_loss', gen_loss, step=epoch)
            tf.summary.scalar('DISC_loss', disc_loss, step=epoch)
            # for i in range(len(self.generator.trainable_variables)):
            #     name = self.generator.trainable_variables[i].name
            #     tf.summary.histogram(name + ' gen',self.generator.trainable_variables[i],step=epoch)
            #
            # for i in range(len(gradients_of_discriminator)):
            #     name = self.discriminator.trainable_variables[i].name
            #     tf.summary.histogram(name + ' disc',self.discriminator.trainable_variables[i],step=epoch)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss, gen_images

    def train(self, dataset):
        self.save_dirs = utils.define_save_dirs()
        self.save_path = os.path.join(self.save_dirs['img'], cfg.path.train_log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(self.save_path)
        self.checkpoint_dir = self.save_dirs['checkpoint_dir']
        self.checkpoint_prefix = self.save_dirs['checkpoint_prefix']
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_prefix, max_to_keep=1)
        if cfg.train.restore == 1:
            self.checkpoint.restore(tf.train.latest_checkpoint(cfg.path.checkpoint_restore))
        epochs = cfg.train.EPOCHS
        for epoch in range(epochs):
            start = time.time()
            gen_losses = []
            disc_losses = []
            train_save_path = os.path.join(self.save_path, 'epoch{:04d}'.format(epoch + 1))
            batch_cnt = 1

            for image_batch, actual_lengths in dataset:
                # image_batch = tf.random.shuffle(image_batch)
                n_full = self.max_len - 1
                step = 1
                n_step = step - 1
                input_img = image_batch[:,n_full,:,:,:]
                real_img = image_batch[:,n_step,:,:,:]
                gen_loss, disc_loss, gen_images = self.train_step(real_img, input_img, epoch)
                gen_losses.append(gen_loss.numpy())
                disc_losses.append(disc_loss.numpy())
                if (epoch + 1) % cfg.train.image_save_step == 0:
                    utils.save_images((epoch + 1), gen_images, real_img, train_save_path,batch_cnt)
                batch_cnt = batch_cnt + 1

            avg_gen_loss = np.mean(gen_losses)
            avg_disc_loss = np.mean(disc_losses)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            print('avg_gen_loss:',avg_gen_loss)
            print('avg_disc_loss:', avg_disc_loss)

            if (epoch + 1) % cfg.train.checkpoint_save_step == 0:
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                if not os.path.exists(self.checkpoint_prefix):
                    os.makedirs(self.checkpoint_prefix)
                # self.checkpoint.save(self.checkpoint_prefix)
                self.manager.save()
                self.gen_lr = utils.reduce_lr(self.gen_lr)
                self.dis_lr = utils.reduce_lr(self.dis_lr)


    def test(self,path,save_pathes):
        self.checkpoint.restore(tf.train.latest_checkpoint(cfg.path.checkpoint_restore))
        if not os.path.exists(save_pathes):
            os.makedirs(save_pathes)

        test_img = utils.load_image(path, cfg.data.image_size, cfg.data.image_channel)
        test_img = (test_img - 127.5) / 127.5
        test_img1 = tf.reshape(test_img, [-1, cfg.data.image_size, cfg.data.image_size, cfg.data.image_channel])
        gen_images = self.generator(test_img1)
        return gen_images


class Trainer1():
    def __init__(self,seq_length, pretrain=None, pre_path=None):
        self.noise_dim = 100
        self.max_len = cfg.data.max_len
        self.seq_length = seq_length
        self.gen_lr = cfg.train.gen_lr
        self.dis_lr = cfg.train.dis_lr
        self.build_model()

    def build_model(self):
        if self.seq_length == 1:
            if cfg.train.step == 1:
                self.generator = Generator1()
            elif cfg.train.step == 2:
                self.generator = Generator2()
            elif cfg.train.step == 3:
                self.generator = Generator3()
            else:
                self.generator = Generator4()
            self.discriminator = Discriminator()
            self.generator_optimizer = tf.keras.optimizers.Adam(self.gen_lr)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(self.dis_lr)
            self.seq_encoder = seq_encoder(self.seq_length)
            self.checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
                generator=self.generator,
                discriminator=self.discriminator,
                encoder=self.seq_encoder
            )
        else:
            # self.seq_encoder1 = seq_encoder(self.seq_length)
            self.seq_encoder2 = seq_encoder(self.seq_length)
            self.seq_encoder3 = seq_encoder(self.seq_length)
            self.seq_encoder4 = seq_encoder(self.seq_length)
            # self.generator1 = Generator1()
            # self.discriminator1 = Discriminator()
            self.generator2 = Generator2()
            self.discriminator2 = Discriminator()
            self.generator3 = Generator3()
            self.discriminator3 = Discriminator()
            self.generator4 = Generator4()
            self.discriminator4 = Discriminator()
            # self.generator1_optimizer = tf.keras.optimizers.Adam(self.gen_lr)
            self.generator2_optimizer = tf.keras.optimizers.Adam(self.gen_lr)
            self.generator3_optimizer = tf.keras.optimizers.Adam(self.gen_lr)
            self.generator4_optimizer = tf.keras.optimizers.Adam(self.gen_lr)
            # self.discriminator1_optimizer = tf.keras.optimizers.Adam(self.dis_lr)
            self.discriminator2_optimizer = tf.keras.optimizers.Adam(self.dis_lr)
            self.discriminator3_optimizer = tf.keras.optimizers.Adam(self.dis_lr)
            self.discriminator4_optimizer = tf.keras.optimizers.Adam(self.dis_lr)
            self.checkpoint = tf.train.Checkpoint(
                #generator1_optimizer=self.generator1_optimizer,
                generator2_optimizer=self.generator2_optimizer,
                generator3_optimizer=self.generator3_optimizer,
                generator4_optimizer=self.generator4_optimizer,
                #discriminator1_optimizer=self.discriminator1_optimizer,
                discriminator2_optimizer=self.discriminator2_optimizer,
                discriminator3_optimizer=self.discriminator3_optimizer,
                discriminator4_optimizer=self.discriminator4_optimizer,
                #gen1=self.generator1,
                gen2=self.generator2,
                gen3=self.generator3,
                gen4=self.generator4,
                # disc1=self.discriminator1,
                disc2=self.discriminator2,
                disc3=self.discriminator3,
                disc4=self.discriminator4,
                # encoder1=self.seq_encoder1,
                encoder2=self.seq_encoder2,
                encoder3=self.seq_encoder3,
                encoder4=self.seq_encoder4,
            )

    @tf.function
    def train_step(self, real_img, input_img, epoch):
        if self.seq_length == 1:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                hidden_encoders, feature = self.seq_encoder(input_img, training=True)
                gen_loss_f = losses.generator_hinge_loss
                disc_loss_f = losses.discriminator_hinge_loss

                gen_images = self.generator(hidden_encoders, feature, training=True)
                real_scores = self.discriminator(real_img, training=True)
                fake_scores = self.discriminator(gen_images, training=True)
                dl_scores = utils.dislocation(real_scores)

                gen_loss = gen_loss_f(fake_scores)
                disc_loss = disc_loss_f(real_scores, fake_scores)
                disc_dl_loss = disc_loss_f(real_scores, dl_scores)
                disc_loss = disc_loss + disc_dl_loss

            gradients_of_generator = gen_tape.gradient(gen_loss, (self.generator.trainable_variables + self.seq_encoder.trainable_variables))
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('GAN_loss', gen_loss, step=epoch)
                tf.summary.scalar('DISC_loss', disc_loss, step=epoch)

            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables + self.seq_encoder.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            return gen_loss, disc_loss, gen_images
        else:
            with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
                # hidden_encoders1, features1 = self.seq_encoder1(input_img, training=True)
                hidden_encoders2, features2 = self.seq_encoder2(input_img, training=True)
                hidden_encoders3, features3 = self.seq_encoder3(input_img, training=True)
                hidden_encoders4, features4 = self.seq_encoder4(input_img, training=True)
                gen_loss_f = losses.generator_hinge_loss
                disc_loss_f = losses.discriminator_hinge_loss
                # dl_loss_f = losses.discriminator_dl_loss

                # gen1_image = self.generator1(hidden_encoders1, features1[0], training=True)
                gen2_image = self.generator2(hidden_encoders2, features2[1], training=True)
                gen3_image = self.generator3(hidden_encoders3, features3[2], training=True)
                gen4_image = self.generator4(hidden_encoders4, features4[3], training=True)
                # real_score1 = self.discriminator1(real_img[0], training=True)
                real_score2 = self.discriminator2(real_img[1], training=True)
                real_score3 = self.discriminator3(real_img[2], training=True)
                real_score4 = self.discriminator4(real_img[3], training=True)
                # fake_score1 = self.discriminator1(gen1_image, training=True)
                fake_score2 = self.discriminator2(gen2_image, training=True)
                fake_score3 = self.discriminator3(gen3_image, training=True)
                fake_score4 = self.discriminator4(gen4_image, training=True)
                # dl_score1 = utils.dislocation(real_score1)
                # dl_score2 = utils.dislocation(real_score2)
                # dl_score3 = utils.dislocation(real_score3)
                # dl_score4 = utils.dislocation(real_score4)

                # gen1_loss = gen_loss_f(fake_score1)
                # disc1_loss = disc_loss_f(real_score1, fake_score1)
                # disc1_dl_loss = disc_loss_f(real_score1, dl_score1)
                # disc1_loss = disc1_loss + disc1_dl_loss * cfg.train.loss_lr

                gen2_loss = gen_loss_f(fake_score2)
                disc2_loss = disc_loss_f(real_score2, fake_score2)
                # disc2_dl_loss = dl_loss_f(real_score2, dl_score2)
                # disc2_loss = disc2_loss + disc2_dl_loss * 1

                gen3_loss = gen_loss_f(fake_score3)
                disc3_loss = disc_loss_f(real_score3, fake_score3)
                # disc3_dl_loss = dl_loss_f(real_score3, dl_score3)
                # disc3_loss = disc3_loss + disc3_dl_loss * 0.8

                gen4_loss = gen_loss_f(fake_score4)
                disc4_loss = disc_loss_f(real_score4, fake_score4)
                # disc4_dl_loss = dl_loss_f(real_score4, dl_score4)
                # disc4_loss = disc4_loss + disc4_dl_loss * 0

            gradients_of_generator4 = gen_tape.gradient(gen4_loss,
                                                        self.generator4.trainable_variables + self.seq_encoder4.trainable_variables)
            gradients_of_discriminator4 = disc_tape.gradient(disc4_loss, self.discriminator4.trainable_variables)
            # gradients_of_generator1 = gen_tape.gradient(gen1_loss,
            #                                             self.generator1.trainable_variables + self.seq_encoder1.trainable_variables)
            gradients_of_generator2 = gen_tape.gradient(gen2_loss,
                                                        self.generator2.trainable_variables + self.seq_encoder2.trainable_variables)
            gradients_of_generator3 = gen_tape.gradient(gen3_loss,
                                                        self.generator3.trainable_variables + self.seq_encoder3.trainable_variables)
            # gradients_of_discriminator1 = disc_tape.gradient(disc1_loss, self.discriminator1.trainable_variables)
            gradients_of_discriminator2 = disc_tape.gradient(disc2_loss, self.discriminator2.trainable_variables)
            gradients_of_discriminator3 = disc_tape.gradient(disc3_loss, self.discriminator3.trainable_variables)

            with self.train_summary_writer.as_default():
                # tf.summary.scalar('GAN1_loss', gen1_loss, step=epoch)
                tf.summary.scalar('GAN2_loss', gen2_loss, step=epoch)
                tf.summary.scalar('GAN3_loss', gen3_loss, step=epoch)
                tf.summary.scalar('GAN4_loss', gen4_loss, step=epoch)
                # tf.summary.scalar('DISC1_loss', disc1_loss, step=epoch)
                tf.summary.scalar('DISC2_loss', disc2_loss, step=epoch)
                tf.summary.scalar('DISC3_loss', disc3_loss, step=epoch)
                tf.summary.scalar('DISC4_loss', disc4_loss, step=epoch)

            self.generator4_optimizer.apply_gradients(zip(gradients_of_generator4,
                                                          self.generator4.trainable_variables + self.seq_encoder4.trainable_variables))
            self.discriminator4_optimizer.apply_gradients(
                zip(gradients_of_discriminator4, self.discriminator4.trainable_variables))
            # self.generator1_optimizer.apply_gradients(zip(gradients_of_generator1, self.generator1.trainable_variables))
            self.generator2_optimizer.apply_gradients(zip(gradients_of_generator2, self.generator2.trainable_variables))
            self.generator3_optimizer.apply_gradients(zip(gradients_of_generator3, self.generator3.trainable_variables))
            # self.discriminator1_optimizer.apply_gradients(
            #     zip(gradients_of_discriminator1, self.discriminator1.trainable_variables))
            self.discriminator2_optimizer.apply_gradients(
                zip(gradients_of_discriminator2, self.discriminator2.trainable_variables))
            self.discriminator3_optimizer.apply_gradients(
                zip(gradients_of_discriminator3, self.discriminator3.trainable_variables))

            gen_loss = [gen2_loss, gen3_loss, gen4_loss]
            disc_loss = [disc2_loss, disc3_loss, disc4_loss]
            gen_image = [gen2_image, gen3_image, gen4_image]
            return gen_loss, disc_loss, gen_image

    def train(self, dataset):
        self.save_dirs = utils.define_save_dirs()
        self.save_path = os.path.join(self.save_dirs['img'], cfg.path.train_log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(self.save_path)
        self.checkpoint_dir = self.save_dirs['checkpoint_dir1']
        self.checkpoint_prefix = self.save_dirs['checkpoint_prefix1']
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_prefix, max_to_keep=1)
        if cfg.train.restore == 1:
            self.checkpoint.restore(tf.train.latest_checkpoint(cfg.path.checkpoint_restore1))
        epochs = cfg.train.EPOCHS1
        for epoch in range(epochs):
            start = time.time()
            if self.seq_length == 1:
                gen_losses = []
                disc_losses = []
            else:
                # gen1_losses = []
                gen2_losses = []
                gen3_losses = []
                gen4_losses = []
                # disc1_losses = []
                disc2_losses = []
                disc3_losses = []
                disc4_losses = []

            train_save_path = os.path.join(self.save_path, 'epoch{:04d}'.format(epoch + 1))
            batch_cnt = 1

            for image_batch in dataset:
                image_batch = image_batch[0]
                # image_batch = tf.random.shuffle(image_batch)
                # print(image_batch[1])
                n_full = self.max_len - 1
                input_img = image_batch[:, n_full, :, :, :]
                if self.seq_length == 1:
                    real_img = image_batch[:, (cfg.train.step - 1), :, :, :]
                    gen_loss, disc_loss, gen_images = self.train_step(real_img, input_img, epoch)
                    gen_losses.append(gen_loss.numpy())
                    disc_losses.append(disc_loss.numpy())
                else:
                    real_img = []
                    for i in range(self.seq_length):
                        real_img_i = image_batch[:, i, :, :, :]
                        real_img.append(real_img_i)
                    gen_loss, disc_loss, gen_image = self.train_step(real_img, input_img, epoch)
                    # gen1_losses.append(gen_loss[0].numpy())
                    gen2_losses.append(gen_loss[0].numpy())
                    gen3_losses.append(gen_loss[1].numpy())
                    gen4_losses.append(gen_loss[2].numpy())
                    # disc1_losses.append(disc_loss[0].numpy())
                    disc2_losses.append(disc_loss[0].numpy())
                    disc3_losses.append(disc_loss[1].numpy())
                    disc4_losses.append(disc_loss[2].numpy())
                    gen_images = tf.concat([gen_image[0], gen_image[1], gen_image[2]], 1)

                if (epoch + 1) % cfg.train.image_save_step == 0:
                    utils.save_images((epoch + 1), gen_images, image_batch, train_save_path, batch_cnt)
                batch_cnt = batch_cnt + 1

            if self.seq_length == 1:
                avg_gen_loss = np.mean(gen_losses)
                avg_disc_loss = np.mean(disc_losses)
                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                print('avg_gen_loss:', avg_gen_loss)
                print('avg_disc_loss:', avg_disc_loss)
            else:
                # avg_gen1_loss = np.mean(gen1_losses)
                avg_gen2_loss = np.mean(gen2_losses)
                avg_gen3_loss = np.mean(gen3_losses)
                avg_gen4_loss = np.mean(gen4_losses)
                # avg_disc1_loss = np.mean(disc1_losses)
                avg_disc2_loss = np.mean(disc2_losses)
                avg_disc3_loss = np.mean(disc3_losses)
                avg_disc4_loss = np.mean(disc4_losses)
                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # print('avg_gen1_loss:', avg_gen1_loss)
                print('avg_gen2_loss:', avg_gen2_loss)
                print('avg_gen3_loss:', avg_gen3_loss)
                print('avg_gen4_loss:', avg_gen4_loss)
                # print('avg_disc1_loss:', avg_disc1_loss)
                print('avg_disc2_loss:', avg_disc2_loss)
                print('avg_disc3_loss:', avg_disc3_loss)
                print('avg_disc4_loss:', avg_disc4_loss)

            if (epoch + 1) % cfg.train.checkpoint_save_step == 0:
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                if not os.path.exists(self.checkpoint_prefix):
                    os.makedirs(self.checkpoint_prefix)
                self.manager.save()
                self.gen_lr = 0.98 * self.gen_lr
                self.dis_lr = 0.98 * self.dis_lr
        # if self.pretrain == None:
        #     pre_path = './' + self.checkpoint_prefix
        #     return pre_path

    def test(self, path, save_pathes):
        self.checkpoint.restore(tf.train.latest_checkpoint(cfg.path.checkpoint_restore1))
        if not os.path.exists(save_pathes):
            os.makedirs(save_pathes)

        test_img = utils.load_image(path, cfg.data.image_size, cfg.data.image_channel)
        test_img = (test_img - 127.5) / 127.5
        test_img1 = tf.reshape(test_img, [-1, cfg.data.image_size, cfg.data.image_size, cfg.data.image_channel])
        # hidden_encoders, features = self.seq_encoder(test_img1)
        if cfg.train.seq_length == 1:
            if cfg.train.step == 1:
                hidden_encoders, features = self.seq_encoder(test_img1)
                gen_images = self.generator1(hidden_encoders, features[0])
            elif cfg.train.step == 2:
                hidden_encoders, features = self.seq_encoder(test_img1)
                gen_images = self.generator2(hidden_encoders, features[1])
            elif cfg.train.step == 3:
                hidden_encoders, features = self.seq_encoder(test_img1)
                gen_images = self.generator3(hidden_encoders, features[2])
            else:
                hidden_encoders, features = self.seq_encoder(test_img1)
                gen_images = self.generator4(hidden_encoders, features[3])
        else:
            hidden_encoders2, feature2 = self.seq_encoder2(test_img1)
            hidden_encoders3, feature3 = self.seq_encoder3(test_img1)
            hidden_encoders4, feature4 = self.seq_encoder4(test_img1)
            gen2_image = self.generator2(hidden_encoders2, feature2[1])
            gen3_image = self.generator3(hidden_encoders3, feature3[2])
            gen4_image = self.generator4(hidden_encoders4, feature4[3])
            gen_images = tf.concat([gen2_image, gen3_image, gen4_image], 1)

        return gen_images




