import tensorflow as tf
cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE, from_logits=True)
def generator_loss(fake_output, seq_len=None):
    loss = cross_entropy(tf.ones_like(fake_output),fake_output,)
    if seq_len is not None:
        if seq_len == 1:
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.reshape(loss, [-1, seq_len])
            loss = tf.reduce_mean(loss)
    else:
        loss = tf.reduce_mean(loss)

    return loss
def generator_hinge_loss(fake_output, seq_len=None):
    loss = fake_output
    if seq_len is not None:
        if seq_len == 1:
            loss = -tf.reduce_mean(loss)
        else:
            loss = tf.reshape(loss, [-1, seq_len])
            loss = -tf.reduce_mean(loss)
    else:
        loss = -tf.reduce_mean(loss)
    return loss

def discriminator_loss(real_output, fake_output,seq_len=None):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output,)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output,)

    if seq_len is not None:
        if seq_len == 1:
            total_loss = real_loss + fake_loss
            total_loss = tf.reduce_mean(total_loss)
        else:
            real_loss = tf.reshape(real_loss, [-1, seq_len])
            fake_loss = tf.reshape(fake_loss, [-1, seq_len])
            total_loss = real_loss + fake_loss
            total_loss = tf.reduce_mean(total_loss)
    else:
        total_loss = real_loss + fake_loss
        total_loss = tf.reduce_mean(total_loss)
    return total_loss
def discriminator_hinge_loss(real_output, fake_output, seq_len = None):
    real_loss = tf.nn.relu(1.0 - real_output)
    fake_loss = tf.nn.relu(1.0 + fake_output)
    if seq_len is not None:
        if seq_len == 1:
            total_loss = real_loss + fake_loss
            total_loss = tf.reduce_mean(total_loss)
        else:
            real_loss = tf.reshape(real_loss, [-1, seq_len])
            fake_loss = tf.reshape(fake_loss, [-1, seq_len])
            total_loss = real_loss + fake_loss
            total_loss = tf.reduce_mean(total_loss)
    else:
        real_loss = tf.reduce_mean(real_loss)
        fake_loss = tf.reduce_mean(fake_loss)
        total_loss = real_loss + fake_loss

    return total_loss

def generator_loss1(cross_entropy,fake_output):
    gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return tf.reduce_mean(gen_loss)

def discriminator_loss1(cross_entropy,real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return tf.reduce_mean(total_loss)

def mse(test_feature,predic_feature):
    a = test_feature
    b = predic_feature
    c = tf.square(a - b)

    return c

def discriminator_dl_loss(real_output, dl_output):
    dl_MSE = mse((real_output * 2 / 3), dl_output)

    dl_loss = tf.reduce_mean(dl_MSE)
    return dl_loss



