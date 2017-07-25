# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the BBBC006 network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import bbbc006_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Path to the BBBC006 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the BBBC006 data set.
IMAGE_WIDTH = bbbc006_input.IMAGE_WIDTH
IMAGE_HEIGHT = bbbc006_input.IMAGE_HEIGHT
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = bbbc006_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = bbbc006_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 72.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
DISCOUNT_WEIGHT = 0.1  # Weight for auxiliary classifier loss.
DROPOUT_RATE = 0.5  # Probability for dropout layers.

# Constants for the model architecture.
NUM_LAYERS = 6
FEAT_ROOT = 32
NUM_CLASSES = 2

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def distorted_inputs():
    """Construct distorted input for BBBC006 training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels = bbbc006_input.distorted_inputs(batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for BBBC006 evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 1] size.
      labels: Labels. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 2] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels = bbbc006_input.inputs(eval_data=eval_data,
                                          batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def conv_layer(in_layer, features, scope, training, layer):
    conv = tf.layers.conv2d(in_layer, features, (3, 3), padding='same',
                            kernel_initializer=tf.contrib.layers.
                            xavier_initializer_conv2d(uniform=False),
                            bias_initializer=tf.contrib.layers.
                            xavier_initializer_conv2d(uniform=False),
                            kernel_regularizer=tf.nn.l2_loss,
                            name=scope.name)
    conv = tf.nn.relu(tf.layers.batch_normalization(conv, training=training))
    if layer > 3:  # Add dropout to layers 5 and 6
        conv = tf.nn.dropout(conv, keep_prob=DROPOUT_RATE)
    return conv


def deconv_layer(in_layer, dc, ds, scope, training):
    shape = [dc * 2, dc * 2, 2, in_layer.get_shape().as_list()[-1]]
    weight = tf.Variable(tf.contrib.layers.
                         xavier_initializer_conv2d(uniform=False)
                         (shape=shape))
    bias = tf.Variable(tf.contrib.layers.
                       xavier_initializer_conv2d(uniform=False)
                       (shape=[NUM_CLASSES]))
    deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(
        in_layer, weight, ds, strides=[1, dc, dc, 1], padding='SAME'),
        bias, name=scope.name)
    return tf.nn.relu(tf.layers.batch_normalization(deconv, training=training))


def inference(images, training=True):
    """Build the BBBC006 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      output_maps: List of output_map 4D tensors of [batch_size, 696, 520, 1]
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    features = FEAT_ROOT
    in_layer = images

    dc = 8  # Deconvolution constant: kernel size = 2 * dc, stride = dc
    ds = [FLAGS.batch_size, bbbc006_input.IMAGE_HEIGHT, bbbc006_input.IMAGE_WIDTH,
          NUM_CLASSES]  # Shape of deconvolution output

    # Up-sampled layers 4-6 output maps for contours and segments, respectively
    c_outputs = []
    s_outputs = []

    for layer in range(NUM_LAYERS):
        # CONVOLUTION
        with tf.variable_scope('conv{}'.format(layer + 1)) as scope:
            # Double the number of features for all but convolution layer 4
            features *= 2 if layer != 4 else 1

            conv = conv_layer(in_layer, features, scope, training, layer)
            _activation_summary(conv)

        # POOLING
        if 0 < layer:  # Convolution layer 0 has no max pooling afterwards
            pool = tf.layers.max_pooling2d(conv, (2, 2), (2, 2), padding='same',
                                           name='pool{}'.format(layer + 1))
            in_layer = pool
            _activation_summary(pool)
        else:
            in_layer = conv

        # Transposed convolution and output mapping for segments and contours
        if layer > 2:  # Only applies to layers 3-5
            for i in range(2):
                # TRANSPOSED CONVOLUTION
                with tf.variable_scope('deconv{0}_{1}'.format(layer + 1, i)) as scope:
                    deconv = deconv_layer(in_layer, dc, ds, scope, training)
                    _activation_summary(deconv)

                # OUTPUT
                with tf.variable_scope('output{0}_{1}'.format(layer + 1, i)) as scope:
                    output = tf.layers.conv2d(deconv, 2, (1, 1), activation=tf.nn.relu,
                                              padding='same', name=scope.name)

                    if i == 0:
                        c_outputs.append(output)
                    else:
                        s_outputs.append(output)
            dc *= 2
    return c_outputs, s_outputs


def add_cross_entropy(labels, logits, pref, layer):
    flat_labels = tf.reshape(labels, [-1])
    flat_logits = tf.reshape(logits, [-1, 2])

    with tf.variable_scope('{0}_cross_entropy{1}'.format(pref, layer)) as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=flat_labels, logits=flat_logits,
            name='{0}_cross_entropy_per_example{1}'.format(pref, layer))
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name=scope.name)
        if layer < 2:
            cross_entropy_mean *= DISCOUNT_WEIGHT
        tf.add_to_collection('losses', cross_entropy_mean)


def loss(c_outputs, s_outputs, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      c_fuse: Contours output map from inference().
      s_fuse: Segments output map from inference().
      labels: Labels from distorted_inputs or inputs()

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.

    # Split the labels tensor into contours and segments image tensors
    # Each has shape [FLAGS.batch_size, 696, 520, 1]
    contours_labels, segments_labels = tf.split(labels, 2, 3)

    for pref in ['c', 's']:
        for layer in range(3):
            labels = contours_labels if pref == 'c' else segments_labels
            logits = c_outputs[layer] if pref == 'c' else s_outputs[layer]
            add_cross_entropy(labels, logits, pref, layer)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in BBBC006 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train BBBC006 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.2)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def get_dice_coef(logits, labels, smooth=1e-5):
    inter = tf.reduce_sum(tf.multiply(logits, labels))
    return tf.reduce_mean((2 * inter + smooth) /
                          (tf.reduce_sum(logits) +
                           tf.reduce_sum(labels) + smooth))


def dice_op(c_fuse, s_fuse, labels, threshold=0.5):
    _, c_logits = tf.split(tf.nn.softmax(c_fuse), 2, 3)
    _, s_logits = tf.split(tf.nn.softmax(s_fuse), 2, 3)

    tf.summary.image('c_logits', c_logits)
    tf.summary.image('s_logits', s_logits)

    c_logits = tf.cast(c_logits > threshold, tf.float32)
    s_logits = tf.cast(s_logits > threshold, tf.float32)

    labels = tf.cast(labels, tf.float32)
    c_labels, s_labels = tf.split(labels, 2, 3)

    tf.summary.image('c_labels', c_labels)
    tf.summary.image('s_labels', s_labels)

    c_dice = get_dice_coef(c_logits, c_labels)
    s_dice = get_dice_coef(s_logits, s_labels)

    return c_dice, s_dice
