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

"""Routine for decoding the BBBC006 binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import pandas as pd

# Process images of this size.
IMAGE_WIDTH = 696
IMAGE_HEIGHT = 520

# Global constants describing the BBBC006 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 692
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 76


def read_from_queue(path):
    return tf.image.decode_png(path, channels=1, dtype=tf.uint8)


def read_bbbc006(all_files_queue):
    """Reads and parses examples from BBBC006 data files.
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.
    Returns:
        An object representing a single example, with the following fields:
            label: a [height, width, 2] uint8 Tensor with contours tensor in depth 0 and
                segments tensor in depth 1.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class BBBC006Record(object):
        pass

    result = BBBC006Record()

    # Read a record, getting filenames from the filename_queue.
    text_reader = tf.TextLineReader()
    _, csv_content = text_reader.read(all_files_queue)

    i_path, c_path, s_path = tf.decode_csv(csv_content,
                                           record_defaults=[[""], [""], [""]])

    result.uint8image = read_from_queue(tf.read_file(i_path))
    contour = read_from_queue(tf.read_file(c_path))
    segment = read_from_queue(tf.read_file(s_path))

    result.label = tf.concat([contour, segment], 2)
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
        image: 3-D Tensor of [height, width, 1] of type.float32.
        label: 3-D Tensor of [height, width, 1] of type.int32.
        min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 1] size.
        labels: Labels. 4D tensor of [batch_size, height, width, 2] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, labels


def gen_csv_paths(data_dir, pref):
    """
    Generate CSV file from image, contour, and segment file paths.
    Args:
        data_dir: BBBC006 data directory path.
        pref: Prefix (either 'train' or 'test')
    """
    filenames = get_png_files(os.path.join(data_dir, 'BBBC006_v1_' + pref))
    contours = get_png_files(os.path.join(data_dir, 'BBBC006_v1_contours_'
                                                     + pref))
    segments = get_png_files(os.path.join(data_dir, 'BBBC006_v1_segments_'
                                                     + pref))

    all_files = [filenames, contours, segments]
    pd_arr = pd.DataFrame(all_files).transpose()
    pd_arr.to_csv(pref + '.csv', index=False, header=False)


def get_read_input(eval_data):
    """
    Fetch input data row by row from CSV files.
    Args:
        eval_data: String representing whether to read from 'train' or 'test' directories.
    Returns:
        read_input: An object representing a single example.
        reshaped_image: Image of type tf.float32, reshaped to correct dimensions.
    """
    # Create queues that produce the filenames and labels to read.
    all_files_queue = tf.train.string_input_producer([eval_data + '.csv'])

    # Read examples from files in the filename queue.
    read_input = read_bbbc006(all_files_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    read_input.label = tf.cast(read_input.label, tf.int32)

    return read_input, reshaped_image


def get_png_files(dirname):
    return [dirname + '/' + f for f in os.listdir(dirname) if f.endswith('.png')]


def inputs(eval_data, batch_size):
    """Construct input for BBBC006 evaluation using the Reader ops.
    Args:
        eval_data: String representing whether to read from 'train' or 'test' directories.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        labels: Labels. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 2] size.
    """
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    read_input, reshaped_image = get_read_input(eval_data)

    # Image processing for evaluation.

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    read_input.label.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 2])

    # Set max intensity to 1
    read_input.label = tf.cast(tf.divide(read_input.label, 255), tf.int32)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    shuffle = False if eval_data == 'test' else True
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)
