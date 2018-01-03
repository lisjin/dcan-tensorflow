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

"""A binary to train BBBC006 using a single GPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import logging
import tensorflow as tf

import bbbc006

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/bbbc006_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_data', 'train',
                           """Either 'test' or 'train'.""")
tf.app.flags.DEFINE_integer('max_steps', 40000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 20,
                            """How often to log results to the console.""")
tf.logging.set_verbosity(tf.logging.DEBUG)


def train():
    """Train BBBC006 for a number of steps."""
    with tf.Graph().as_default():
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        global_step_init = -1
        global_step = tf.contrib.framework.get_or_create_global_step()
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1]
                                   .split('-')[-1])

        # Get images and labels for BBBC006.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up
        # on GPU and resulting in a slow down.
        images, labels = bbbc006.inputs(eval_data=FLAGS.eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        c_fuse, s_fuse = bbbc006.inference(images)
        bbbc006.get_show_preds(c_fuse, s_fuse)

        # Calculate loss.
        loss = bbbc006.loss(c_fuse, s_fuse, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = bbbc006.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = global_step_init
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

            if ckpt:
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                logging.info("Model restored from file: %s" % ckpt.model_checkpoint_path)
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
