# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Build and train mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import inception_v4
import dataset_utils

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_integer('num_classes', 6, 'Number of classes to distinguish')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 299, 'Input image resolution')
flags.DEFINE_string('fine_tune_checkpoint', '../inception_v4_2016_09_09/inception_v4.ckpt',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('checkpoint_dir', '../checkpoints',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('dataset_dir', '../datasets/train/train.tfrecord', 'Location of dataset')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100,
                     'How often to save checkpoints, secs')

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94


def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    # If we are fine tuning a checkpoint we need to start at a lower learning
    # rate since we are farther along on training.
    return 1e-4
  else:
    return 0.045


def build_model():
  """Builds graph for model to train with rewrites for quantization.

  Returns:
    g: Graph with fake quantization ops and batch norm folding suitable for
    training quantized weights.
    train_tensor: Train op for execution during training.
  """
  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):
    inputs, labels, _ = dataset_utils.get_batch(FLAGS.dataset_dir, FLAGS.batch_size)
    inputs = tf.image.resize_images(inputs, [FLAGS.image_size, FLAGS.image_size])
    
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
      logits, _ = inception_v4.inception_v4(
          inputs,
          is_training=True,
          num_classes=FLAGS.num_classes)

    labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
    tf.losses.softmax_cross_entropy(labels, logits)

    total_loss = tf.losses.get_total_loss(name='total_loss')
    # Configure the learning rate using an exponential decay.
    num_epochs_per_decay = 2.5
    imagenet_size = 7340
    decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(
        get_learning_rate(),
        tf.train.get_or_create_global_step(),
        decay_steps,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, epsilon=1.0)

    train_tensor = slim.learning.create_train_op(
        total_loss,
        optimizer=opt)

  slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
  slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
  return g, train_tensor


def get_checkpoint_init_fn():
  """Returns the checkpoint init_fn if the checkpoint is provided."""
  if FLAGS.fine_tune_checkpoint:
    variables_to_restore = slim.get_variables_to_restore(exclude=['InceptionV4/Logits','InceptionV4/AuxLogits'])
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    # When restoring from a floating point model, the min/max values for
    # quantized weights and activations are not present.
    # We instruct slim to ignore variables that are missing during restoration
    # by setting ignore_missing_vars=True
    slim_init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.fine_tune_checkpoint,
        variables_to_restore,
        ignore_missing_vars=True)

    def init_fn(sess):
      slim_init_fn(sess)
      # If we are restoring from a floating point model, we need to initialize
      # the global step to zero for the exponential decay to result in
      # reasonable learning rates.
      sess.run(global_step_reset)
    return init_fn
  else:
    return None


def train_model():
  """Trains mobilenet_v1."""
  g, train_tensor = build_model()
  #tf.train.write_graph(g, FLAGS.checkpoint_dir, 'graph.pbtxt')

  with g.as_default():
    slim.learning.train(
        train_tensor,
        FLAGS.checkpoint_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(),
        global_step=tf.train.get_global_step())


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)
