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
"""Validate mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

import inception_v4
import dataset_utils

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_integer('num_classes', 6, 'Number of classes to distinguish')
flags.DEFINE_integer('num_examples', 3500, 'Number of examples to evaluate')
flags.DEFINE_integer('image_size', 299, 'Input image resolution')
flags.DEFINE_string('checkpoint_dir', '../checkpoints/frozen_graph.pb', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', '../eval_logs', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '../datasets/val/*.tfrecord', 'Location of dataset')

FLAGS = flags.FLAGS


def metrics(logits, labels):
  """Specify the metrics for eval.

  Args:
    logits: Logits output from the graph.
    labels: Ground truth labels for inputs.

  Returns:
     Eval Op for the graph.
  """
  labels = tf.squeeze(labels)
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'Accuracy': tf.metrics.accuracy(predictions=tf.argmax(logits, 1), labels=labels),
      'Recall_5': tf.metrics.recall_at_k(labels, logits, 5),
  })
  for name, value in names_to_values.iteritems():
    slim.summaries.add_scalar_summary(
        value, name, prefix='eval', print_summary=True)
  return names_to_updates.values()


def build_model():
  """Build the mobilenet_v1 model for evaluation.

  Returns:
    g: graph with rewrites after insertion of quantization ops and batch norm
    folding.
    eval_ops: eval ops for inference.
    variables_to_restore: List of variables to restore from checkpoint.
  """
  g = tf.Graph()
  with g.as_default():
    inputs, labels, _ = dataset_utils.get_batch(FLAGS.dataset_dir, FLAGS.batch_size)
    inputs = tf.image.resize_images(inputs, [FLAGS.image_size, FLAGS.image_size])

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
      _, end_points = inception_v4.inception_v4(
          inputs,
          is_training=False,
          num_classes=FLAGS.num_classes)

    eval_ops = metrics(end_points['Predictions'], labels)

  return g, eval_ops


def eval_model():
  """Evaluates mobilenet_v1."""
  g, eval_ops = build_model()
  with g.as_default():
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))
    slim.evaluation.evaluate_once(
        FLAGS.master,
        FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_ops)


def main(unused_arg):
  eval_model()


if __name__ == '__main__':
  tf.app.run(main)
