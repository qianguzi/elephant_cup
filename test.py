from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim

import inception_v4
import dataset_utils


def read_tfrecord(file_path, batch_size, shuffle=True):
    reader = tf.TFRecordReader()

    file_path = glob.glob(file_path)
    filename_queue = tf.train.string_input_producer(
        file_path, shuffle=shuffle)

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'data': tf.FixedLenFeature([256, 256], tf.float32),
                  'label': tf.FixedLenFeature([], tf.int64),
                  'id': tf.FixedLenFeature([], tf.int64)})
    img = tf.expand_dims(features['data'], -1)
    img = tf.image.grayscale_to_rgb(img)
    label = features['label']
    idx = features['id']

    img_batch, label_batch, idx_batch = tf.train.batch(
      [img, label, idx], batch_size, capacity=3 * batch_size, num_threads=4)
    return img_batch, label_batch, idx_batch


def cnvert_ckpt_to_pb(ckpt_path, output_path):
  g = tf.Graph()
  output_node_names = ['Outputs']
  with g.as_default():
    inputs = tf.placeholder(tf.float32, [None, 256, 256, 3], 'Inputs')
    inputs = tf.image.resize_images(inputs, [299, 299])

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
      _, end_points = inception_v4.inception_v4(
          inputs,
          is_training=False,
          num_classes=6)
      outputs = tf.argmax(end_points['Predictions'], 1, name='Outputs')
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, ckpt_path)
      output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, g.as_graph_def(), output_node_names)
      with tf.gfile.GFile(output_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def model_test(tfrecord_path, pb_path='../checkpoints/frozen_graph.pb', batch_size=1):
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
      od_graph_def.ParseFromString(f.read())
      inputs, outputs = tf.import_graph_def(od_graph_def, return_elements=["Inputs:0", "Outputs:0"])
    init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    image_tensor, label_tensor, _ = read_tfrecord(tfrecord_path, batch_size, False)
    with tf.Session() as sess:
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      predictions = []
      labels = []
      for i in range(int(170/batch_size)):
        image, label = sess.run([image_tensor, label_tensor])
        labels = labels + list(label)
        prediction = sess.run(outputs, feed_dict={inputs: image})
        predictions = predictions + list(prediction)
      coord.request_stop()
      coord.join(threads)
  #print(predictions, labels)
  return predictions

def main():
  #ckpt_path = '../checkpoints/model.ckpt-71221'
  #output_path = '../checkpoints/frozen_graph.pb'
  #cnvert_ckpt_to_pb(ckpt_path, output_path)
  tfrecord_path = '../datasets/val/TFcodeX_8.tfrecord'
  label = model_test(tfrecord_path)

if __name__ == '__main__':
  main()