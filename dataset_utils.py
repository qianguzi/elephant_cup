from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2, glob
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tfrecord(file_path, dataset_dir):
    ''' Covert Image dataset to tfrecord. '''
    writer = tf.python_io.TFRecordWriter(file_path)

    with open(dataset_dir+'name.txt', 'r+') as f:
      for name in f.readlines():
        name = name.strip('\n')
        img = cv2.imread(dataset_dir+name, 0)
        img = cv2.resize(img, (256,256))
        img = np.reshape(img, [-1])
        img = (img / 127.5 - 1).astype(np.float32)
        label = int(name[0])
        id = int(name[2:-4])

        example = tf.train.Example(features=tf.train.Features(feature={
            'data': _float_feature(img),
            'label': _int64_feature(label),
            'id': _int64_feature(id)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord(file_path, shuffle=True):
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

    return img, label, idx


def get_batch(file_path, batch_size, shuffle=True):
    '''Get batch.'''

    img, label, idx = read_tfrecord(file_path, shuffle)
    capacity = 3 * batch_size

    img_batch, label_batch, idx_batch = tf.train.batch([img, label, idx], batch_size,
                                                        capacity=capacity, num_threads=4)
    return img_batch, label_batch, idx_batch


if __name__ == '__main__':
    file_path = '../datasets/train/*.tfrecord'
    create_path = '../datasets/train/train.tfrecord'
    dataset_dir = '/home/myfile/dl_chrome/flower_photos/'
    #create_tfrecord(create_path, dataset_dir)
    img, label, idx = read_tfrecord(create_path)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                imgs, labels, idxs = sess.run([img, label, idx])
                imgs = ((imgs + 1) * 127.5).astype(int)
                cv2.imwrite('../datasets/'+str(labels)+'/'+str(idxs) + '.jpg', imgs)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            coord.join(threads)