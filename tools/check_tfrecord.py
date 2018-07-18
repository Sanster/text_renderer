import os
import numpy as np
import argparse

import tensorflow as tf
from tqdm import tqdm
import cv2


def main(args):
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer([args.tfrecords_file])

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'file_name': tf.FixedLenFeature([], tf.string)
                                       })

    tf_image = tf.cast(features['image'], tf.string)
    tf_label = tf.cast(features['label'], tf.string)
    tf_file_name = tf.cast(features['file_name'], tf.string)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        labels = []

        num = int(args.tfrecords_file.split('.')[0].split('_')[1])
        if args.num is not None and args.num <= num:
            num = args.num

        for i in tqdm(range(num)):
            img_str, label, file_name = sess.run([tf_image, tf_label, tf_file_name])
            file_name = file_name.decode("utf-8")
            label = label.decode("utf-8")
            labels.append(label)
            save_path = os.path.join(args.output_dir, file_name)

            nparr = np.fromstring(img_str, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(save_path, img_np)

        coord.request_stop()
        coord.join(threads)

        with open(args.label_file, 'w') as f:
            for l in labels:
                f.write('%s\n' % l)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_file', type=str, default='test_10000.tfrecords',
                        help='File name should contain image total count')
    parser.add_argument('--output_dir', type=str, help='Extract dir in this folder', default='./test')
    parser.add_argument('--num', type=int, default=None, help='If None, extract all images in tfrecords file')
    args = parser.parse_args()

    if not os.path.exists(args.tfrecords_file):
        parser.error("tfrecords_file not exist")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.label_file = os.path.join(args.output_dir, 'labels.txt')

    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main(parse_arguments())
