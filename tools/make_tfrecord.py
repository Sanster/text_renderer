import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def read_image(path, size=None):
    img = cv2.imread(path, 0)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    return img


def read_image_raw(img_path):
    return open(img_path, 'rb').read()


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def load_labels(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    labels = [l[:-1].strip() for l in labels]
    return labels


def build_img_paths(img_dir, img_count):
    """
    Image name should be eight length with continue num. e.g. 00000000.jpg, 00000001.jpg
    """
    img_paths = []
    for i in range(img_count):
        base_name = "{:08d}".format(i)
        img_path = os.path.join(img_dir, base_name + ".jpg")
        img_paths.append(img_path)

    return img_paths


def main(args):
    labels = load_labels(args.label_file)
    paths = build_img_paths(args.img_dir, len(labels))

    data_name = 'image'
    label_name = 'label'

    with tf.python_io.TFRecordWriter(args.output_file) as writer:
        for i in tqdm(range(len(paths))):
            if args.raw:
                img = read_image_raw(paths[i])
            else:
                img = read_image(paths[i]).tostring()
            label = labels[i]

            # create a feature
            feature = {label_name: bytes_feature(label.encode(encoding='utf-8')),
                       data_name: bytes_feature(tf.compat.as_bytes(img))}

            # create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/home/cwq/ssd_data/more_bg_corpus/val',
                        help='Images to make tf_record file, should include labels.txt file')
    parser.add_argument('--output_file', type=str, default='test.tfrecords')
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument('-f', '--force', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.img_dir):
        parser.error("img_dir not exist")

    if not args.force and os.path.exists(args.output_file):
        parser.error('output_file already exists')

    args.label_file = os.path.join(args.img_dir, 'labels.txt')

    if not os.path.exists(args.label_file):
        parser.error('labels.txt not exist')

    return args


if __name__ == '__main__':
    main(parse_arguments())
