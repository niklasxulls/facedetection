import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import *
import numpy as np
import sys, os
from workspace._global.config import *
from workspace._global.funcs import *
from workspace._global.bbx import *
sys.path.extend([RESEARCH])
from models.research.object_detection.utils import config_util
from models.research.object_detection.builders import model_builder
from IPython import get_ipython
import cv2 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.config.run_functions_eagerly(True)

ipython = get_ipython()
MAX_CLASSES  = 2

FACERECOG_FEATURE_DESCRIPTION = {
        'image/filename': tf.io.VarLenFeature(dtype=tf.string),
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
        'image/label':tf.io.VarLenFeature(dtype=tf.float32),
        'image/labelname':tf.io.VarLenFeature(dtype=tf.string),
}

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_feature(img_bytes, relative_path, label_name, label_num, max_classes):
    values = [1. if i == label_num else 0. for i in range(max_classes)]
    return tf.train.Example(
        features = tf.train.Features(feature = {
            'image/filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(relative_path, 'utf-8') ])),
            'image/encoded': _bytes_feature(img_bytes),
            'image/label':tf.train.Feature(float_list = tf.train.FloatList(value = values)),
            'image/labelname':tf.train.Feature(bytes_list = tf.train.BytesList(value = [bytes(label_name, 'utf-8') ])),
        })
    )

def parse_fn(proto):
    ex = tf.io.parse_single_example(proto, FACERECOG_FEATURE_DESCRIPTION)
    image_raw = tf.io.decode_jpeg(ex['image/encoded'], channels=3)
    # image = tf.cast(image, tf.uint8)
    # image = tf.cast(image / 255., tf.float32)
    image = tf.cast(image_raw , tf.float32) * (1. / 255)
    # index =  tf.cast(ex['image/label'], tf.uint8)[0]
    # label_index = tf.cast(ex['image/label']
    # labels = [1. if i == label_index else 0. for i in range(label_index+1)]
    labels = ex['image/label']
    if isinstance(labels, tf.SparseTensor):
       labels = tf.sparse.to_dense(labels)

    return image, labels

def parse_fn(proto, max_classes):
    ex = tf.io.parse_single_example(proto, FACERECOG_FEATURE_DESCRIPTION)
    image_raw = tf.io.decode_jpeg(ex['image/encoded'], channels=3)
    # image = tf.cast(image, tf.uint8)
    # image = tf.cast(image / 255., tf.float32)
    image = tf.cast(image_raw , tf.float32) * (1. / 255)
    # index =  tf.cast(ex['image/label'], tf.uint8)[0]
    label_index = tf.cast(ex['image/label'], tf.uint8)[0]
    labels = [1. if i == label_index else 0. for i in range(label_index+1)]
    labels = ex['image/label']
    if isinstance(labels, tf.SparseTensor):
       labels = tf.sparse.to_dense(labels)

    return image, tf.convert_to_tensor(labels)

def parse_images(output_tfrecord_name, img_dir_path, label_name, label_num, max_classes):
    if not max_classes:
        max_classes = label_num

    with tf.io.TFRecordWriter(output_tfrecord_name) as writer:
        for file in os.listdir(img_dir_path):
            if ".tfrecord" in file:
                continue
            with tf.io.gfile.GFile(os.path.join(img_dir_path, file), 'rb') as fid:
                img_bytes = fid.read()

                writer.write(
                    tf.train.Example.SerializeToString(create_feature(img_bytes, file, label_name, label_num, max_classes))
                )

def extend_label_map(label_num, label_name): 
    MAX_CLASSES = label_num+1
    with open(FACES_RECOG_LABEL_MAP, 'a') as the_file:
            the_file.write('item\n')
            the_file.write('{\n\t')
            the_file.write('id :{}'.format(str(label_num)))
            the_file.write('\n\t')
            the_file.write("name :'{0}'".format(str(label_name)))
            the_file.write('\n')
            the_file.write('}\n')