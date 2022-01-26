import os
import tensorflow as tf
from tensorflow.keras import *
import numpy as np
import sys, os, shutil
from workspace._global.config import *
from workspace._global.funcs import *
from workspace._global.bbx import *
from workspace._global.recognition import *
sys.path.extend([RESEARCH])
from models.research.object_detection.utils import config_util
from models.research.object_detection.builders import model_builder
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
import urllib.request
import cv2
import shutil
import tensorflowjs as tfjs
ipython = get_ipython()

def prepare_and_get_dataset(img_root_path, label_name, label_num):
    t = os.path.join(INSTALLATION_PATH, "TEMP_IMGS")
    if os.path.exists(t):
        shutil.rmtree(t)

    #adjust images to shape

    # shutil.copytree(img_root_path, t)
    # for file in os.listdir(t):
    #     tpath = os.path.join(t, file)
    #     face = cv2.imread(tpath)
    #     face = cv2.resize(face, (RECOG_IMAGE_SIZES[1], RECOG_IMAGE_SIZES[0]))
    #     cv2.imwrite(tpath, face)

    temp_tfrecord_name = os.path.join(FACE_RECOG_DATA, f"{label_num}.tfrecord")
    parse_images(temp_tfrecord_name, img_root_path, label_name, label_num, label_num)

    records = [os.path.join(FACE_RECOG_DATA, x) for x in os.listdir(FACE_RECOG_DATA)]
    #https://www.google.com/search?q=AttributeError%3A+%27Tensor%27+object+has+no+attribute+%27numpy%27+inside+parse+function&rlz=1C1LRNT_deAT968AT968&biw=1536&bih=750&sxsrf=AOaemvL50QYnbrYU291P-ngiovGkYzM9Ng%3A1642804617638&ei=iTXrYa_BJsCGxc8P8Z-4iAk&ved=0ahUKEwjvqa2N9MP1AhVAQ_EDHfEPDpEQ4dUDCA4&uact=5&oq=AttributeError%3A+%27Tensor%27+object+has+no+attribute+%27numpy%27+inside+parse+function&gs_lcp=Cgdnd3Mtd2l6EAM6BAghEApKBAhBGAFKBAhGGABQoQhYxxhggBtoAXAAeACAAWqIAYoKkgEEMTQuMZgBAKABAcABAQ&sclient=gws-wiz
    dataset = tf.data.TFRecordDataset(records)
    dataset = dataset.map(parse_fn)
    dataset = dataset.batch(FACE_RECOG_BATCH_SIZE)


    return t, dataset



def new_face_added(face_name, img_root_path, override_existing=True):
    model = models.load_model(OUTPUT_FACES_RECOG)

    curr_classes = model.layers[:].pop().output_shape[-1]

    new_model = Sequential()
    for layer in model.layers[:-1]:
        new_model.add(layer)

    prev_output_layer = model.layers.pop()
    prev_output = prev_output_layer.get_weights()
    
    prev_weights = prev_output[0]
    prev_bias = prev_weights[1]
    
    curr_classes += 1


    new_weights = np.zeros(shape=(len(prev_weights), curr_classes))
    for i in range(len(prev_weights)):
        for j in range(curr_classes-1):
            new_weights[i][j] = prev_weights[i][j]
        new_weights[i][curr_classes-1] = 0.

    new_bias = np.zeros(shape=(curr_classes))
    for i in range(list(prev_bias.shape)[-1]):
        new_bias[i] = prev_bias[i] 
    new_bias[curr_classes-1] = 0.
    
    
    # new_weights = initializers.Constant(new_weights)
    # xxxx = new_weights.shape
    # new_bias = initializers.Constant(new_bias)
    # yyyy = new_bias.shape

    new_output_layer = layers.Dense(curr_classes, 
                                    activation='softmax', name="output_layer")
    new_model.add(new_output_layer)
    new_output_layer.set_weights([new_weights, new_bias])


    extend_label_map(curr_classes, face_name)

    path, dataset = prepare_and_get_dataset(img_root_path, face_name, curr_classes)
    new_model.compile(optimizer=model.optimizer, loss=losses.CategoricalCrossentropy(from_logits=False), 
                      metrics='accuracy', run_eagerly=True) 

    new_model.fit(dataset, epochs=6)

    shutil.rmtree(path)


    if override_existing:
        os.remove(OUTPUT_FACES_RECOG)
        models.save_model(new_model, OUTPUT_FACES_RECOG)
        shutil.rmtree(FACES_RECOG_MODEL_EXPORT)
        os.mkdir(FACES_RECOG_MODEL_EXPORT)
        tfjs.converters.save_keras_model(model, FACES_RECOG_MODEL_EXPORT)


    return new_model, curr_classes



def detect_face(model, categories, path):
    face = cv2.imread(path)
    face = cv2.resize(face, (RECOG_IMAGE_SIZES[1], RECOG_IMAGE_SIZES[0]))
    face = tf.convert_to_tensor(face, tf.uint8)
    face = tf.expand_dims(face, 0)
    pred = model.predict(face)
    pred_vals = pred[0]
    max_index = 0
    for i in range(len(pred_vals)):
        if pred_vals[i] > pred_vals[max_index]:
            max_index = i
 

    label =  categories[i].name if pred_vals[max_index] > 0.65  else 'unknown'
    return label, pred_vals[max_index]

# def new_face_added(model, face_name, img_root_path, override_existing=True):
#     curr_classes = model.layers[:].pop().output_shape[-1]
#     last_dense_before_output_neurons = model.layers[:][len(model.layers)-2]
#     xxxx = last_dense_before_output_neurons.get_weights()

#     new_model = Sequential()
#     for layer in model.layers[:-1]: # go through until last layer
#         new_model.add(layer)

#     prev_output_layer = model.layers.pop()
#     prev_output = prev_output_layer.get_weights()
    
#     prev_weights = prev_output[0]
#     prev_bias = prev_weights[1]
    
#     curr_classes += 1


#     new_weights = np.zeros(shape=(len(prev_weights), curr_classes))
#     for i in range(len(prev_weights)):
#         for j in range(curr_classes-1):
#             new_weights[i][j] = prev_weights[i][j]
#         new_weights[i][curr_classes-1] = 0.

#     new_bias = np.zeros(shape=(curr_classes))
#     for i in range(list(prev_bias.shape)[-1]):
#         new_bias[i] = prev_bias[i] 
#     new_bias[curr_classes-1] = 0.
    
#     # prev_weights = np.reshape(prev_output, (sum))
#     # prev_weights = np.array(prev_weights.tolist())
#     # prev_weights = np.array(prev_weights.tolist().append(bias))
#     # for _ in range(sum):
#     #     prev_weights = np.insert(prev_weights, 0, 0.0, 0)
    
#     new_weights = tf.constant_initializer(new_weights)
#     new_bias = tf.constant_initializer(new_bias)

#     new_output_layer = layers.Dense(curr_classes, 
#                                     activation='softmax', 
#                                     kernel_initializer=new_weights,
#                                     bias_initializer=new_bias)

#     new_model.add(new_output_layer)


#     path, dataset = prepare_and_get_dataset(img_root_path, face_name, curr_classes-1)
#     new_model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(from_logits=False), 
#                       metrics='accuracy', run_eagerly=True) 

#     new_model.fit(dataset, epochs=3)

#     shutil.rmtree(path)

#     extend_label_map(curr_classes, face_name)

#     if override_existing:
#         models.save_model(new_model, FACES_RECOG)

#     return new_model

    

# def new_face_added(model, face_name, img_root_path, override_existing=True):
#     curr_classes = model.layers[:].pop().output_shape[-1]
#     last_dense_before_output_neurons = model.layers[:][len(model.layers)-1]
    
#     prev_output_layer = model.layers.pop()
#     prev_weights = prev_output_layer.get_weights()[0]
#     bias = prev_weights[1]
#     sum = 1
#     for shape in prev_weights.shape:
#         sum *= shape
#     prev_weights = np.reshape(prev_weights, (sum))
#     prev_weights = np.array(prev_weights.tolist().append(bias))
#     curr_classes += 1
#     for _ in range(last_dense_before_output_neurons):
#         prev_weights = np.insert(prev_weights, 0, 0.0, 0)
    
#     new_init_weights = tf.constant_initializer(prev_weights)
#     new_output_layer = layers.Dense(curr_classes, activation='softmax', kernel_initializer=new_init_weights) 
#     model.add(new_output_layer)

#     dataset = prepare_and_get_dataset(img_root_path, face_name, curr_classes)
#     model.fit(dataset, epochs=3)

#     extend_label_map(curr_classes +1, face_name)

#     if override_existing:
#         models.save_model(model, FACES_RECOG)

#     return model
