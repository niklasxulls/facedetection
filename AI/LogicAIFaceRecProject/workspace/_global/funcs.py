import os
import requests
import tarfile
import tensorflow as tf
import albumentations as A
import tensorflow_addons as tfa
from workspace._global.config import *
from models.research.object_detection.utils import config_util
from models.research.object_detection.protos import pipeline_pb2
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils
from models.research.object_detection.builders import model_builder
import numpy as np
from PIL import Image
import cv2

bbx_params = A.BboxParams('albumentations', min_area=1, min_visibility=-2, label_fields=None, check_each_transform=False)
transform_resize_w_bbxs = A.Compose([
    A.Resize(IMAGE_SIZES[0], IMAGE_SIZES[1]),
], bbox_params = bbx_params)
transform_resize = A.Compose([
    A.Resize(IMAGE_SIZES[0], IMAGE_SIZES[1]),
])

FEATURE_DESCRIPTION = {
    'filename': tf.io.VarLenFeature(tf.string),
    'shape' : tf.io.FixedLenFeature([3], tf.int64),
    'image' : tf.io.FixedLenFeature([], tf.string),
    'bbxs' : tf.io.VarLenFeature(tf.int64), 
    'faces' : tf.io.FixedLenFeature([1], tf.int64),
}
gl = tfa.losses.GIoULoss()

def extractTar(url, extract_path, output_name):
    os.chdir(INSTALLATION_PATH)

    downloaded = requests.get(url)
    open(output_name, 'wb').write(downloaded.content)

    tar = tarfile.open(output_name, 'r')
    for item in tar:
        tar.extract(item, extract_path)

    tar.close()
    os.remove(os.path.join(INSTALLATION_PATH, output_name))


def prepare_bbxs(bbxs, width, height):
    bbx_to_transform = []
    for bbx in bbxs:
        temp = bbx
        temp[2] += temp[0]
        temp[2] /= width
        
        temp[3] += temp[1]
        temp[3] /= height

        temp[0] /= width
        temp[1] /= height

        if temp[0] > 1 or temp[1] > 1 or temp[2] > 1 or temp[2] <= temp[0] or temp[3] > 1 or temp[3] <= temp[1]:
            continue
        
        temp2 = temp.copy()
        # temp2[0] = temp[1]
        # temp2[1] = temp[0]
        # temp2[2] = temp[3]
        # temp2[3] = temp[2]

        temp2.append('x')

        bbx_to_transform.append(temp2)
    return bbx_to_transform


def decode_img(img, bbxs = [], resize=False):
    image = tf.image.decode_jpeg(img , channels=3)

    height, width, channels = image.shape
    image = tf.cast(image, tf.float32)

    bbxs_return = []
    if resize:
        image = tf.image.resize(image, [IMAGE_SIZES[0], IMAGE_SIZES[1]])
        image = image.numpy()
        if(len(bbxs) > 0):
            bbx_to_transform = prepare_bbxs(bbxs, width, height)
            transformed = transform_resize_w_bbxs(image=image, bboxes=bbx_to_transform)
            bbxs_temp = transformed['bboxes']

            for bbx in bbxs_temp:
                bbxs_return.append([
                     round(bbx[0] * IMAGE_SIZES[0]),
                     round(bbx[1] * IMAGE_SIZES[1]),
                     round((bbx[2] - bbx[0]) * IMAGE_SIZES[0]),
                     round((bbx[3] - bbx[1]) * IMAGE_SIZES[1])
                ])
        
        else: 
            transformed = transform_resize(image=image)
            bbxs_return.append([])

        image = tf.convert_to_tensor(transformed['image'])
    else:
        return image, image.shape
    return image, bbxs_return, image.shape


def parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
  image, _ = decode_img(example['image'])
  image = tf.cast(image, tf.uint8)
  #   print(example['bbxs'])
  bbxs = tf.cast(example['bbxs'], tf.int64)

  if isinstance(bbxs, tf.SparseTensor):
      bbxs = tf.sparse.to_dense(bbxs)
      
  return image, bbxs



def prepare_dataset(tfrecord_paths):

    if not isinstance(tfrecord_paths, list):
        tfrecord_paths = [tfrecord_paths]

    total = 0

    for path in tfrecord_paths:
        for _ in tf.compat.v1.python_io.tf_record_iterator(path):
            total += 1

    dataset = tf.data.TFRecordDataset(tfrecord_paths)

    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(total)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset

def loss_fn_practical(y_true, y_pred):
        # now contains 32 lists (a batch) of bbxs -> shape is (32, 7876)
        # print(y_true)
        batch_size = y_pred.shape[0]
        bbx_true = y_true

        # now contains 32 lists (a batch) of bbxs here we have to double access [0] in order to get the entry itself 
        # -> shape is (32, 1, 1, 7876)
        bbx_pred = tf.reshape(y_pred, (batch_size, MAX_BBXS * 4))

        losses = []
        curr_true = []
        curr_pred = []

        unpacked_true = tf.unstack(bbx_true)
        unpacked_pred = tf.unstack(bbx_pred)
        for i in range(batch_size):
            curr_true = unpacked_true[i] 
            curr_pred = unpacked_pred[i]


            curr_true = tf.split(curr_true, MAX_BBXS)
            curr_pred = tf.split(curr_pred, MAX_BBXS)

            if len(curr_true) == 0:
                curr_true.append(tf.concat([0., 0.,0.,0.]))

            curr_loss = gl(curr_true, curr_pred)

            losses.append(curr_loss)

        return tf.math.reduce_mean(losses, axis=-1)


@tf.function
def detect_fn(input_tensor, model):
    image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections

def get_best_bbx(bbxs, scores):
    _max = -1
    index = 0
    for i in range(len(bbxs)):
        if scores[i] > _max:
            _max = scores[i]
            index = i
    return bbxs[index]

def predict_for_image(img_np, model, draw_bbxs=False):

    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)

    prediction = detect_fn(input_tensor, model)

    num_detections = int(prediction.pop('num_detections'))
    prediction = {key: value[0, :num_detections].numpy()
                for key, value in prediction.items()}


    label_map = label_map_util.load_labelmap(LABEL_MAP_PATH)

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)   

    prediction['detection_classes'] = prediction['detection_classes'].astype(np.int64) +1

    if not draw_bbxs:
        return prediction

    img_with_detections = img_np.copy()


    viz_utils.visualize_boxes_and_labels_on_image_array(img_with_detections,
                                                        prediction['detection_boxes'],
                                                        prediction['detection_classes'],
                                                        prediction['detection_scores'],
                                                        category_index,
                                                        min_score_thresh=.5,
                                                        agnostic_mode=False,
                                                        use_normalized_coordinates=True
                                                        )

    return img_with_detections


def predict_for_image_with_path(path, model):
    image_np = np.array(Image.open(path))
    
    return predict_for_image(image_np, model)


def find_and_extract_face(model, img, output_path_with_filename=None, with_best_bbx=False):
    shape = img.shape
    prediction = predict_for_image(np.array(img), model)

    best_bbx = get_best_bbx(prediction['detection_boxes'],prediction['detection_scores'])
    best_bbx[0]*= shape[0]
    best_bbx[1] *= shape[1]
    best_bbx[2] *= shape[0]
    best_bbx[3] *= shape[1]
    # face = img[int(shape[0] -best_bbx[2]):int(shape[0] -best_bbx[0]), int(best_bbx[1]):int(best_bbx[3])]

    face = img[int(best_bbx[0]):int(best_bbx[2] + best_bbx[0]), int(best_bbx[1]):int(best_bbx[3])]
    face = cv2.resize(face, (RECOG_IMAGE_SIZES[1], RECOG_IMAGE_SIZES[0]))
    if output_path_with_filename == None:
        return face if not with_best_bbx else (face, best_bbx)
    cv2.imwrite(output_path_with_filename,face)