import os
INSTALLATION_PATH = 'C:\\xampp\\htdocs\\projects\\Face_Recognition_Ullsperger\\LogicAIFaceRecProject'
RESEARCH = os.path.join(INSTALLATION_PATH, "models\\research")
OBJECT_DETECTION = os.path.join(RESEARCH, "object_detection")

PYTHON_3_8_10_PATH = 'C:\\Other\\Python\\Python_3_8_10\\python.exe'
WORKSPACE_PATH = os.path.join(INSTALLATION_PATH, 'workspace')
DATA_PATH = os.path.join(WORKSPACE_PATH, 'data')
PRETRAINED_MODELS_PATH = os.path.join(WORKSPACE_PATH, 'pretrained_models') 
MODELS_PATH = os.path.join(WORKSPACE_PATH, 'models') 

TRAIN_LABELS = os.path.join(DATA_PATH, "labels\\wider_face_train_bbx_gt.txt")
TRAIN_IMAGES = os.path.join(DATA_PATH, "train\\images")

# train_1204_1336.tfrecord"
OUTPUT_TRAIN_TFRECORD = os.path.join(DATA_PATH, "train_1209_1207.tfrecord")

VALIDATION_LABELS = os.path.join(DATA_PATH, "labels\\wider_face_val_bbx_gt.txt")
VALIDATION_IMAGES = os.path.join(DATA_PATH, "validation\\images")

# validation_1204_1336.tfrecord
OUTPUT_VALIDATION_TFRECORD = os.path.join(DATA_PATH, "validation_1209_1207.tfrecord")

TEST_IMAGES = os.path.join(DATA_PATH, "test\\images")

RAW_MODEL = os.path.join(MODELS_PATH, "raw_model.h5")
OUTPUT_MODEL = os.path.join(MODELS_PATH, "face_recog_20211204_1323.h5")

PRETRAINED_MODEL_DOWNLOAD = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
PRETRAINED_MODEL = os.path.join(PRETRAINED_MODELS_PATH, PRETRAINED_MODEL_NAME)

MODEL = os.path.join(MODELS_PATH, PRETRAINED_MODEL_NAME, "v5")
PIPELINE_CONFIG_PATH = os.path.join(MODEL, "pipeline.config")

LABEL_MAP_PATH = os.path.join(DATA_PATH, "labels\\label_map.pbtxt")

IMAGE_SIZES = [640, 640]
BATCH_SIZE = 32

TRAINING_SCRIPT = os.path.join(WORKSPACE_PATH, "model_main_tf2.py")


FACES_RECOG = os.path.join(WORKSPACE_PATH, "recognition")
FACES_RECOG_IMG = os.path.join(WORKSPACE_PATH, "recognition\\img")
FACE_RECOG_BATCH_SIZE = 4
FACES_RECOG_TEMP_TFRECORD = os.path.join(DATA_PATH, "temp_1210_2323.tfrecord") 
FACES_RECOG_MODEL = os.path.join(MODELS_PATH, "recognition")
FACES_RECOG_MODEL_EXPORT = os.path.join(FACES_RECOG_MODEL, "export")
FACE_RECOG_DATA = os.path.join(DATA_PATH, "recognition")

OUTPUT_FACES_RECOG = os.path.join(FACES_RECOG_MODEL, "face_recog_0121_2323.h5")
FACES_RECOG_LABEL_MAP = os.path.join(DATA_PATH, "labels\\face_recog_label_map.pbtxt")

PHP_DATA_PATH = "C:\\xampp\\htdocs\\projects\\LogicAIFaceRecProject\\backend\\api\\v1\\data"



#height, width
RECOG_IMAGE_SIZES = [250, 200]


def find_max_bbxs(filename):
    with open(filename) as f:
        lines = f.readlines()
        curr_max = 0
        curr_bounding_boxes_count = 0
        for line in lines:
                line = line.replace("\n", "")
                if len(line) < 4:
                    continue
                
                if "/" in line:
                    curr_max = max(curr_max, curr_bounding_boxes_count)
                    curr_bounding_boxes_count = 0
                    continue

                curr_bounding_boxes_count += 1
    return curr_max

MAX_BBXS = find_max_bbxs(TRAIN_LABELS)