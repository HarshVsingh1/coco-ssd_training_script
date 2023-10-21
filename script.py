import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection import model_lib_v2
from object_detection.utils import dataset_util


def create_tf_example(image_path, bbox, label_map):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    filename = os.path.basename(image_path).encode('utf8')
    image_format = b'jpeg'
    xmins = [bbox[0] / width]
    ymins = [bbox[1] / height]
    xmaxs = [bbox[2] / width]
    ymaxs = [bbox[3] / height]
    classes_text = [label_map[bbox[4]].encode('utf8')]
    classes = [bbox[4]]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(output_filename, image_annotations, label_map):
    writer = tf.io.TFRecordWriter(output_filename)
    for image_path, bbox in image_annotations:
        tf_example = create_tf_example(image_path, bbox, label_map)
        writer.write(tf_example.SerializeToString())
    writer.close()


def train_coco_ssd(model_config_path, train_tfrecord, eval_tfrecord, num_epochs, label_map_path):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(model_config_path)
    model_config = configs['model']
    detection_model = model_lib_v2.create_center_net_model(model_config, None)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore('path/to/pretrained/checkpoint').expect_partial()

    # Load label map
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Create TFRecord files
    train_annotations, eval_annotations = load_annotations('path/to/your/annotation.txt')
    create_tf_record('train.record', train_annotations, category_index)
    create_tf_record('eval.record', eval_annotations, category_index)

    # Train the model
    for epoch in range(num_epochs):
        model_lib_v2.train_loop(
            pipeline_config=model_config,
            model=detection_model,
            config=configs['train_config'],
            train_input_config=configs['train_input_config'],
            train_steps=configs['train_config'].num_steps_per_iteration,
            checkpoint_dir='path/to/save/checkpoints',
            use_tpu=False  
        )


def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    label_map = {label.strip(): i + 1 for i, label in enumerate(set(line.split()[-1] for line in lines))}

    image_annotations = []
    for line in lines:
        parts = line.split()
        image_path, bbox = parts[0], list(map(int, parts[1:]))
        image_annotations.append((image_path, bbox))

 
    train_annotations, eval_annotations = train_test_split(image_annotations, test_size=0.2, random_state=42)

    return train_annotations, eval_annotations


model_config_path = 'path/to/your/model.config'
train_tfrecord = 'train.record'
eval_tfrecord = 'eval.record'
num_epochs = 100
label_map_path = 'path/to/your/label_map.pbtxt'


train_coco_ssd(model_config_path, train_tfrecord, eval_tfrecord, num_epochs, label_map_path)
