from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION
print(f'find tf version {VERSION}')
if VERSION >= "2.0.0a0":
    print('run tf1 compatible mode')
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

#od api가 설치되어야함  
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
flags.DEFINE_string('labelmap', '', 'Path to labelmap file')
FLAGS = flags.FLAGS

_labelmap_dic = {}
_label_box_count = 0

def loadLabelMap(label_file) :
    with open(label_file,"rt") as fd :
        # fd = open('./res/labelmap.pbtxt',"rt")
        _text = fd.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
        text_format.Merge(_text, label_map)
    except text_format.ParseError:
        label_map.ParseFromString(_text)

        # print(label_map)
        
    for item in label_map.item:
        print(item)
        # print(item.name)
        # print(item.id)
        _labelmap_dic[item.name] = item.id
    print('label load done!')



def class_text_to_int(row_label):
    global _label_box_count

    _label_box_count += 1

    _id = _labelmap_dic[row_label]
    print(f'{_label_box_count} : {row_label} => {_id}')
    return _id
    # if row_label == 'Raspberry_Pi_3':
    #     return 1
    # elif row_label == 'Arduino_Nano':
    #     return 2
    # elif row_label == 'ESP8266':
    #     return 3
    # elif row_label == 'Heltec_ESP32_Lora':
    #     return 4
    # else:
    #     return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

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


def main(_):
    loadLabelMap(FLAGS.labelmap)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

if __name__ == '__main__':
    tf.app.run()