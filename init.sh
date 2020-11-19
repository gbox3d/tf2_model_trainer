#!/bin/bash

python xml_to_csv.py -d dataset

python generate_tfrecord.py --csv_input=dataset/train_labels.csv --image_dir=dataset/train --output_path=train.record --labelmap=dataset/labelmap.pbtxt
python generate_tfrecord.py --csv_input=dataset/test_labels.csv --image_dir=dataset/test --output_path=test.record --labelmap=dataset/labelmap.pbtxt