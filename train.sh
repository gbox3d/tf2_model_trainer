#!/bin/bash
python ../models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=dataset/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
    --model_dir=output /
    --alsologtostderr