#!/bin/bash
python ../models/research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir=output \
    --pipeline_config_path=dataset/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
    --output_directory inference_graph
