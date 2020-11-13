# savemodel 예제 
# %%
#load saved model and infulence example

# import module start
import io
import os
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import label_map_util

from IPython.display import display

# import utils.visual_util

import sys
sys.path.append('./libs')
from utils_ai import pil_draw_lib as pdl


# print(f'load tf ok {tf.__version__}')


print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.list_physical_devices("GPU") else "사용 불가능")


# %%
# load trained model
print('loading trained model')

# model_name = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'

model_dir = f"./inference_graph/saved_model"
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(str(model_dir))
end_time = time.time()
elapsed_time = end_time - start_time
print(f'load complete {model_dir} Elapsed time: {elapsed_time}s ')

# %%
# load image
image_path = './dataset/test/IMG_20181228_102706.jpg'
print('load image')
# numpy형으로 변환
image_np = np.array(Image.open(image_path))

display(Image.fromarray(image_np))
# print(image_np.shape)
# print(type(image_np))
# print(f'{image_path} load ok')
# plt.imshow(image_np)


# %%
# do inference
input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)
# print(detections)
print('inference complete')

# %%
boxes = detections['detection_boxes'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)
scores = detections['detection_scores'][0].numpy()

# 추정치가 50% 이상만 추출
_detections = [v for v in zip(scores, classes, boxes) if v[0] > 0.3]

print(_detections)

#%%
#결괴물 출력 
img_with_dection = Image.fromarray(image_np.copy()) #원본복사하여 pil로 변환

for _d in _detections :
  print(_d)
  _score = _d[0]
  _class = _d[1]
  ymin, xmin, ymax, xmax = _d[2] #감지 박스 구하기 
  pdl.draw_bounding_box_on_image(img_with_dection,ymin, xmin, ymax, xmax)

display(img_with_dection)
# %%
