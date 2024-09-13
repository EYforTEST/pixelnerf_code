import os
import glob
import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images

import numpy as np

import cv2
import shutil

import matplotlib
import matplotlib.pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='./nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--image_path', default='D:/shapenet_dtset/srn_cars/cars_train', type=str, help='path to a test image or folder of images')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load model into GPU / CPU
print('Loading model...')
model = load_model(args.model, custom_objects=custom_objects, compile=False)
print('\nModel loaded ({0}).'.format(args.model))

# Compute results
image_path = args.image_path

# Load image and preprocess
obj_list = os.listdir(image_path)

for obj in obj_list:
    # shutil.rmtree(os.path.join(image_path, obj) + '/depth')
    os.makedirs(os.path.join(image_path, obj) + '/densedepth', exist_ok=True)

    obj_path = os.path.join(image_path, obj) + '/rgb'

    img_list = os.listdir(obj_path) # 000000.png
    
    for img in img_list:
        input_path = os.path.join(obj_path, img)
        inputs = load_images( glob.glob(input_path) )[:,:,:,:3]

        outputs0 = predict(model, inputs)
        print(outputs0.shape)

        # SAVE
        output_directory = os.path.join(image_path, obj) + '/densedepth'
        name_dest_npy = os.path.join(output_directory, f"{img.split('.')[0]}_densedepth.npy")

        np.save(name_dest_npy, outputs0)

print('-> Done!')