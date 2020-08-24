"""
    Inferencing functions and utils
     - Config class for inferencing
     - Building MaskRCNN model form weights path and device type string
     - Processing image which is uploaded
     - Getting predictions (RoI and segmentation coverage)
     - Getting percent coverage of segmented masks over the whole image.
"""
import os
import skimage
import numpy as np
from mrcnn.model import MaskRCNN
from mrcnn.visualize import save_masked_image
from mrcnn.trashmask import TrashoutConfig
import tensorflow as tf
from typing import Dict, Tuple


class InferenceConfig(TrashoutConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_model(weight_path: str, device: str) -> MaskRCNN:
    tf_device = '/cpu:0' if device == 'cpu' else '/gpu:0'
    cfg = InferenceConfig()
    with tf.device(tf_device):
        mdl = MaskRCNN(mode="inference", model_dir='logs',
                       config=cfg)
    mdl.load_weights(weight_path, by_name=True)
    return mdl


def process_image(model: MaskRCNN, img_path: str, result_dir: str, output_filename: str) -> Tuple[Dict, float]:
    r, coverage = get_predictions(img_path, model=model)
    img_arr = skimage.io.imread(img_path)
    output_filepath = os.path.join(result_dir, output_filename)
    save_masked_image(img_arr, r['rois'], r['masks'], r['class_ids'],
                      ['BG', 'Magazine', 'Newspaper', 'Books', 'Aerosol can', 'Bulky plastic', 'Cardboard', 
                       'Construction material', 'Electronic', 'Flexible plastic', 'Furniture', 'Mattress',
                       'Glass', "Flexible bags, plastic", 'Metal cap', 'Other glass', 'Other Hazardous',
                       'Aluminum foil', 'Battery', 'Dead Animal', 'Organic waste', 'Paper', 'other plastic', 
                       'Textile', 'Metal can', 'Other paper', 'Drink carton', 'Glass jar/bottle/container', 
                       "Other Plastic CD's", "Other Plastic Vinyl", "Other Plastic Styrofoam", "Loose paper/envelope"], r['scores'], save_filepath=output_filepath)
    return r, coverage


def get_predictions(image_path: str, model: MaskRCNN) -> Tuple[Dict, float]:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=0)
    r = results[0]
    coverage = get_mask_coverage(r['masks'])
    return r, coverage

def predict(image_path: str, model: MaskRCNN) -> Tuple[Dict, float]:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=0)
    r = results[0]
    classes =['BG', 'Magazine', 'Newspaper', 'Books', 'Aerosol can', 'Bulky plastic', 'Cardboard', 
                       'Construction material', 'Electronic', 'Flexible plastic', 'Furniture', 'Mattress',
                       'Glass', "Flexible bags, plastic", 'Metal cap', 'Other glass', 'Other Hazardous',
                       'Aluminum foil', 'Battery', 'Dead Animal', 'Organic waste', 'Paper', 'other plastic', 
                       'Textile', 'Metal can', 'Other paper', 'Drink carton', 'Glass jar/bottle/container', 
                       "Other Plastic CD's", "Other Plastic Vinyl", "Other Plastic Styrofoam", "Loose paper/envelope"]  
    cats=[]
    for i in r['class_ids']:
        cats.append(classes[i])
    return cats


def get_mask_coverage(masks: np.ndarray) -> float:
    mask_pixels = np.sum(masks)
    tot_pixels = masks.shape[0] * masks.shape[1]
    return (mask_pixels/tot_pixels) * 100
