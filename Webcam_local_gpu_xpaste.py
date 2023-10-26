
import fiftyone as fo
import fiftyone.zoo as foz

import torch, detectron2

print("PyTorch version: ", torch.__version__)
print("CUDA version: ", torch.version.cuda)


# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling import build_model, build_roi_heads
from detectron2.config import get_cfg
from detectron2.data.datasets.lvis import get_lvis_instances_meta, register_lvis_instances
from detectron2.config import CfgNode as CN
from custom_timm import *
from config import add_xpaste_config
from center_config import add_centernet_config


# Continuation Training but with a scheduler for optimization

# register_lvis_instances("lvis_v1_val", {}, ".output/lvis_v1_val.json", ".output/images")

lvis_metadata = MetadataCatalog.get("lvis_v1_val")


from detectron2.engine import DefaultTrainer

cfg = get_cfg()

add_xpaste_config(cfg)  # Extend the default configuration

add_centernet_config(cfg)
# Add custom TIMM config
# cfg.MODEL.TIMM = CN()
# cfg.MODEL.TIMM.BASE_NAME = "resnet50"  # default value, you can change it
# cfg.MODEL.TIMM.OUT_LEVELS = [2, 3, 4, 5]  # default value, you can change it
# cfg.MODEL.TIMM.FREEZE_AT = 0  # default value, you can change it
# cfg.MODEL.TIMM.NORM = "FrozenBN"  # default value, you can change it

# cfg.MODEL.BACKBONE.NAME = "build_timm_backbone"


cfg.MODEL.DEVICE = 'cuda'
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file("output/YAML/Base-C2-basep2-mask_L_R5021k_640b64_4x.yaml")
cfg.DATASETS.TRAIN = ("lvis_v1_val",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2





# Point to the saved checkpoint from the previous training session


# cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/Gamma_office_Dataset/Weights_1/model_final.pth"
cfg.MODEL.WEIGHTS = "models/baseline_R50.pth"


cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.004
cfg.SOLVER.MAX_ITER = 40000
cfg.SOLVER.STEPS = []
cfg.SOLVER.GAMMA = 0.1
# cfg.SOLVER.WARMUP_ITERS = 2000
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(lvis_metadata.thing_classes)
# cfg.OUTPUT_DIR = "/content/drive/MyDrive/Gamma_office_Dataset/Weights_1"

cfg.SOLVER.CHECKPOINT_PERIOD = 2400

# Print the number of classes
print(f"Number of classes: {len(lvis_metadata.thing_classes)}")


# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

import torch

# Load the model checkpoint
checkpoint = torch.load(os.path.join(cfg.OUTPUT_DIR, "models/baseline_R50.pth"))


# Extract the number of iterations
iterations = checkpoint['iteration']

print(f"The model was trained for {iterations} iterations.")


# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file("/content/drive/MyDrive/detectron/output/content/output/config.yml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "models/baseline_R50.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)


# import fiftyone as fo


# export_dir = "/content/drive/MyDrive/Alpha_dataset_02"
# export_dir = "alpha_dataset_02/Alpha_dataset_02"

# Import the dataset
# imported_dataset = fo.Dataset.from_dir(
#     dataset_dir=export_dir,
#     dataset_type=fo.types.FiftyOneDataset,
#     name="imported_dataset_name",
#     overwrite=True  # Overwrite existing dataset
# )


# Get the list of classes for the "ground_truth" field
# GT_Classes = imported_dataset.distinct("ground_truth.detections.label")

# Print the classes
# print(GT_Classes)

# classes = ['Apple', 'Orange', 'Peach', 'Strawberry', 'Grape', 'Pear', 'Lemon', 'Banana',
# 'Bottle', 'Beer', 'Juice', 'Wine',
# 'Carrot', 'Bell pepper', 'Cucumber', 'Broccoli', 'Garden Asparagus', 'Zucchini', 'Radish', 'Artichoke', 'Mushroom', 'Potato',
# 'Pretzel', 'Popcorn', 'Muffin', 'Cheese', 'Cake', 'Cookie', 'Pastry', 'Doughnut',
# 'Pen', 'Adhesive tape', 'Pencil case', 'Stapler', 'Scissors', 'Ruler',
# 'Ball', 'Balloon', 'Dice', 'Flying disc', 'Teddy bear',
# 'Platter', 'Bowl', 'Knife', 'Spoon', 'Saucer', 'Chopsticks', 'Drinking straw', 'Mug',
# 'Glove', 'Belt', 'Sock', 'Tie', 'Watch', 'Computer mouse', 'Coin', 'Calculator', 'Box', 'Boot', 'Towel', 'Shorts', 'Swimwear',
# 'Shirt', 'Clock', 'Hat', 'Scarf', 'Roller skates', 'Skirt', 'Mobile phone',
# 'Plastic bag', 'High heels', 'Handbag', 'Clothing', 'Oyster', 'Tablet computer', 'Book', 'Flower', 'Candle', 'Camera', 'Remote control']


# num_classes = len(classes)
# print("Number of classes: ", num_classes)

# Empty Polygon Error correction for fiftyone.
from detectron2.structures import BoxMode



# Sets up Catalog Data for default fiftyone structured files



# metadata = MetadataCatalog.get("fiftyone_train")

metadata = MetadataCatalog.get("lvis_v1_val")


from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time

import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

import cv2
import requests
import json


import requests
import json
import gzip
import base64


def classify_objects(objects_list):
    for object_info in objects_list:
        class_name = object_info["class_name"]
        object_id = object_info["object_id"]
        segmentation = object_info["segmentation"]
        box = object_info["box"]

        print(f"Class Name: {class_name}")
        print(f"Object ID: {object_id}")
        print(f"Box: {box}")
        print(f"Segmentation: {segmentation}")
        print("\n")




def generate_objects_list(outputs, cfg):
    instances = outputs["instances"]

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_catalog = metadata.thing_classes

    return [{
        "class_name": class_catalog[instances.pred_classes[idx]],
        "object_id": idx,
        "segmentation": instances.pred_masks[idx],
        "box": coordinates.tolist()
    } for idx, coordinates in enumerate(instances.pred_boxes.tensor)]



import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    if not ret:
        break

    # Make the predictions
    outputs = predictor(img)
    objects_list = generate_objects_list(outputs, cfg)
    classify_objects(objects_list)

    v = Visualizer(img[:, :, ::-1],
                 metadata=metadata,
                 scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Display the resulting frame
    cv2.imshow('Webcam', out.get_image()[:, :, ::-1])

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


