
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

# Continuation Training but with a scheduler for optimization


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fiftyone_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

# Point to the saved checkpoint from the previous training session


# cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/Gamma_office_Dataset/Weights_1/model_final.pth"
cfg.MODEL.WEIGHTS = "models/model_final.pth"


cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.004
cfg.SOLVER.MAX_ITER = 40000
cfg.SOLVER.STEPS = []
cfg.SOLVER.GAMMA = 0.1
# cfg.SOLVER.WARMUP_ITERS = 2000
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
# cfg.OUTPUT_DIR = "/content/drive/MyDrive/Gamma_office_Dataset/Weights_1"

cfg.SOLVER.CHECKPOINT_PERIOD = 2400

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

import torch

# Load the model checkpoint
checkpoint = torch.load(os.path.join(cfg.OUTPUT_DIR, "models/model_final.pth"), map_location=torch.device('cpu'))


# Extract the number of iterations
iterations = checkpoint['iteration']

print(f"The model was trained for {iterations} iterations.")


# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file("/content/drive/MyDrive/detectron/output/content/output/config.yml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "models/model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)


# import fiftyone as fo


# export_dir = "/content/drive/MyDrive/Alpha_dataset_02"
export_dir = "alpha_dataset_02/Alpha_dataset_02"

# Import the dataset
imported_dataset = fo.Dataset.from_dir(
    dataset_dir=export_dir,
    dataset_type=fo.types.FiftyOneDataset,
    name="imported_dataset_name",
    overwrite=True  # Overwrite existing dataset
)


# Get the list of classes for the "ground_truth" field
GT_Classes = imported_dataset.distinct("ground_truth.detections.label")

# Print the classes
print(GT_Classes)

classes = ['Apple', 'Orange', 'Peach', 'Strawberry', 'Grape', 'Pear', 'Lemon', 'Banana',
'Bottle', 'Beer', 'Juice', 'Wine',
'Carrot', 'Bell pepper', 'Cucumber', 'Broccoli', 'Garden Asparagus', 'Zucchini', 'Radish', 'Artichoke', 'Mushroom', 'Potato',
'Pretzel', 'Popcorn', 'Muffin', 'Cheese', 'Cake', 'Cookie', 'Pastry', 'Doughnut',
'Pen', 'Adhesive tape', 'Pencil case', 'Stapler', 'Scissors', 'Ruler',
'Ball', 'Balloon', 'Dice', 'Flying disc', 'Teddy bear',
'Platter', 'Bowl', 'Knife', 'Spoon', 'Saucer', 'Chopsticks', 'Drinking straw', 'Mug',
'Glove', 'Belt', 'Sock', 'Tie', 'Watch', 'Computer mouse', 'Coin', 'Calculator', 'Box', 'Boot', 'Towel', 'Shorts', 'Swimwear',
'Shirt', 'Clock', 'Hat', 'Scarf', 'Roller skates', 'Skirt', 'Mobile phone',
'Plastic bag', 'High heels', 'Handbag', 'Clothing', 'Oyster', 'Tablet computer', 'Book', 'Flower', 'Candle', 'Camera', 'Remote control']


num_classes = len(classes)
print("Number of classes: ", num_classes)

# Empty Polygon Error correction for fiftyone.
from detectron2.structures import BoxMode

def get_fiftyone_dicts(samples, classes):
    samples.compute_metadata()

    dataset_dicts = []
    for sample in samples.select_fields(["id", "filepath", "metadata", "ground_truth"]):
        height = sample.metadata["height"]
        width = sample.metadata["width"]
        record = {}
        record["file_name"] = sample.filepath
        record["image_id"] = sample.id
        record["height"] = height
        record["width"] = width

        objs = []
        for det in sample.ground_truth.detections:
            if det.label in classes:
                tlx, tly, w, h = det.bounding_box
                bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
                fo_poly = det.to_polyline()
                if fo_poly.points:
                    poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
                    poly = [p for x in poly for p in x]
                    if len(poly) >= 6:  # Check if the coordinates are sufficient to form a polygon
                        obj = {
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": [poly],
                            "category_id": classes.index(det.label),
                        }
                        objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


# Sets up Catalog Data for default fiftyone structured files


for d in ["train", "val"]:
    MetadataCatalog.get("fiftyone_" + d).set(thing_classes=classes)

metadata = MetadataCatalog.get("fiftyone_train")

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

        scores = {
            category: score_dict[category].get(class_name, 0) * 100
            for category in score_dict
        }

        if all(score == 0 for score in scores.values()):
            scores["Misc"] = 100

        print(f"Class Name: {class_name}")
        print(f"Object ID: {object_id}")
        print(f"Box: {box}")
        print(f"Segmentation: {segmentation}")
        for category, score in scores.items():
            if score != 0:
                print(f"{category}: {score}%")
        print("\n")


    #     # Convert the numpy array to bytes
    #     segmentation_bytes = segmentation.cpu().numpy().tobytes()

    #     # Encode the bytes to a base64 string
    #     segmentation_b64 = base64.b64encode(segmentation_bytes).decode('utf-8')

    #     payload = {
    #         "class_name": class_name,
    #         "object_id": object_id,
    #         "box": box,
    #         "segmentation": segmentation_b64,
    #         "shape": segmentation.shape
    #     }

    #     json_data = json.dumps(payload)
    #     compressed_data = gzip.compress(bytes(json_data, 'utf-8'))
    #     headers = {'Content-Encoding': 'gzip'}

    #     response = requests.post(url, data=compressed_data, headers=headers)
    #     print("Response: ", response.json())

    # print("---------------------------------------------------")


# import requests
# import json

# # define your Flask-ngrok url here
# url = "http://montclair-object-detector.ngrok.app/"

# def classify_objects(objects_list):
#     for object_info in objects_list:
#         class_name = object_info["class_name"]
#         object_id = object_info["object_id"]
#         segmentation = object_info["segmentation"]
#         box = object_info["box"]

#         scores = {
#             category: score_dict[category].get(class_name, 0) * 100
#             for category in score_dict
#         }

#         if all(score == 0 for score in scores.values()):
#             scores["Misc"] = 100

#         print(f"Class Name: {class_name}")
#         print(f"Object ID: {object_id}")
#         print(f"Box: {box}")
#         print(f"Segmentation: {segmentation}")
#         for category, score in scores.items():
#             if score != 0:
#                 print(f"{category}: {score}%")
#         print("\n")

#         # Sending POST request to the Flask server
#         payload = {
#             "class_name": class_name,
#             "object_id": object_id,
#             "box": box,
#             "segmentation": segmentation.cpu().numpy().tolist()  # Convert to list here
#         }

#         response = requests.post(url, json=payload)
#         print("Response: ", response.json())

#     print("---------------------------------------------------")


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


score_dict = {
    "Fruit": {
        'Apple': 0.98, 'Orange': 0.46, 'Peach': 0.62, 'Strawberry': 0.92,
        'Grape': 0.96, 'Pear': 0.91, 'Banana': 0.95, 'Carrot': 0.88,
        'Pepper': 0.82, 'Cucumber': 0.91, 'Zucchini': 0.97,
        'Radish': 0.76, 'Artichoke': 0.7, 'Cheese': 0.75
    },
    "Drink": {
        'Bottle': 0.66, 'Beer': 0.82, 'Juice': 0.73, 'Wine': 0.84,
        'Mug': 0.96, 'Spoon': 0.46, 'Saucer': 0.46, 'Hat': 0.46,
    },
    "Vegetable": {
        'Strawberry': 0.7, 'Banana': 0.76, 'Carrot': 0.97, 'Pepper': 0.7,
        'Cucumber': 0.88, 'Broccoli': 0.82, 'Garden Asparagus': 0.91,
        'Zucchini': 0.89, 'Artichoke': 0.75, 'Mushroom': 0.76,
        'Potato': 0.91, 'Cheese': 0.7
    },
    "Snacks": {'Apple': 0.69, 'Banana': 0.78, 'Popcorn': 0.89, 'Muffin': 0.84},
    "Stationery": {
        'Pen': 0.87, 'Adhesive tape': 0.67, 'Stapler': 0.67,
        'Ruler': 0.81, 'Calculator': 0.58, 'Box': 0.96, 'Clock': 0.67
    },
    "Toys": {'Ball': 0.75, 'Flying disc': 0.46, 'Teddy bear': 0.87},
    "Tableware": {
        'Orange': 0.83, 'Plate': 0.64, 'Knife': 0.46, 'Spoon': 0.46, 'Saucer': 0.46, 'Chopsticks': 0.58
    }
}

# def classify_objects(objects_list):
#     for object_info in objects_list:
#         class_name = object_info["class_name"]
#         object_id = object_info["object_id"]
#         segmentation = object_info["segmentation"]
#         box = object_info["box"]

#         scores = {
#             category: score_dict[category].get(class_name, 0) * 100
#             for category in score_dict
#         }

#         if all(score == 0 for score in scores.values()):
#             scores["Misc"] = 100

#         print(f"Class Name: {class_name}")
#         print(f"Object ID: {object_id}")
#         print(f"Box: {box}")
#         print(f"Segmentation: {segmentation}")
#         for category, score in scores.items():
#             if score != 0:
#                 print(f"{category}: {score}%")
#         print("\n")

#     print("---------------------------------------------------")

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


