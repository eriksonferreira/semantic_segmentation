# Databricks notebook source
from PIL import Image
import numpy as np
from segments.utils import bitmap2file
from segments import SegmentsClient
import os
from os import listdir
from os.path import isfile, join
import shutil
import json
import segments
import io
from dotenv import load_dotenv

# COMMAND ----------

# Initialize the segments client
SEGMENTS_KEY = os.getenv("SEGMENTS_KEY")
client = SegmentsClient(SEGMENTS_KEY)

# COMMAND ----------

#define paths to images and masks
path_images = r'/dbfs/mnt/cobble/cleansed/images/segmentation/image/'
path_labels = r'/dbfs/mnt/cobble/cleansed/images/segmentation/labels/'

# COMMAND ----------

from io import BytesIO

# COMMAND ----------


def uploadRespectiveMasks(arquivo, nome, nome2):
        # UPLOAD THE LABEL

        im = Image.open(join(path_labels, arquivo))


        segmentation_bitmap = np.array(im.convert("P"))
        # copyMask = join(path2, file)
        # shutil.copy(copyMask, target)   
        
        threshold = 127
        segmentation_bitmap = np.where(segmentation_bitmap > threshold, 255, 0) 
        segmentation_bitmap = segmentation_bitmap.astype(np.uint32)
        # print(np.unique(segmentation_bitmap)) 
        # print(segmentation_bitmap.shape)

        # Convert it to the segments format using the bitmap2file util function, and upload it to our server

        file = bitmap2file(segmentation_bitmap)
        asset = client.upload_asset(file, nome2) # The name doesn't matter here.
        asset = dict(asset)

        # Now add the label to the sample through the API.
        sample_uuid = sample["uuid"] # This refers to the sample you just uploaded in the previous cell
        labelset = "ground-truth"

        annotations = [{
                "id": 0,
                "category_id": 0
            },
            {
                "id": 255,
                "category_id": 1
            }]
        # for value in np.unique(segmentation_bitmap):
        #     annotations.append({
        #         "id": int(value),
        #         "category_id": int(value)
        #     })
            
        attributes = {
            "format_version": "0.1",
            "annotations": annotations,
            "segmentation_bitmap": {
                "url": asset["url"]
            },
        }

        label = client.add_label(sample_uuid, labelset, attributes, label_status='PRELABELED')
        print(label)



# COMMAND ----------

# UPLOAD THE IMAGE
for root, _, files in os.walk(path_images):
    n = 0       #print (files)
    for file in files:
        
        # First upload the image file to our server, if it's not in the cloud yet. https://docs.segments.ai/python-sdk#upload-a-file-as-an-asset
        image_file = join(path_images, file)
        name1 = file
        asset = None
        try:
            with open(image_file, "rb") as f:
                asset = client.upload_asset(f)
                asset = dict(asset)
                # print(asset)
        except segments.exceptions.NetworkError as e:
             print(e)
             continue

        # Then use the resulting image url to create a sample in a dataset. https://docs.segments.ai/python-sdk#create-a-sample
        dataset_identifier = "eferreira/ltq_v3" # name of your own dataset here. Should already exist.
        name2 = file
        attributes = {
            "image": {
                "url": asset["url"]
            }
        }
        try:
            sample = client.add_sample(dataset_identifier, name2, attributes)
            sample = dict(sample)
            # print(sample)

            uploadRespectiveMasks(file, name1, name2)
            n += 1
        except segments.exceptions.AlreadyExistsError as e:
            print(e)
            continue
    

# COMMAND ----------


