# Databricks notebook source
# MAGIC %md
# MAGIC #Installing

# COMMAND ----------

# MAGIC %md
# MAGIC #Imports

# COMMAND ----------

from fastai.vision.all import *
from torchvision import transforms
import pandas as pd
from tqdm.notebook import tqdm
import os
import mlflow
from PIL import Image
import cv2

# COMMAND ----------

# MAGIC %md
# MAGIC #Metrics

# COMMAND ----------

class recall(Metric):
    def __init__(self, axis=1): self.axis = axis    
    def reset(self): self.FN, self.TP = 0,0
    def accumulate(self, learn):
        y_pred, y_true = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)  

        self.TP += len(np.where(y_pred.cpu() + y_true.cpu() ==2)[0])
        self.FN += len(np.where(y_pred.cpu() - y_true.cpu()  == 1)[0])
        
    @property
    def value(self):
        return  self.TP/(self.FN + self.TP) if (self.FN + self.TP) > 0 else 0  

class accuracy_bgd(Metric):
    def __init__(self, axis=1): self.axis = axis    
    def reset(self): self.TP, self.TN, self.FN, self.FP = 0,0,0,0
    def accumulate(self, learn):
        
        y_pred, y_true = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
        # self.TPeN += (y_pred == y_true).sum()
        # self.total += y_true.numel()

        self.TP += len(np.where(y_pred.cpu() + y_true.cpu() ==2)[0])
        self.FN += len(np.where(y_pred.cpu() - y_true.cpu()  == 1)[0])
        self.TN += len(np.where(y_pred.cpu() + y_true.cpu() == 0)[0])
        self.FP += len(np.where(y_pred.cpu() - y_true.cpu()  == -1)[0])

    @property
    def value(self):
        
        return ((self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)) if (self.TP + self.FN + self.TN + self.FP) > 0 else 0


class f1score(Metric):
    def __init__(self, axis=1): self.axis = axis    
    def reset(self): self.FP, self.TP, self.FN, self.TN = 0,0,0,0
    def accumulate(self, learn): 
        y_pred, y_true = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
        self.TP += len(np.where(y_pred.cpu() + y_true.cpu() ==2)[0])
        self.FN += len(np.where(y_pred.cpu() - y_true.cpu()  == 1)[0])
        self.TN += len(np.where(y_pred.cpu() + y_true.cpu() == 0)[0])
        self.FP += len(np.where(y_pred.cpu() - y_true.cpu()  == -1)[0])      
    
    @property
    def value(self):
        if ((self.TP+self.FP)!=0):    
            precision = self.TP/(self.TP+self.FP)
        else:
            precision = 0
        if ((self.FN + self.TP)!=0):  
            recall = self.TP/(self.FN + self.TP) 
        else:
            recall = 0       
        return 2*((recall*precision/(recall+precision))) if (recall+precision) > 0 else 0


class precision(Metric):
    def __init__(self, axis=1): self.axis = axis    
    def reset(self): self.FP, self.TP = 0,0
    def accumulate(self, learn):
        y_pred, y_true = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
        self.TP += len(np.where(y_pred.cpu() + y_true.cpu() ==2)[0])
        self.FP += len(np.where(y_pred.cpu() - y_true.cpu()  == -1)[0])
       
                      
    @property    
    def value(self):
        
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0


# COMMAND ----------

# MAGIC %md
# MAGIC #Custom functions

# COMMAND ----------

def label_func(fn): return Path(str(fn.parent)+'_label')/f'{fn.stem}_label_ground-truth_semantic.png'

# COMMAND ----------

# MAGIC %md
# MAGIC #Load model

# COMMAND ----------

def setup_learner(model_path):
    
    # await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(model_path)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


# COMMAND ----------

# MAGIC %md
# MAGIC #Function to show predicts

# COMMAND ----------

import time
def show_my_predictions(original, model):
    
    original = Path(str(original))
    original_pred = original
    original = Image.open(original)
    start = time.time()
    pred = model.predict(original_pred)[1]
    end = time.time()
    print("time to inference: ",(end-start))
    convert_tensor = transforms.ToTensor()
    original = convert_tensor(original)
    original = original.permute(1, 2, 0)
    images = (original,pred)

    plt.figure(figsize=(30, 30))
    title = ['Input Image', 'Predicted Mask','Overlayed']
    for i in range(len(images)+1):
        plt.subplot(1, len(images)+1, i+1)
        plt.title(title[i])
        if (title[i]  == "Overlayed" and i == 2):
            plt.imshow(images[0])
            plt.imshow(images[1], alpha=0.2, cmap='plasma')
            plt.axis('off')
        else:
            if (i<3):
                plt.imshow(images[i], cmap='plasma')
                plt.axis('off')
            else:
                continue
    plt.show()

# COMMAND ----------

def select_n_images(n, dataset_path, stand):
    images = []
    # print(os.listdir(dataset_path))
    all_imgs = os.listdir(dataset_path)
    files = list(filter(lambda k: stand in k, all_imgs))
    if len(files) > 1:
        files = random.sample(files, n if len(files) > n else len(files))
        for file in tqdm(files):
            ext = os.path.splitext(file)[-1]
            if ("processed" not in dataset_path) and (ext == '.jpg'):
                file_path = os.path.join(dataset_path, file)
                images.append(file_path)
                # print(file_path)
    return images

# COMMAND ----------

# MAGIC %md
# MAGIC #Show predicts

# COMMAND ----------

test_path = "/dbfs/mnt/cobble/cleansed/images/test/best_frame/"
model_path = '/dbfs/mnt/cobble/cleansed/images/segmentation/new/models/strip_segmentation_V2.pkl'


# COMMAND ----------

from torchvision.utils import save_image
from pylab import ioff, ion

def pred_and_save(img, model, filename, destination):
    ioff()
    img = Path(str(img))
    img_pred = img
    img = Image.open(img)
    pred = model.predict(img_pred)
    pred = pred[0]
    my_dpi = 96
    fig = plt.figure(figsize=(224/my_dpi, 224/my_dpi), dpi=my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(pred, aspect='auto', cmap='gist_gray')
    plt.savefig(os.path.join(destination, filename))
    plt.clf()
    plt.close()
    ion() 

# COMMAND ----------

# from tqdm.notebook import tqdm

# pre_label_images ='/dbfs/mnt/cobble/cleansed/images/segmentation/image'
# pre_label_path ='/dbfs/mnt/cobble/cleansed/images/segmentation/labels/'


# learn = setup_learner(model_path)
# all_imgs = os.listdir(pre_label_images)
# for img in tqdm(all_imgs):
#     img_full_path = os.path.join(pre_label_images, img)
#     pred_and_save(img_full_path, learn, img, pre_label_path)

# COMMAND ----------

# imgs = select_n_images(5, test_path, 'F5')
# learn = setup_learner(model_path)

# for img in imgs:
#     print(img)
#     show_my_predictions(os.path.join(test_path, img), learn)

# COMMAND ----------

# MAGIC %md
# MAGIC #Loading with mlflow

# COMMAND ----------

model_uri = 'runs:/b1458a19c4c645199aea9cb6a86605ec/segmentation_1600'
learn = mlflow.fastai.load_model(model_uri)

# COMMAND ----------

imgs = select_n_images(5, test_path, 'F5')
# learn = setup_learner(model_path)

for img in imgs:
    print(img)
    show_my_predictions(os.path.join(test_path, img), learn)

# COMMAND ----------

# MAGIC %md
# MAGIC #Getting new labels

# COMMAND ----------

model_uri = "runs:/688fa4b120ee4130ba5694e04b0e8efe/segmentation"
learn = mlflow.fastai.load_model(model_uri)

# COMMAND ----------

import cv2

# COMMAND ----------

img='/dbfs/mnt/cobble/cleansed/images/dataset_V2/best_frame/F5_SUP_TOPO_318_MB11418.jpg'
# img_pred = Path(str(img))
# print(img_pred)
im = cv2.imread(img)
pred = learn.predict(img_pred)
pred

# COMMAND ----------

def label_func(fn):
    return Path(str(fn.parent)+'_label')/f'{fn.stem}_label_ground-truth_semantic.png'

# COMMAND ----------


mask_array = np.array(pred[0])  # Assuming the predicted mask is the first channel
mask_array = np.where(mask_array == 1, 255, 0)
mask_image = Image.fromarray(mask_array.astype('uint8'))
mask_image.save('/dbfs/mnt/cobble/cleansed/images/segmentation/new/models/segmentation_mask.png')

# COMMAND ----------

folder = '/dbfs/mnt/cobble/cleansed/images/segmentation/image'
for img in os.listdir(folder):
    im = cv2.imread(os.path.join(folder, img))
    pred = learn.predict(im)
    mask_array = np.array(pred[0])  # Assuming the predicted mask is the first channel
    mask_array = np.where(mask_array == 1, 255, 0)
    mask_image = Image.fromarray(mask_array.astype('uint8'))
    mask_image.save(f'/dbfs/mnt/cobble/cleansed/images/segmentation/labels/{img}')


# COMMAND ----------


