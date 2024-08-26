# Databricks notebook source
from fastai.vision.all import *
import pandas as pd
# import mlflow -- removido para testes locais

# COMMAND ----------

path = Path(r'/home/eferreira/master/mestrado_cnn/projeto_final/semantic_segmentation/dataset')
fnames_train = get_image_files(path/'train/')
fnames_val = get_image_files(path/'val/')

# COMMAND ----------

fnames = get_image_files(path/'train/')
for item in fnames_train:
  fnames.append(item)

# COMMAND ----------

codes = array(['void','strip'], dtype='<U17')

train = 'train'
valid = 'val'
def label_func(fn): return Path(str(fn.parent)+'_label')/f'{fn.stem}_label_ground-truth_semantic.png'

# COMMAND ----------

def _parent_idxs(items, name):
    def _inner(items, name): return mask2idxs(Path(o).parent.name == name for o in items)
    return [i for n in L(name) for i in _inner(items,n)]

# COMMAND ----------

def ParentSplitter(train_name='train', valid_name='val'):
    "Split `items` from the parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _parent_idxs(o, train_name),_parent_idxs(o, valid_name)
    return 

# COMMAND ----------

class SegmentationCustomDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for segmentation problems"
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func(cls, path, label_func, train='train', valid='val', valid_pct=None, seed=None, codes=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from list of `fnames` in `path`s with `label_func`."
        
        dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes=codes)),
                           splitter=ParentSplitter(train_name=train, valid_name=valid),
                           get_items = partial(get_image_files, folders=[train, valid]),
                           get_y=label_func,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        res = cls.from_dblock(dblock, path, path=path, **kwargs)
        return res 

# COMMAND ----------

dls = SegmentationCustomDataLoaders.from_label_func(
        path, label_func = label_func, codes = codes, bs=48, batch_tfms=aug_transforms(), num_workers=1
    )

# COMMAND ----------

dls.show_batch(max_n=6)

# COMMAND ----------

class accuracy_bgd(Metric):
    def __init__(self, axis=1): self.axis = axis    
    def reset(self): self.TP, self.TN, self.FN, self.FP = 0,0,0,0
    def accumulate(self, learn):
        
        y_pred, y_true = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)

        self.TP += len(np.where(y_pred.cpu() + y_true.cpu() ==2)[0])
        self.FN += len(np.where(y_pred.cpu() - y_true.cpu()  == 1)[0])
        self.TN += len(np.where(y_pred.cpu() + y_true.cpu() == 0)[0])
        self.FP += len(np.where(y_pred.cpu() - y_true.cpu()  == -1)[0])
        
    @property
    def value(self):
        
        return ((self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)) if (self.TP + self.FN + self.TN + self.FP) > 0 else 0


# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    
    "Computes non-background accuracy for multiclass segmentation"
    targ = cast(targ.squeeze(1), TensorBase)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean()

# COMMAND ----------

metrics= {JaccardCoeff}

# COMMAND ----------

learn = unet_learner(dls, resnet34, loss_func=CrossEntropyLossFlat(axis=1), metrics=metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Fine tuning more stands

# COMMAND ----------

# with mlflow.start_run():
#     learn.load('/dbfs/mnt/cobble/cleansed/images/segmentation/new/models/strip_segmentation_V2')
#     lr = 0.0001
#     learn.fine_tune(15)
#     lr = 0.00005
#     learn.fit_one_cycle(25,lr)
#     learn.unfreeze()
#     lr = 0.00005
#     learn.fit_one_cycle(25, lr)
#     learn.fine_tune(15)
#     jaccard = float(learn.recorder.values[-1][2]) 
#     mlflow.log_metric("jaccard", jaccard)
#     mlflow.fastai.log_model(learn, "segmentation_1600")

lr = 0.0001
learn.fine_tune(15)
lr = 0.00005
learn.fit_one_cycle(25,lr)
learn.unfreeze()
lr = 0.00005
learn.fit_one_cycle(25, lr)
learn.fine_tune(15)
jaccard = float(learn.recorder.values[-1][2]) 

# COMMAND ----------

learn.export(fname='/home/eferreira/master/mestrado_cnn/projeto_final/semantic_segmentation/models/strip_segmentation.pkl')
learn.save('/home/eferreira/master/mestrado_cnn/projeto_final/semantic_segmentation/dataset/models/strip_segmentation')

# COMMAND ----------

interp = SegmentationInterpretation.from_learner(learn)
interp.plot_top_losses(k=5)

# COMMAND ----------

from torchvision import transforms
from PIL import Image

import seaborn as sn
import csv

def label_func(fn): return Path(str(fn.parent)+'_label')/f'{fn.stem}.png'

def show_my_predictions(original, model):
    original = Path(str(original))
    original_pred = original
    original = Image.open(original)
    pred = model.predict(original_pred)[1]
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

from tqdm.notebook import tqdm
def select_n_images(n, dataset_path, stand):
    images = []
    all_imgs = os.listdir(dataset_path)
    files = list(filter(lambda k: stand in k, all_imgs))
    if len(files) > 1:
        files = random.sample(files, n if len(files) > n else len(files))
        for file in tqdm(files):
            ext = os.path.splitext(file)[-1]
            if ("processed" not in dataset_path) and (ext == '.jpg'):
                file_path = os.path.join(dataset_path, file)
                images.append(file_path)
    return images

# COMMAND ----------

# test_path = "/dbfs/mnt/cobble/cleansed/images/test/best_frame/"

# imgs = select_n_images(5, test_path, 'F4')

# for img in imgs:
#     print(img)
#     show_my_predictions(os.path.join(test_path, img), learn)

# COMMAND ----------


