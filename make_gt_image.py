"""
com
|
|----SEG_Train_Datasets
|    |
|    |----Train_Annotations
|    |----Train_Images
|----make_gt_image.py
"""

import os
import json
import cv2
import numpy as np
import shutil
labels = ["STAS"]
labels_color = [(0, 255, 0)]

labels_gray = ["STAS"]
labels_color_gray = [255]

mapping = {
    0: 0,  
    255: 1, #STAS
}

def mask_to_class(mask):
    for k in mapping:
        mask[mask==k] = mapping[k]
    return mask

def get_poly(annot_path): 
    with open(annot_path) as handle:
        data = json.load(handle)
    shape_dicts = data['shapes']
    return shape_dicts

def create_multi_masks(filename,shape_dicts, jsonname):
    #print("jsonname", jsonname)
    cls = [x['label'] for x in shape_dicts]
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts] 
    blank = np.zeros(shape=(942, 1716, 3), dtype=np.uint8)
    for i, (label, poly) in enumerate(zip(cls, poly)):
        if label in cls:
            print(label)
            cv2.fillPoly(blank, [poly], labels_color[labels.index(label)])
    gt_image = cv2.imread(os.path.join(image_dirname, jsonname.split(".")[0] + ".jpg"))
    #print(gt_image.shape)
    #rint(blank.shape)
    dst=cv2.addWeighted(gt_image,1,blank,0.5,0)
    cv2.imwrite(save_dirname + "/" + jsonname[0:-5] + ".png", dst)


def create_gray_masks(shape_dicts, jsonname):
    cls = [x['label'] for x in shape_dicts]
    #print(cls)
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts] 
    blank = np.zeros(shape=(942, 1716), dtype=np.uint8)
    for i, (label, poly) in enumerate(zip(cls, poly)):
        if label in cls:
            cv2.fillPoly(blank, [poly], labels_color_gray[labels_gray.index(label)])
    cv2.imwrite(save_dirname_gray + "/" + jsonname[0:-5] + ".png", blank)

def create_gray_masks_mmsegmentation():
    for i in os.listdir(os.path.join("SEG_Train_Datasets","Train_mask")):
        img = cv2.imread(os.path.join("SEG_Train_Datasets","Train_mask", i), 0)
        img = mask_to_class(img)
        cv2.imwrite(os.path.join("SEG_Train_Datasets","Mmsegmentation_Datasets","labels", i), img)

    for i in os.listdir(os.path.join("SEG_Train_Datasets","Train_Images")):
        shutil.copyfile(os.path.join("SEG_Train_Datasets","Train_Images", i), os.path.join("SEG_Train_Datasets","Mmsegmentation_Datasets", "images",i))

    path = os.path.join(save_dirname_mmsegmentation_splits, "train.txt")
    f = open(path,'w')
    for i in os.listdir(save_dirname_mmsegmentation_labels)[0:900]:
        f.write(str(i.split(".")[0]) + "\n")
    f.close()
    path = os.path.join(save_dirname_mmsegmentation_splits, "val.txt")
    f = open(path,'w')
    for i in os.listdir(save_dirname_mmsegmentation_labels)[900:1053]:
        f.write(str(i.split(".")[0]) + "\n")
    f.close()

annot_dirname = os.path.join("SEG_Train_Datasets", "Train_Annotations")
image_dirname = os.path.join("SEG_Train_Datasets","Train_Images")
save_dirname = os.path.join("SEG_Train_Datasets","Groundtrust_train_image")
save_dirname_gray = os.path.join("SEG_Train_Datasets","Train_mask")

save_dirname_mmsegmentation_labels = os.path.join("SEG_Train_Datasets","Mmsegmentation_Datasets", "labels")
save_dirname_mmsegmentation_images = os.path.join("SEG_Train_Datasets","Mmsegmentation_Datasets", "images")
save_dirname_mmsegmentation_splits = os.path.join("SEG_Train_Datasets","Mmsegmentation_Datasets", "splits")
if not os.path.isdir(os.path.join("SEG_Train_Datasets","Mmsegmentation_Datasets")):
    os.mkdir(os.path.join("SEG_Train_Datasets","Mmsegmentation_Datasets"))
if not os.path.isdir(save_dirname):
    os.mkdir(save_dirname)
if not os.path.isdir(save_dirname_gray):
    os.mkdir(save_dirname_gray)
if not os.path.isdir(save_dirname_mmsegmentation_labels):
    os.mkdir(save_dirname_mmsegmentation_labels)
if not os.path.isdir(save_dirname_mmsegmentation_images):
    os.mkdir(save_dirname_mmsegmentation_images)
if not os.path.isdir(save_dirname_mmsegmentation_splits):
    os.mkdir(save_dirname_mmsegmentation_splits)
for i in os.listdir(annot_dirname):
    shape_dicts = get_poly(annot_dirname+'/'+i)
    create_multi_masks(annot_dirname+'/'+i, shape_dicts, i)
    create_gray_masks(shape_dicts, i)
create_gray_masks_mmsegmentation()
