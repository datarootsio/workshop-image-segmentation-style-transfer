from tensorflow import keras
import cv2
import os
import numpy as np
import albumentations as A
from typing import Callable

# classes for data loading and preprocessing
class Dataset:
    """Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir: str, 
            masks_dir: str, 
            classes: list=None, 
            augmentation: Callable[[int],A.core.composition.Compose]=None, 
            preprocessing: Callable[[None],A.core.composition.Compose]=None
    ):
        files = os.listdir(images_dir)
        self.ids = [image_id for image_id in files if '.jpg' in image_id]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids] 
        self.masks_fps = [os.path.join(masks_dir, image_id).replace(".jpg", ".png") for image_id in self.ids]
        self.resize = False
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i: int):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i])
        
        if self.resize:
            image = cv2.resize(image,(self.width,self.length), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask,(self.width,self.length), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = mask.astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

    def limit_data(self, amount_img: int):
        """ Resets the dataset to a smaller subset """
        self.ids = self.ids[:amount_img]
        self.images_fps = self.images_fps[:amount_img]
        self.masks_fps = self.masks_fps[:amount_img]

    def set_resize(self, width:int, length: int):
        """ Sets the measurements for the resizing of the images
            (This must be a multiple of 32) """
        self.resize = True
        self.width = width
        self.length= length
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset:Dataset, batch_size:int=1, shuffle:bool=False,n_classes:int=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()
        self.n_classes = n_classes

    def __getitem__(self, i:int):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size

        width_img,height_img,depth_img= self.dataset[start][0].shape
        width_mask,height_mask,depth_mask= self.dataset[start][1].shape
        data_img  = np.empty((self.batch_size, width_img, height_img, depth_img)) 
        data_mask = np.empty((self.batch_size, width_mask, height_mask, depth_mask))

        for j in range(start, stop):
            data_img[j-start],data_mask[j-start] = self.dataset[j]
        
        return data_img,data_mask
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes) 

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(IMG_SIZE):
    train_transform = [
        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),

        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, always_apply=True, border_mode=0),
        A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),

        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(IMG_SIZE):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
        A.Lambda(mask=round_clip_0_1)
    ]

    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
