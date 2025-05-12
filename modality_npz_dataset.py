import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import cv2
from transformers import CLIPModel, CLIPTokenizer
from os.path import join, exists, isfile, isdir, basename
import random

join = os.path.join
import json


def reshape_MR(img):
    
    original_shape = img.shape
    sorted_axes = np.argsort(original_shape)
    new_img = img.transpose(sorted_axes)
    
    return new_img

class ModalityNpzDataset(Dataset):
    def __init__(self,
                 data_root,
                 points=True,
                 contents=True,
                 image_size=256,
                 bbox_shift=5,
                 data_aug=True):
        
        self.data_root = data_root


        json_data = json.load(open("case_data.json", "r"))
        self.file_paths = json_data
        
        assert len(self.file_paths) == 11
        
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        self.points = points
        self.contents = contents

        self.categories_map = {
            "CT": 0,
            "MR": 1,
            "Endoscopy": 2,
            "XRay": 3,
            "X-Ray": 3,
            "PET": 4,
            "Dermoscopy": 5,
            "Mammography": 6,
            "Mammo": 6,
            "US": 7,
            "OCT": 8,
            "Fundus": 9,
            "Microscopy": 10,
            "Microscope": 10
        }

        self.model1 = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.model1.requires_grad_(False)



    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

    def vis(self, image, bboxes, title):
        _, axs = plt.subplots(1, 2, figsize=(10, 10))

        axs[0].imshow(image, cmap="gray")
        self.show_box(bboxes, axs[0])
        axs[0].axis('off')
        axs[0].set_title(title)

        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            "test.png",
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def vis_crop(self, image, title):

        plt.imshow(np.transpose(image, (1,2,0)))
        plt.axis('off')
        plt.title(title)

        plt.savefig(
            "test.png",
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def __getitem__(self, index):
        #! add the random index
        
        modality_map = [
            "CT",
            "MR",
            "Endoscopy",
            "X-ray",
            "PET",
            "Dermoscopy",
            "Mammography", 
            "US",
            "OCT", 
            "Fundus",
            "Microscopy"
        ]
        modality_index = random.randint(0, 10)
        index = random.randint(0, len(self.file_paths[modality_map[modality_index]])-1)
        file_path = self.file_paths[modality_map[modality_index]][index][0]
        temp = '/'.join(file_path.split('/')[7:])
        file_path = self.data_root+'/'+temp

        
        npz = np.load(file_path, 'r', allow_pickle=True)
        img_name = basename(file_path)
        
        mt = img_name.split("_")[0]
        if mt=="2D" or mt=="3D":
            mt = img_name.split("_")[1]
        category_text = f"{mt} Image"
        category_idx = self.categories_map[mt]
        gts = npz["gts"] 
        img = npz["imgs"]

        # special case for MR_totalseg        
        if "MR_totalseg" in img_name:
            img = reshape_MR(img)
            gts = reshape_MR(gts)
            if img.shape[1] <=100:
                return self.__getitem__(random.randint(0,len(self)-1))
        
        if len(gts.shape) > 2: ## 3D image
            i=random.randint(0,gts.shape[0]-1)
            img = img[i, :, :]
            gts = gts[i, :, :]
            img_3c = np.repeat(img[:, :, None], 3, axis=-1) # (H, W, 3)
            img_resized = self.resize_longest_side(img_3c)
        else:
            if len(img.shape) < 3:
                img_3c = np.repeat(img[:, :, None], 3, axis=-1)
            else:
                img_3c = img
            img_resized = self.resize_longest_side(img_3c)
        gts = np.uint16(gts)
        
        # Resizing
        img_resized = (img_resized - img_resized.min()) / np.clip(img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resized) #self.pad_image(img_resize) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        
        label_ids = np.unique(gts)
        label_ids = label_ids.tolist()

        try:
            label_ids.remove(0)
            label_id = random.choice(label_ids)
            gt2D_original = np.uint8(gts == label_id) 
            gt = cv2.resize(
                gt2D_original,
                (img_resized.shape[1], img_resized.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            gt2D = self.pad_image(gt)

        except:
            return self.__getitem__(random.randint(0,len(self)-1))
        
        
        box_original = self.get_bbox(gt2D_original)
        x_mino, y_mino, x_maxo, y_maxo = box_original

        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
            
        try:
            gt2D = np.uint8(gt2D > 0)
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        except:
            return self.__getitem__(random.randint(0,len(self)-1))
        
        if self.points:
            mid_x = (x_min+x_max)//2
            mid_y = (y_min+y_max)//2
            cl = [[y_min, mid_y, x_min, mid_x], [mid_y,y_max,x_min,mid_x], [mid_y,y_max, mid_x,x_max], [y_min,mid_y, mid_x,x_max]]
            coords = []
            for i in range(4):
                gt2D_tmp = np.zeros((H, W))
                gt2D_tmp[cl[i][0]:cl[i][1], cl[i][2]:cl[i][3]] = gt2D[cl[i][0]:cl[i][1], cl[i][2]:cl[i][3]]
                y_indices, x_indices = np.where(gt2D_tmp > 0)
                if y_indices.size==0:
                    coords.append([mid_x, mid_y])
                else:
                    x_point = np.random.choice(x_indices)
                    y_point = np.random.choice(y_indices)
                    coords.append([x_point, y_point])
            coords = np.array(coords).reshape(4, 2)
            coords = torch.tensor(coords).float()
        else:
            coords = None

        if self.contents:
            try:
                crops = img_3c[y_mino:y_maxo,x_mino:x_maxo,:]
                crops_64 = self.m2_pre_img(crops, image_size=64)  # change here for the size of cropped part
                crops_224 = self.m2_pre_img(crops)
            except:
                crops_64 = torch.zeros((3, 64, 64))
                crops_224 = torch.zeros((3, 224, 224))
            crops_224 = crops_224.unsqueeze(0)
            text_token = self.tokenizer(category_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            with torch.no_grad():
                image_features = self.model1.get_image_features(crops_224)
                text_features = self.model1.get_text_features(text_token)
        else:
            crops_64 = None
            image_features = None
            text_features = None


        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "coords": coords,
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(),
            "image_crop": crops_64.float(),
            "image_feature": image_features.float(),
            "text_feature": text_features.float(),
            "category_idx": category_idx,
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_padded.shape[0], img_padded.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }
    
    def __len__(self):
        return 108714
    
    def get_bbox(self, mask_256, bbox_shift=5):
        y_indices, x_indices = np.where(mask_256 > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask_256.shape
        x_min = max(0, x_min - random.randint(0, bbox_shift))
        x_max = min(W, x_max + random.randint(0, bbox_shift))
        y_min = max(0, y_min - random.randint(0, bbox_shift))
        y_max = min(H, y_max + random.randint(0, bbox_shift))
    
        bboxes256 = np.array([x_min, y_min, x_max, y_max])
    
        return bboxes256

    def m2_pre_img(self, image_data, image_size=224):
        transform1 = transforms.Compose([
            transforms.ToTensor(), # normalize to [0.0,1.0]
            transforms.Resize([image_size, image_size], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            ]
        )
        
        resize_img_torch = transform1(image_data)        
        return resize_img_torch
        
    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded

