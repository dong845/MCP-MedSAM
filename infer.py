from os import makedirs
from os.path import join, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models import PromptEncoder, TwoWayTransformer, TinyViT, MaskDecoder_F4
from matplotlib import pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from transformers import CLIPModel, CLIPTokenizer

torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='',
    # required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='',  
    help='directory to save the prediction',
)
parser.add_argument(
    '-lite_medsam_checkpoint_path',
    type=str,
    default="",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)

parser.add_argument(
    '-png_save_dir',
    type=str,
    default=None,
    help='directory to save the overlay image'
)

args = parser.parse_args()

data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
num_workers = args.num_workers

if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)

lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256
model1 = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32", resume_download=True)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16", resume_download=True)
model1.requires_grad_(False)


def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, points, boxes, masks, features, crops, text_features, category_idx):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            boxes = torch.as_tensor(boxes, dtype=torch.float32, device=image.device)
            if len(boxes.shape) == 2:
                boxes = boxes[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
            features=features,
            crops=crops,
            text_features = text_features,
            category_idx=category_idx
        )
        low_res_masks, iou_predictions, category_predictions, clip_vec, img_vec = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     

def show_points(points, ax):
    points = points.numpy()
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, color='yellow', s=15) 

def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box, ratio


def get_points_256(box, gt2D):
    gt2D = np.mean(gt2D, axis=-1)
    if len(box)==1:
        x_min, y_min, x_max, y_max = box[0]
    else:
        x_min, y_min, x_max, y_max = box

    try:
        bounder_shiftx = np.random.randint(int((x_max-x_min)/5), int(2*(x_max-x_min)/5), (1,))
        # bounder_shiftx = int((x_max-x_min)/5)
    except:
        bounder_shiftx = 0
    try:
        bounder_shifty = np.random.randint(int((y_max-y_min)/5), int(2*(y_max-y_min)/5), (1,))
        # bounder_shifty = int((y_max-y_min)/5)
    except:
        bounder_shifty = 0
    
    mid_x = int((x_min+x_max)//2)
    mid_y = int((y_min+y_max)//2)
    x_min = int(x_min+bounder_shiftx)
    x_max = int(x_max-bounder_shiftx)
    y_min = int(y_min+bounder_shifty)
    y_max = int(y_max-bounder_shifty)
    cl = [[y_min, mid_y, x_min, mid_x], [mid_y,y_max,x_min,mid_x], [mid_y,y_max, mid_x,x_max], [y_min,mid_y, mid_x,x_max]]

    coords = []
    for i in range(4):
        gt2D_tmp = np.zeros((256, 256))
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
    return coords

def get_points_256_v0(box, gt2D):
    gt2D = np.mean(gt2D, axis=-1)
    if len(box)==1:
        x_min, y_min, x_max, y_max = box[0]
    else:
        x_min, y_min, x_max, y_max = box
    mid_x = int((x_min+x_max)//2)
    mid_y = int((y_min+y_max)//2)
    try:
        bounder_shiftx = np.random.randint(int((x_max-x_min)/3), int(2*(x_max-x_min)/4)-1, (1,))
        # bounder_shiftx = 0
    except:
        bounder_shiftx = 0
    try:
        bounder_shifty = np.random.randint(int((y_max-y_min)/3), int(2*(y_max-y_min)/4)-1, (1,))
        # bounder_shifty = 0
    except:
        bounder_shifty = 0
    x_min = int(x_min+bounder_shiftx)
    x_max = int(x_max-bounder_shiftx)
    y_min = int(y_min+bounder_shifty)
    y_max = int(y_max-bounder_shifty)
    # cl = [[y_min, mid_y, x_min, mid_x], [mid_y,y_max,x_min,mid_x], [mid_y,y_max, mid_x,x_max], [y_min,mid_y, mid_x,x_max]]

    coords = []
    gt2D_tmp = np.zeros((256, 256))
    gt2D_tmp[y_min:y_max, x_min:x_max] = gt2D[y_min:y_max, x_min:x_max]
    for i in range(4):
        y_indices, x_indices = np.where(gt2D_tmp > 0)
        if y_indices.size==0:
            coords.append([mid_x, mid_y])
        else:
            x_point = np.random.choice(x_indices)
            y_point = np.random.choice(y_indices)
            coords.append([x_point, y_point])
    coords = np.array(coords).reshape(4, 2)
    coords = torch.tensor(coords).float()
    return coords

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, features, crops, text_features, category_idx, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    features = features.unsqueeze(0).to(device)
    crops = crops.unsqueeze(0).to(device)
    category_idx = torch.tensor([category_idx]).to(device)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
        features=features,
        crops=crops,
        text_features = text_features,
        category_idx=category_idx
    )

    low_res_logits, iou, _, _, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg, iou

medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder_F4(
    num_multimask_outputs=3,
    transformer=TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    ),
    modality=True,
    contents=True,
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
)


medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(lite_medsam_checkpoint["model"])
medsam_lite_model.to(device)
medsam_lite_model.eval()


def m2_pre_img(image_data, image_size=224):
    transform1 = transforms.Compose([
        transforms.ToTensor(), # normalize to [0.0,1.0]
        transforms.Resize([image_size, image_size], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        ]
    )
    
    resize_img_torch = transform1(image_data)        
    return resize_img_torch

def get_contents(img, box):
    if len(box)==1:
        x_mino, y_mino, x_maxo, y_maxo = box[0]
    else:
        x_mino, y_mino, x_maxo, y_maxo = box
    crops = img[y_mino:y_maxo,x_mino:x_maxo,:]
    crops_128 = m2_pre_img(crops, image_size=64)
    crops_224 = m2_pre_img(crops)
    crops_224 = crops_224.unsqueeze(0)
    with torch.no_grad():
        image_features = model1.get_image_features(crops_224)
    return crops_128, image_features

def get_text_features(modality_text):
    
    text_token = tokenizer(modality_text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        text_features = model1.get_text_features(text_token)
    return text_features
    

def get_category(idx):
    categories_map = {
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
    return categories_map[idx]

def change_name(name):
    if name=="Microscope":
        name = "Microscopy"
    return name

def MedSAM_infer_npz_2D(img_npz_file):
    npz_name = basename(img_npz_file)
    c_name = change_name(npz_name.split('_')[1])
    modality_text = f"{c_name} Image"
    category_idx = get_category(c_name)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)
    text_features = get_text_features(modality_text)
    text_features = torch.tensor(text_features).unsqueeze(0).to(device)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
    
    for idx, box in enumerate(boxes, start=1):
        crops, features = get_contents(img_3c, box)
        box256, ratio = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, features, crops, text_features, category_idx, (newh, neww), (H, W))
        segs[medsam_mask>0] = idx%256
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
    
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

    # visualize image, mask and bounding box
    if save_overlay and "Microscope" not in npz_name:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            # show_points(points[i], ax[1])
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


def MedSAM_infer_npz_3D(img_npz_file):
    npz_name = basename(img_npz_file)
    c_name = change_name(npz_name.split('_')[1])
    modality_text = f"{c_name} Image"
    category_idx = get_category(c_name)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    # not used in this demo because it treats each slice independently
    # spacing = npz_data['spacing'] 
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]
    text_features = get_text_features(modality_text)
    text_features = torch.tensor(text_features).unsqueeze(0).to(device)

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)
            
            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                crops, features = get_contents(img_3c, mid_slice_bbox_2d)
                box_256, _ = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                if np.max(pre_seg) > 0:
                    box_original = get_bbox256(pre_seg)
                    crops, features = get_contents(img_3c, box_original)
                    pre_seg256 = resize_longest_side(pre_seg)
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    crops, features = get_contents(img_3c, mid_slice_bbox_2d)
                    box_256, _ = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, features, crops, text_features, category_idx, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[z+1, :, :]
            # pre_seg = segs[z+1, :, :]
            if np.max(pre_seg) > 0:
                box_original = get_bbox256(pre_seg)
                crops, features = get_contents(img_3c, box_original)
                pre_seg256 = resize_longest_side(pre_seg)
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                crops, features = get_contents(img_3c, mid_slice_bbox_2d)
                scale_256 = 256 / max(H, W)
                box_256 = mid_slice_bbox_2d * scale_256
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, features, crops, text_features, category_idx, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )            

    # visualize image, mask and bounding box
    if save_overlay and "Microscope" not in npz_name:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    
    img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith('3D'):
            MedSAM_infer_npz_3D(img_npz_file)
        else:
            MedSAM_infer_npz_2D(img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
