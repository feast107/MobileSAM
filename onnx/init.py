import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import mobile_sam
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from mobile_sam import SamAutomaticMaskGenerator


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


checkpoint = "../weights/mobile_sam.pt"
model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint=checkpoint)

path = "C:\\Users\\13532\\Pictures\\74dcc8d9493295e781ef9e7541cda7b8.png"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

ort_session = onnxruntime.InferenceSession("../mobile_sam_single.onnx")
sam.to(device='cpu')
predictor = SamPredictor(sam)

predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()

shape = image_embedding.shape
input_point = np.array([[250, 375]])
input_label = np.array([1])
onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)
ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}
masks, _, low_res_logits = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold
plt.figure(figsize=(10, 10))
plt.imshow(image)
gca = plt.gca()
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for i, img in enumerate(masks[0]):
    axs[i].imshow(img, cmap='gray')
    axs[i].axis('off')

plt.show()
show_mask(masks, gca)
show_points(input_point, input_label, gca)
plt.axis('off')
plt.show()

