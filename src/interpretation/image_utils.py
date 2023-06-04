from typing import List, Callable

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import matplotlib.cm as cm

from torchvision import transforms
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

cudnn.benchmark = True  # fire on all cylinders


def im_show(im_path):
    img = np.array(Image.open(im_path).convert('RGB'))
    img = cv2.resize(img, (224, 224))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    plt.imshow(img)
    return img


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    # print(raw_image.shape)
    raw_image = cv2.resize(raw_image, (224, 224))
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate([image_paths]):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


def tensor_x_cam(input_tensor, cam):
    return input_tensor * cam


def scores_on_targets(model, targets: List[Callable], input: torch.Tensor):
    with torch.no_grad():
        logits = model(input)
        scores = [target(output).cpu().numpy() for target, output in zip(targets, logits)]
        return np.float32(scores)


def confidence_change_apply_cam(input_tensor: torch.Tensor, grayscale_cams: np.ndarray, targets: List[Callable], model,
                                return_diff=True):
    modified = []
    for i in range(input_tensor.size(0)):
        tensor = tensor_x_cam(input_tensor[i, ...].cpu(), torch.from_numpy(grayscale_cams[i]))
        tensor = tensor.to(input_tensor.device)
        modified.append(tensor.unsqueeze(0))
    modified = torch.cat(modified)

    scores_on_modified = scores_on_targets(targets=targets, input=modified, model=model)
    scores_raw = scores_on_targets(targets=targets, input=input_tensor, model=model)
    scores_difference = scores_on_modified - scores_raw

    return scores_raw, scores_on_modified, scores_difference, modified


def view(input_tensor, modified_tensor, scores_difference, preservation=False):
    input_tens = input_tensor.detach().clone()
    modified_tens = modified_tensor.detach().clone()

    input_tens = input_tens.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    modified_tens = modified_tens.squeeze(0).cpu().numpy().transpose((1, 2, 0))

    print(f"Confidence change: {100 * scores_difference} %")
    raw_img = Image.fromarray(deprocess_image(input_tens))
    modified_img = Image.fromarray(deprocess_image(modified_tens))

    subplots = [
        ('Source img', [(raw_img, None, None)]),
        ('Source img & Saliency mapping', [(modified_img, None, None)])
    ]
    if preservation:
        subplots = [
            ('Before_addition', [(raw_img, None, None)]),
            ('After addition', [(modified_img, None, None)])
        ]

    num_subplots = len(subplots)

    fig = plt.figure(figsize=(16, 3))

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()
        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)
        ax.set_title(title)


def get_target_index(model, input_tensor):
    model.eval()
    out = model(input_tensor.cuda())
    _, index = torch.max(out, 1)
    return out, index


def target_outs(model, input_tensor, softmax=True):
    out, index = get_target_index(model, input_tensor)
    if len(out.shape) == 1:
        if softmax:
            return torch.softmax(out, dim=-1)[index]
        else:
            return out[index]
    else:
        if softmax:
            return torch.softmax(out, dim=-1)[:, index]
        else:
            return out[:, index]


def deprocess_image(img):
    img = 0.1 * (img - np.mean(img)) / (np.std(img) + 1e-5) + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def preprocess_image(
        img: np.ndarray,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


class ClassifierOutputSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]


def to_plot_bbox(bbox):
    bbox = bbox.detach().numpy() / 300 * 224
    return (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1]


def iou_loc(bbox, cam, gray_res):
    bbox = (bbox.detach().numpy() / 300 * 224).astype(int)
    external_pixels = 0
    delta = 0.15
    for i in range(gray_res.shape[0]):
        for j in range(gray_res.shape[1]):
            if gray_res[i][j] > delta and (i < bbox[1] or i > bbox[3] or j < bbox[0] or j > bbox[2]):
                external_pixels += 1

    bboxed = gray_res[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    bounding_box = bboxed.shape[0] * bboxed.shape[1]
    internal = np.count_nonzero(~np.isnan(bboxed))

    iou_loc_ = internal / (bounding_box + external_pixels)
    return iou_loc_
