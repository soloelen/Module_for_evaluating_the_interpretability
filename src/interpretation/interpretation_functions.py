"""
**Common methods**
- Saliency maps - done
- Occlussion sensativity - done
- Integrated Gradients - **should be tested**
- LRPs
- Deep Taylor Decomposition

**Backprops**
- VanilaBackprop
- GuidedBackprop - done

**CAMs**
- GradCAM - done
- HiResCAM - done
- GradCAM ElementWise - done
- GradCAM++ - done
- XGradCAM - done
- AblationCAM - done
- ScoreCAM - done
- EigenCAM - done
- EigenGradCAM - done
- LayerCAM- done
- RandomCAM - done
- FullGrad - done
- Deep Feature Factorizations

**Global methods**
- SHAP
- LIME
"""
from collections.abc import Sequence

import torch
import os
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torch.autograd import Variable
from tqdm.notebook import tqdm
from pytorch_grad_cam import EigenCAM, EigenGradCAM, LayerCAM
from pytorch_grad_cam import GradCAM, GradCAMElementWise
from pytorch_grad_cam import AblationCAM, RandomCAM, FullGrad, ScoreCAM, HiResCAM, XGradCAM
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image

from .image_utils import load_images, save_sensitivity
from ..config.config import DEVICE

cudnn.benchmark = True  # fire on all cylinders


def eigen_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = EigenCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def eigengrad_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = EigenGradCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def grad_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = GradCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def gradpp_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = GradCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def ablation_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = AblationCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def random_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = RandomCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def fullgrad_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = FullGrad(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def score_cam_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = ScoreCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def hires_cam_gen(model, img, target_layers):  # need test
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = HiResCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def elw_grad_cam_gen(model, img, target_layers):  # need test
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = GradCAMElementWise(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def xgrad_cam_gen(model, img, target_layers):  # need test
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = XGradCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def layer_cam_gen(model, img, target_layers):  # need test
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_()
    tensor.to(DEVICE)
    cam = LayerCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def guided_backprop_gen(model, img, target_layers):
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    tensor.to(DEVICE)
    cam = GuidedBackpropReLUModel(model, target_layers)
    grayscale_cam = cam(tensor)[:, :, 0]
    print(grayscale_cam.shape)
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image, grayscale_cam


def saliency_gen(img, model):
    # we don't need gradients weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    input = img
    input.unsqueeze_(0)
    input.requires_grad = True
    preds = model(input)
    score, indices = torch.max(preds, 1)
    score.backward()
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)

    return slc


# Integrated gradients should be updated.
def integrated_grads_gen(batch_x, model, batch_blank_type='zero', iterations=100):
    mean_grad = 0

    transform = transforms.ToTensor()
    batch_x = transform(batch_x).unsqueeze(0)
    batch_x = batch_x.to(DEVICE)
    model.to(DEVICE)

    if batch_blank_type == 'zero':
        batch_blank = torch.zeros_like(batch_x)
    elif batch_blank_type == 'one':
        batch_blank = torch.ones_like(batch_x)
    elif batch_blank_type == 'rand':
        batch_blank = torch.rand_like(batch_x)

    batch_blank = batch_blank.to(DEVICE)

    for i in tqdm(range(1, iterations + 1)):
        k = i / iterations
        x = batch_blank + k * (batch_x - batch_blank)
        x = Variable(x, requires_grad=True)
        x = x.to(DEVICE)

        with torch.enable_grad():
            outputs = model(x)

            value, preds = torch.max(outputs, 1)

            print(value, preds)
            predictions = preds.type(torch.cuda.FloatTensor)
            predictions = Variable(predictions, requires_grad=True)
            predictions.retain_grad()

            # Comment underline is a 1st approach to get grads
            # predictions.backward(retain_graph=True)
            # print(x.grad)
            # grad = x.grad

            # Comment underline is a 2nd approach to get grads
            (grad,) = torch.autograd.grad(predictions, x, allow_unused=True)

            # grad = grad.retain_grad()

        print('iter = ', i, 'grad ', grad)
        if grad is None:
            grad = 0
        mean_grad += grad / iterations

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients, mean_grad


def occlusion_sensitivity(model,
                          images,
                          ids,
                          mean=None,
                          patch=35,
                          stride=1,
                          n_batches=128):
    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i: i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h: grid_h + patch_H, grid_w: grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps


def occlusion_sens_gen(image_paths, class_names, model, output_dir, cuda, topk, stride, n_batches):
    classes = class_names

    # Model from torchvision
    model = model
    model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    model.eval()

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(DEVICE)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 50]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=os.path.join(
                        output_dir, "new" + str(i) + ".png"
                    ),
                    maps=sensitivity[j],
                )
