import numpy as np
from PIL.Image import Image
import json
import matplotlib.pyplot as plt

from .image_utils import ClassifierOutputSoftmaxTarget, confidence_change_apply_cam, get_target_index, view


def deletion_metric(input_tensor, cam, model, percentile, outs=True):
    """change something above nth percentile with zeros, check confidence change"""
    new_cam = np.where(cam > np.percentile(cam, percentile), 0, cam)
    out, index = get_target_index(model, input_tensor)
    targets = [ClassifierOutputSoftmaxTarget(index[0])]
    score_raw, scores_on_modified, scores_difference, modified = confidence_change_apply_cam(input_tensor,
                                                                                             new_cam, targets, model)
    if outs:
        print("...On deletion:", scores_on_modified)
        view(input_tensor=input_tensor, modified_tensor=modified, scores_difference=scores_on_modified)
    return new_cam, scores_on_modified, modified


def deletion_game(input_tensor, gray_res, cam, model):
    fin_array = []
    for perc in range(1000, 0, -10):
        perc = perc / 100
        cam_, diff_, modified = deletion_metric(input_tensor, gray_res, model, percentile=perc, outs=False)
        fin_array.append([diff_, perc, cam_, modified])
        fin_array.sort(key=lambda row: (row[0], row[1]), reverse=True)
    diff_, perc, cam_, modified = fin_array[0]
    print("...On deletion game:")
    view(input_tensor=input_tensor, modified_tensor=modified, scores_difference=diff_)
    return diff_, perc, cam_


def preservation_metric(input_tensor, cam, model, left_percentile, right_percentile, outs=True):
    '''change something above nth percentile with zeros, check confidence change'''
    new_cam_left = np.where(cam < np.percentile(cam, left_percentile), cam, 0)
    new_cam_right = np.where(cam < np.percentile(cam, right_percentile), cam, 0)
    out, index = get_target_index(model, input_tensor)
    targets = [ClassifierOutputSoftmaxTarget(index[0])]
    score_raw_left, scores_on_modified_left, scores_difference_left, modified_left = confidence_change_apply_cam(
        input_tensor.cuda(), new_cam_left, targets, model)
    score_raw_right, scores_on_modified_right, scores_difference_right, modified_right = confidence_change_apply_cam(
        input_tensor.cuda(), new_cam_right, targets, model)
    scores_difference = scores_on_modified_right[0] - scores_on_modified_left[0]

    if outs:
        print(score_raw_left, scores_on_modified_left, scores_difference_left)
        print(score_raw_right, scores_on_modified_right, scores_difference_right)
        print("...On addition:")
        view(input_tensor=modified_left, modified_tensor=modified_right, scores_difference=scores_difference,
             preservation=True)
    return scores_difference, modified_left, modified_right, new_cam_left, new_cam_right


def preservation_game(input_tensor, cam, model, cam_):
    fin_array = []
    eps = 0.01
    for perc in range(10, 1000, 10):
        perc_l, perc_r = (perc - 10) * eps, perc * eps
        scores_difference, new_cam_left, new_cam_right, _, _ = preservation_metric(input_tensor, cam, model, perc_l,
                                                                                   perc_r, outs=False)
        fin_array.append([perc_r, scores_difference, new_cam_left, new_cam_right])
    fin_array.sort(key=lambda row: (row[1]), reverse=True)
    perc_r, diff_, modified_left, modified_right = fin_array[0]
    #print("...On preservation game:")
    #view(input_tensor=input_tensor, modified_tensor=modified_right, scores_difference=diff_, preservation=True)
    return diff_, perc, cam_


def average_drop_item(input_tensor, cam, model, index=219):
    targets = [ClassifierOutputSoftmaxTarget(index)]
    score_raw, scores_on_modified, scores_difference, modified = confidence_change_apply_cam(input_tensor.cuda(), cam,
                                                                                             targets, model)
    return max(0, -scores_difference[0]) * 100 / score_raw[0]


def avg_drop_list(list_tens, list_cam, model, index):
    list_out = []
    for tensor, cam, id in zip(list_tens, list_cam, index):
        list_out.append(average_drop_item(tensor, cam, model, id))
    return np.average(list_out)


def increase_in_confidence_item(input_tensor, cam, model, index=219):
    targets = [ClassifierOutputSoftmaxTarget(index)]
    score_raw, scores_on_modified, scores_difference, modified = confidence_change_apply_cam(input_tensor.cuda(), cam,
                                                                                             targets, model)
    if scores_difference[0] > 0:
        return 1
    return 0


def increase_in_confidence_list(list_tens, list_cam, model, index):
    list_out = []
    for tensor, cam, id in zip(list_tens, list_cam, index):
        list_out.append(increase_in_confidence_item(tensor, cam, model, id))
    return np.average(list_out)


def topk_img(input_image, cam, k, gray_res):
    topk = np.sort(gray_res.flatten())[-k:]
    out_image = input_image.copy().reshape(3, 224, 224)

    for i in range(out_image.shape[0]):
        out_image[i] = out_image[i] * 255 * (~np.isin(cam, topk)).astype(int)
    # assert (out_image != input_image).sum() != 0
    return Image.fromarray((out_image.reshape(224, 224, 3)).astype(np.uint8), 'RGB')


def sparsity(cam):
    gr_r = cam.copy()
    gr_r_norm = gr_r - np.min(gr_r) / np.max(gr_r) - np.min(gr_r)
    return 1 / np.mean(gr_r_norm)


def iou_loc(bbox, gray_res):
    bbox = bbox.astype(int)
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