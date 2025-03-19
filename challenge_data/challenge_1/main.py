
import glob
import json
import math
import os
import numpy as np
import numba
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path
from multiprocessing import Pool
import torch
from pytorch3d.ops import box3d_overlap



##################################
# Evaluation Script for 3D Object Detection
##################################


iou_threshold_dict = {
    "CAR": 0.1,
    "TRUCK": 0.1,
    "TRAILER": 0.1,
    "VAN": 0.1,
    "MOTORCYCLE": 0.1,
    "BUS": 0.1,
    "PEDESTRIAN": 0.1,
    "BICYCLE": 0.1,
    "EMERGENCY_VEHICLE": 0.1,
    "OTHER": 0.1,
}


# @cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4,), dtype=numba.float32)
    corners_y = cuda.local.array((4,), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 *
                i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i
                + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


# @cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((16,), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


# @cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/pcdet/rotation).

    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims,
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    for i, query_box in enumerate(query_boxes):
        for j, box in enumerate(boxes):
            iou[i, j] = devRotateIoUEval(query_box, box, criterion)
    # threadsPerBlock = 8 * 8
    # cuda.select_device(device_id)
    # blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    # stream = cuda.stream()
    # with stream.auto_synchronize():
    #     boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
    #     query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
    #     iou_dev = cuda.to_device(iou.reshape([-1]), stream)
    #     rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
    #         N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
    #     iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


def get_3d_box(box_size, heading_angle, center):
    """Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    """

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def rotate_iou_cpu_one(box_info):
    gt_box, pred_box = box_info
    if np.linalg.norm(gt_box[:3] - pred_box[:3]) > 5:
        return 0.0

    # diff_rot = gt_box[6] - pred_box[6]
    # diff_rot = np.abs(diff_rot)
    # if diff_rot > np.pi:
    #     diff_rot = 2 * np.pi - diff_rot
    # if diff_rot > np.pi / 2:
    #     return 0.0

    corners_3d_ground = get_3d_box(gt_box[3:6], gt_box[-1], gt_box[[0, 2, 1]])
    corners_3d_predict = get_3d_box(pred_box[3:6], pred_box[-1], pred_box[[0, 2, 1]])
    # Sanity checks with ground truth revealed instability of this
    # method, so replaced by new method from Facebook
    # iou_3d, _ = box3d_iou(corners_3d_ground, corners_3d_predict)
    # Preparation for Facebook method https://pytorch3d.org/docs/iou3d
    corners_3d_ground = torch.from_numpy(np.expand_dims(corners_3d_ground, axis=0).astype(np.float32))
    corners_3d_predict = torch.from_numpy(np.expand_dims(corners_3d_predict, axis=0).astype(np.float32))
    _, iou_3d_pytorch3d = box3d_overlap(corners_3d_ground, corners_3d_predict)
    iou_3d_pytorch3d = iou_3d_pytorch3d.numpy().item()
    return iou_3d_pytorch3d


def rotate_iou_cpu_eval(gt_boxes, pred_boxes):
    """

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes:

    Returns:

    """
    data_list = []
    gt_num = len(gt_boxes)
    for gt_box in gt_boxes:
        for pred_box in pred_boxes:
            data_list.append((gt_box, pred_box))
    with Pool(8) as pool:
        result = pool.map(rotate_iou_cpu_one, data_list)
    # For Debugging: same result but no multithread
    # result = []
    # for item in data_list:
    #     result.append(rotate_iou_cpu_one(item))
    result = np.array(result)
    if result.size > 0:
        result = result.reshape((gt_num, -1))
    return result


def compute_split_parts(num_samples, num_parts):
    part_samples = num_samples // num_parts
    remain_samples = num_samples % num_parts
    if part_samples == 0:
        return [num_samples]
    if remain_samples == 0:
        return [part_samples] * num_parts
    else:
        return [part_samples] * num_parts + [remain_samples]


def overall_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    if len(boxes) == 0:
        return ignore

    # calculate euclidian distance
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0:
        flag = dist < 64
    elif level == 1:  # 0-40m
        flag = dist < 40
    elif level == 2:  # 40-50m
        flag = (dist >= 40) & (dist < 50)
    elif level == 3:  # 50m-inf
        # TODO: temp crop labels at 64 m
        flag = (dist >= 50) & (dist < 64)
    else:
        assert False, "level < 4 for overall & distance metric, found level %s" % (str(level))

    ignore[flag] = False
    return ignore


def overall_distance_filter(boxes, level):
    # check if boxes are empty
    if len(boxes) == 0:
        print("shape of boxes", boxes.shape)
        return np.ones(boxes.shape[0], dtype=bool)
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if level == 0:
        flag = np.ones(boxes.shape[0], dtype=bool)
    elif level == 1:  # 0-40m
        flag = dist < 40
    elif level == 2:  # 40-50m
        flag = (dist >= 40) & (dist < 50)
    elif level == 3:  # 50m-inf
        # TODO: temp crop labels at 64 m
        flag = (dist >= 50) & (dist < 64)
    else:
        assert False, "level < 4 for overall & distance metric, found level %s" % (str(level))

    ignore[flag] = False
    return ignore




def get_evaluation_results(
    gt_annotation_frames,
    pred_annotation_frames,
    classes,
    iou_thresholds=None,
    num_pr_points=50,
    difficulty_mode="Overall&Distance",
    ap_with_heading=True,
    num_parts=100,
    print_results=False
):
    if iou_thresholds is None:
        iou_thresholds = iou_threshold_dict

    assert len(gt_annotation_frames) == len(pred_annotation_frames), "the number of GT must match predictions"
    assert difficulty_mode in ["EASY", "MODERATE", "HARD", "OVERALL"], "difficulty mode is not supported"

    num_samples = len(gt_annotation_frames)
    split_parts = compute_split_parts(num_samples, num_parts)
    # Use GPU for IoU 3D calculation
    #ious = compute_iou3d(gt_annotation_frames, pred_annotation_frames)
    ious = compute_iou3d_cpu(gt_annotation_frames, pred_annotation_frames)
    num_classes = len(classes)
    num_difficulties = 4
    difficulty_types = ["overall_0_inf", "0-40m", "40-50m", "50m-64"]
    precision = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    iou_3d = np.zeros([num_classes, num_difficulties])
    pos_err = np.zeros([num_classes, num_difficulties])
    rot_err = np.zeros([num_classes, num_difficulties])

    gt_class_occurrence = {}
    pred_class_occurrence = {}
    for cur_class in classes:
        gt_class_occurrence[cur_class] = 0
        pred_class_occurrence[cur_class] = 0

    for sample_idx in range(num_samples):
        gt_anno = gt_annotation_frames[sample_idx]
        pred_anno = pred_annotation_frames[sample_idx]

        if len(gt_anno["name"]) == 0 or len(pred_anno["name"]) == 0:
            print("no gt or prediction")
            continue


        for cur_class in classes:
            if gt_anno["name"].size > 0:
                gt_class_occurrence[cur_class] += (gt_anno["name"] == cur_class.upper()).sum()
            if pred_anno["name"].size > 0:
                pred_class_occurrence[cur_class] += (pred_anno["name"] == cur_class.upper()).sum()

    for cls_idx, cur_class in enumerate(classes):
        iou_threshold = iou_thresholds[cur_class.upper()]
        for diff_idx in range(num_difficulties):
            ### filter data & determine score thresholds on p-r curve ###
            accum_all_scores, accum_all_ious, accum_all_pos, accum_all_rot, gt_flags, pred_flags = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            num_valid_gt = 0

            for sample_idx in range(num_samples):
                gt_anno = gt_annotation_frames[sample_idx]
                pred_anno = pred_annotation_frames[sample_idx]

                pred_score = pred_anno["score"]
                if len(ious) > 0:
                    iou = ious[sample_idx]
                    gt_flag, pred_flag = filter_data(
                        gt_anno,
                        pred_anno,
                        difficulty_mode,
                        difficulty_level=diff_idx,
                        class_name=cur_class.upper()
                    )
                    gt_flags.append(gt_flag)
                    pred_flags.append(pred_flag)
                    num_valid_gt += sum(gt_flag == 0)
                    if iou.size > 0:
                        accum_scores, accum_iou, accum_pos, accum_rot = accumulate_scores(
                            gt_anno["boxes_3d"],
                            pred_anno["boxes_3d"],
                            iou,
                            pred_score,
                            gt_flag,
                            pred_flag,
                            iou_threshold=iou_threshold,
                        )
                    else:
                        # continue
                        print("iou is empty")
                        accum_scores, accum_iou, accum_pos, accum_rot = (
                            np.array([]),
                            np.array([]),
                            np.array([]),
                            np.array([]),
                        )
                    accum_all_scores.append(accum_scores)
                    accum_all_ious.append(accum_iou)
                    accum_all_pos.append(accum_pos)
                    accum_all_rot.append(accum_rot)
                else:
                    print("No iou found in data. Use an iou threshold of e.g. iou=0.7")

            all_scores = np.concatenate(accum_all_scores, axis=0)
            all_ious = np.concatenate(accum_all_ious, axis=0)
            all_pos = np.concatenate(accum_all_pos, axis=0)
            all_rot = np.concatenate(accum_all_rot, axis=0)
            thresholds = get_thresholds(all_scores, num_valid_gt, num_pr_points=num_pr_points)

            ### compute avg iou, pos/rot error ###
            iou_3d[cls_idx, diff_idx] = np.average(all_ious) if len(all_ious) else 0.0
            pos_err[cls_idx, diff_idx] = np.average(all_pos) if len(all_pos) else 0.0
            rot_err[cls_idx, diff_idx] = np.average(all_rot) if len(all_rot) else 0.0

            ### compute tp/fp/fn ###
            confusion_matrix = np.zeros([len(thresholds), 3])  # only record tp/fp/fn
            for sample_idx in range(num_samples):
                pred_score = pred_annotation_frames[sample_idx]["score"]
                iou = ious[sample_idx]
                gt_flag, pred_flag = gt_flags[sample_idx], pred_flags[sample_idx]
                for th_idx, score_th in enumerate(thresholds):
                    if iou.size > 0:
                        tp, fp, fn = compute_statistics(
                            iou, pred_score, gt_flag, pred_flag, score_threshold=score_th, iou_threshold=iou_threshold
                        )
                        confusion_matrix[th_idx, 0] += tp
                        confusion_matrix[th_idx, 1] += fp
                        confusion_matrix[th_idx, 2] += fn

            ### draw p-r curve ###
            for th_idx in range(len(thresholds)):
                recall[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / (
                    confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 2]
                )
                precision[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / (
                    confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 1]
                )

            for th_idx in range(len(thresholds)):
                precision[cls_idx, diff_idx, th_idx] = np.max(precision[cls_idx, diff_idx, th_idx:], axis=-1)
                recall[cls_idx, diff_idx, th_idx] = np.max(recall[cls_idx, diff_idx, th_idx:], axis=-1)

    AP = 0

    for i in range(1, precision.shape[-1]):
        AP += precision[..., i]
    AP = AP / num_pr_points * 100

    ret_str = "|%-18s|" % "Classes"
    ret_str += "%-12s|" % "Precision"
    ret_str += "%-12s|" % "Recall"
    ret_str += "\n"
    precision_values = []
    recall_values = []
    for idx, cur_class in enumerate(classes):
        ret_str += "|%-18s|" % cur_class
        ret_str += "%-12.2f|" % (np.mean(precision[idx], axis=-1)[0] * 100)
        precision_values.append(np.mean(precision[idx], axis=-1)[0] * 100)
        ret_str += "%-12.2f|" % (np.mean(recall[idx], axis=-1)[0] * 100)
        recall_values.append(np.mean(recall[idx], axis=-1)[0] * 100)
        ret_str += "\n"
    ret_dict = {}
    ret_dict["precision"] = np.mean(precision_values)
    ret_dict["recall"] = np.mean(recall_values)


    ret_str = "|AP@%-15s|" % (str(num_pr_points))
    for diff_type in difficulty_types:
        ret_str += "%-15s|" % diff_type
    ret_str += "%-20s|" % "Occurrence (pred/gt)"
    ret_str += "%-10s|" % "IOU_3D"
    ret_str += "%-10s|" % "Pos-RMSE"
    ret_str += "%-10s|" % "Rot-RMSE"
    ret_str += "\n"

    for cls_idx, cur_class in enumerate(classes):
        ret_str += "|%-18s|" % cur_class
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            key = "AP_" + cur_class + "/" + diff_type
            # TODO: Adopt correction of TP=0, FP=0 -> AP = 0 for all difficulty
            # types by counting occurrence individually for each difficulty type
            # if pred_class_occurrence[cur_class] == 0 and gt_class_occurrence[cur_class] == 0:
            #     AP[cls_idx, diff_idx] = 100
            ap_score = AP[cls_idx, diff_idx]
            ret_dict[key] = ap_score
            ret_str += "%-15.2f|" % ap_score
        ret_str += "%-20s|" % (str(pred_class_occurrence[cur_class]) + "/" + str(gt_class_occurrence[cur_class]))
        ret_str += "%-10.2f|" % np.average(iou_3d[cls_idx].flatten())
        ret_str += "%-10.2f|" % np.average(pos_err[cls_idx].flatten())
        ret_str += "%-10.2f|" % np.average(rot_err[cls_idx].flatten())
        ret_str += "\n"
    mAP = np.mean(AP, axis=0)
    
    ret_str += "|%-18s|" % "mAP"
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        key = "AP_mean" + "/" + diff_type
        ap_score = mAP[diff_idx]
        ret_dict[key] = ap_score
        ret_str += "%-15.2f|" % ap_score
    ret_dict["3d_map"] = mAP[0] # 3D mAP for distance [0 - inf]
    ret_str += "%-20s|" % (
        str(np.sum(list(pred_class_occurrence.values())))
        + "/"
        + str(np.sum(list(gt_class_occurrence.values())))
        + " (Total)"
    )
    ret_str += "%-10.2f|" % np.average(iou_3d.flatten())
    ret_str += "%-10.2f|" % np.average(pos_err.flatten())
    ret_str += "%-10.2f|" % np.average(rot_err.flatten())
    ret_str += "\n"

    ####################
    ## pretty print (for excel sheet)
    ####################
    ret_header_str = "Class,Precision,Recall,AP_overall,distance_0_40,distance_40_50,distance_50_64,Occurrence (pred/gt),IOU_3D,Pos-RMSE,Rot-RMSE\n"
    for cls_idx, cur_class in enumerate(classes):
        ret_header_str += f"{cur_class},{np.mean(precision[cls_idx], axis=-1)[0] * 100:.2f},{np.mean(recall[cls_idx], axis=-1)[0] * 100:.2f},"
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            ap_score = AP[cls_idx, diff_idx]
            ret_header_str += f"{ap_score:.2f},"
        ret_header_str += f"{pred_class_occurrence[cur_class]}/{gt_class_occurrence[cur_class]},"
        ret_header_str += f"{np.average(iou_3d[cls_idx].flatten()):.2f},"
        ret_header_str += f"{np.average(pos_err[cls_idx].flatten()):.2f},"
        ret_header_str += f"{np.average(rot_err[cls_idx].flatten()):.2f}\n"

    ret_header_str += "mAP,,,"
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        ap_score = mAP[diff_idx]
        ret_header_str += f"{ap_score:.2f},"
    ret_header_str += f"{np.sum(list(pred_class_occurrence.values()))}/{np.sum(list(gt_class_occurrence.values()))},"
    ret_header_str += f"{np.average(iou_3d.flatten()):.2f},"
    ret_header_str += f"{np.average(pos_err.flatten()):.2f},"
    ret_header_str += f"{np.average(rot_err.flatten()):.2f}\n"

    ret_dict["3d_iou"] = np.average(iou_3d.flatten())
    ret_dict["position_rmse"] = np.average(pos_err.flatten())
    ret_dict["rotation_rmse"] = np.average(rot_err.flatten())

    # print pretty table results for excel sheet
    # print(ret_header_str)
    ####################
    return ret_str, ret_dict


@numba.jit(nopython=True)
def get_thresholds(scores, num_gt, num_pr_points):
    eps = 1e-6
    scores.sort()
    scores = scores[::-1]
    recall_level = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (r_recall + l_recall < 2 * recall_level) and i < (len(scores) - 1):
            continue
        thresholds.append(score)
        recall_level += 1 / num_pr_points

        while r_recall + l_recall + eps > 2 * recall_level:
            thresholds.append(score)
            recall_level += 1 / num_pr_points
    return thresholds


# TODO: only use annotation in live mode (comment for debugging)
@numba.jit(nopython=True)
def accumulate_scores(gt_shapes, pred_shapes, iou, pred_scores, gt_flag, pred_flag, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assert num_gt == len(gt_shapes)
    assert num_pred == len(pred_shapes)

    assigned = np.full(num_pred, False)
    accum_scores = np.zeros(num_gt)
    accum_ious = np.zeros(num_gt)
    accum_pos_rmse = np.zeros(num_gt)
    accum_rot_rmse = np.zeros(num_gt)
    accum_idx = 0
    for gt_id, gt_shape in enumerate(gt_shapes):
        if gt_flag[gt_id] == -1:  # not the same class
            continue
        det_idx = -1
        detected_score = -1
        det_iou = -1
        det_pos_rmse = -1
        det_rot_rmse = -1
        for pred_id, pred_shape in enumerate(pred_shapes):
            if pred_flag[pred_id] == -1:  # not the same class
                continue
            if assigned[pred_id]:
                continue
            iou_ij = iou[gt_id, pred_id]
            pred_score = pred_scores[pred_id]
            if (iou_ij > iou_threshold) and (pred_score > detected_score) and (iou_ij > det_iou):
                det_idx = pred_id
                detected_score = pred_score
                det_iou = iou_ij
                det_pos_rmse = np.linalg.norm(gt_shape[:3] - pred_shape[:3])
                det_rot_rmse = abs(gt_shape[6] - pred_shape[6]) % math.pi
                if det_rot_rmse > math.pi * 0.5:
                    det_rot_rmse = det_rot_rmse - math.pi * 0.5

        if (detected_score == -1) and (gt_flag[gt_id] == 0):  # false negative
            pass
        elif (detected_score != -1) and (gt_flag[gt_id] == 1 or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected_score != -1:  # true positive
            accum_scores[accum_idx] = pred_scores[det_idx]
            accum_ious[accum_idx] = det_iou
            accum_pos_rmse[accum_idx] = det_pos_rmse
            accum_rot_rmse[accum_idx] = det_rot_rmse
            accum_idx += 1
            assigned[det_idx] = True

    return accum_scores[:accum_idx], accum_ious[:accum_idx], accum_pos_rmse[:accum_idx], accum_rot_rmse[:accum_idx]


@numba.jit(nopython=True)
def compute_statistics(iou, pred_scores, gt_flag, pred_flag, score_threshold, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    under_threshold = pred_scores < score_threshold

    tp, fp, fn = 0, 0, 0
    for i in range(num_gt):
        if gt_flag[i] == -1:  # different classes
            continue
        det_idx = -1
        detected = False
        best_matched_iou = 0
        gt_assigned_to_ignore = False

        for j in range(num_pred):
            if pred_flag[j] == -1:  # different classes
                continue
            if assigned[j]:  # already assigned to other GT
                continue
            if under_threshold[j]:  # compute only boxes above threshold
                continue
            iou_ij = iou[i, j]
            if (iou_ij > iou_threshold) and (iou_ij > best_matched_iou or gt_assigned_to_ignore) and pred_flag[j] == 0:
                best_matched_iou = iou_ij
                det_idx = j
                detected = True
                gt_assigned_to_ignore = False
            elif (iou_ij > iou_threshold) and (not detected) and pred_flag[j] == 1:
                det_idx = j
                detected = True
                gt_assigned_to_ignore = True

        if (not detected) and gt_flag[i] == 0:  # false negative
            fn += 1
        elif detected and (gt_flag[i] == 1 or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected:  # true positive
            tp += 1
            assigned[det_idx] = True

    for j in range(num_pred):
        if not (assigned[j] or pred_flag[j] == -1 or pred_flag[j] == 1 or under_threshold[j]):
            fp += 1

    return tp, fp, fn


def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level, class_name):
    """
    Filter data by class name and difficulty

    Args:
        gt_anno:
        pred_anno:
        difficulty_mode:
        difficulty_level:
        class_name:

    Returns:
        gt_flags/pred_flags:
            1 : same class but ignored with different difficulty levels
            0 : accepted
           -1 : rejected with different classes
    """
    num_gt = len(gt_anno["name"])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if num_gt > 0:
        reject = gt_anno["name"] != class_name
        gt_flag[reject] = -1
    num_pred = len(pred_anno["name"])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if num_pred > 0:
        reject = pred_anno["name"] != class_name
        pred_flag[reject] = -1

    if difficulty_mode == "OVERALL":
        ignore = overall_filter(gt_anno["boxes_3d"], difficulty_level)
        gt_flag[ignore] = 1
        ignore = overall_filter(pred_anno["boxes_3d"], difficulty_level)
        pred_flag[ignore] = 1

    return gt_flag, pred_flag


def iou3d_kernel(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    iou3d = intersection_3d / union_3d
    return iou3d


def iou3d_kernel_with_heading(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    iou3d = intersection_3d / union_3d

    # rotation orientation filtering
    diff_rot = gt_boxes[:, [6]] - pred_boxes[:, [6]].T
    diff_rot = np.abs(diff_rot)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[diff_rot >= np.pi]  # constrain to [0-pi]
    iou3d[diff_rot > np.pi / 2] = 0  # unmatched if diff_rot > 90
    return iou3d


def rotate_iou_kernel_eval(gt_boxes, pred_boxes):
    iou3d_cpu = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
    return iou3d_cpu


def compute_iou3d(gt_annos, pred_annos, split_parts, with_heading):
    """
    Compute iou3d of all samples by parts

    Args:
        with_heading: filter with heading
        gt_annos: list of dicts for each sample
        pred_annos:
        split_parts: for part-based iou computation

    Returns:
        ious: list of iou arrays for each sample
    """
    gt_num_per_sample = np.stack([len(anno["name"]) for anno in gt_annos], 0)
    pred_num_per_sample = np.stack([len(anno["name"]) for anno in pred_annos], 0)
    ious = []
    sample_idx = 0
    for num_part_samples in split_parts:
        gt_annos_part = gt_annos[sample_idx : sample_idx + num_part_samples]
        pred_annos_part = pred_annos[sample_idx : sample_idx + num_part_samples]

        gt_boxes = np.concatenate([anno["boxes_3d"] for anno in gt_annos_part], 0)
        pred_boxes = np.concatenate([anno["boxes_3d"] for anno in pred_annos_part], 0)

        if with_heading:
            iou3d_part = iou3d_kernel_with_heading(gt_boxes, pred_boxes)
        else:
            iou3d_part = iou3d_kernel(gt_boxes, pred_boxes)

        gt_num_idx, pred_num_idx = 0, 0
        for idx in range(num_part_samples):
            gt_box_num = gt_num_per_sample[sample_idx + idx]
            pred_box_num = pred_num_per_sample[sample_idx + idx]
            ious.append(iou3d_part[gt_num_idx : gt_num_idx + gt_box_num, pred_num_idx : pred_num_idx + pred_box_num])
            gt_num_idx += gt_box_num
            pred_num_idx += pred_box_num
        sample_idx += num_part_samples
    return ious


def compute_iou3d_cpu(gt_annos, pred_annos):
    ious = []
    gt_num = len(gt_annos)
    for i in range(gt_num):
        gt_boxes = gt_annos[i]["boxes_3d"]
        pred_boxes = pred_annos[i]["boxes_3d"]

        iou3d_part = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
        ious.append(iou3d_part)
    return ious


def get_attribute_by_name(attribute_list, attribute_name):
    for attribute in attribute_list:
        if attribute["name"] == attribute_name:
            return attribute
    return None


def load_3d_boxes(input_file_path):
    labels_list = []
    name = []
    boxes_3d = []
    num_points_in_gt = []
    json_data = json.load(open(input_file_path))
    scores = []
    if "openlabel" in json_data:
        for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
            if len(frame_obj["objects"].items()) == 0:
                print("no detections in frame: {}".format(input_file_path))
                continue
            for object_id, label in frame_obj["objects"].items():
                # Dataset in ASAM OpenLABEL format
                category = label["object_data"]["type"]
                l = float(label["object_data"]["cuboid"]["val"][7])
                w = float(label["object_data"]["cuboid"]["val"][8])
                h = float(label["object_data"]["cuboid"]["val"][9])
                quat_x = float(label["object_data"]["cuboid"]["val"][3])
                quat_y = float(label["object_data"]["cuboid"]["val"][4])
                quat_z = float(label["object_data"]["cuboid"]["val"][5])
                quat_w = float(label["object_data"]["cuboid"]["val"][6])
                if np.linalg.norm([quat_x, quat_y, quat_z, quat_w]) == 0.0:
                    continue

                #rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler()
                # convert quaternion to euler angle
                rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("zyx")[0]
                position_3d = [
                    float(label["object_data"]["cuboid"]["val"][0]),
                    float(label["object_data"]["cuboid"]["val"][1]),
                    float(label["object_data"]["cuboid"]["val"][2]),  # - h / 2  # To avoid floating bounding boxes
                ]

                attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "num_points")
                num_points = 0
                if attribute is not None:
                    num_points = int(float(attribute["val"]))
                
                # Specify how many minimum points there should be before a label is included.
                #if num_points >= 5:
                name.append(category.upper())
                boxes_3d.append(np.hstack((position_3d, l, w, h, rotation_yaw)))
                num_points_in_gt.append(num_points)

                attribute = get_attribute_by_name(label["object_data"]["cuboid"]["attributes"]["num"], "score")
                if attribute is not None:
                    score = attribute["val"]
                    scores.append(score)
       
        label_dict = {
            "name": np.array(name),
            "boxes_3d": np.array(boxes_3d),
            "num_points_in_gt": np.array(num_points_in_gt),
            "score": np.array(scores),
        }
        labels_list.append(label_dict)
    return labels_list





def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: File path to test_annotation_file on the server
        `user_submission_file`: File path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    object_min_points = 5
    classes = [
        "CAR",
        "TRUCK",
        "TRAILER",
        "VAN",
        "MOTORCYCLE",
        "BUS",
        "PEDESTRIAN",
        "BICYCLE",
        "EMERGENCY_VEHICLE",
        "OTHER",
    ]
    gt_data = load_3d_boxes(test_annotation_file)
    pred_data = load_3d_boxes(user_submission_file)


    result_str, result_dict = get_evaluation_results(
        gt_data,
        pred_data,
        classes,
        difficulty_mode="OVERALL",
    )
    output = {}

    
    print("Evaluating for Test Phase")
    # TODO: check keys
    output["result"] = [
        {
            "test_split": {
                "Precision": result_dict["precision"],
                "Recall": result_dict["recall"],
                "3D_IoU": result_dict["3d_iou"],
                "Position_RMSE": result_dict["position_rmse"],
                "Rotation_RMSE": result_dict["rotation_rmse"],
                "3D_mAP": result_dict["3d_map"],
            }
        },
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]

    print("Precision: ", result_dict["precision"])
    print("Recall: ", result_dict["recall"])
    print("3D_IoU: ", result_dict["3d_iou"])
    print("Position_RMSE: ", result_dict["position_rmse"])
    print("Rotation_RMSE: ", result_dict["rotation_rmse"])
    print("3D_mAP: ", result_dict["3d_map"])

    print("Completed evaluation for Test Phase")
    return output
