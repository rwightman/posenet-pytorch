from posenet.decode import *
from posenet.constants import *
import time
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F


def within_nms_radius_fast(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)


def get_instance_score_fast(
        exist_pose_coords,
        squared_nms_radius,
        keypoint_scores, keypoint_coords):

    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)


def build_part_with_score_torch(score_threshold, local_max_radius, scores):
    lmd = 2 * local_max_radius + 1
    max_vals = F.max_pool2d(scores, lmd, stride=1, padding=1)
    max_loc = (scores == max_vals) & (scores > score_threshold)
    max_loc_idx = max_loc.nonzero()
    scores_vec = scores[max_loc]
    sort_idx = torch.argsort(scores_vec, descending=True)
    return scores_vec[sort_idx], max_loc_idx[sort_idx]


# FIXME leaving here as reference for now
# def build_part_with_score_fast(score_threshold, local_max_radius, scores):
#     parts = []
#     num_keypoints = scores.shape[0]
#     lmd = 2 * local_max_radius + 1
#
#     # NOTE it seems faster to iterate over the keypoints and perform maximum_filter
#     # on each subarray vs doing the op on the full score array with size=(lmd, lmd, 1)
#     for keypoint_id in range(num_keypoints):
#         kp_scores = scores[keypoint_id, :, :].copy()
#         kp_scores[kp_scores < score_threshold] = 0.
#         max_vals = ndi.maximum_filter(kp_scores, size=lmd, mode='constant')
#         max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)
#         max_loc_idx = max_loc.nonzero()
#         for y, x in zip(*max_loc_idx):
#             parts.append((
#                 scores[keypoint_id, y, x],
#                 keypoint_id,
#                 np.array((y, x))
#             ))
#
#    return parts


def decode_multiple_poses(
        scores, offsets, displacements_fwd, displacements_bwd, output_stride,
        max_pose_detections=10, score_threshold=0.5, nms_radius=20, min_pose_score=0.5):

    # perform part scoring step on GPU as it's expensive
    # TODO determine how much more of this would be worth performing on the GPU
    part_scores, part_idx = build_part_with_score_torch(score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    part_scores = part_scores.cpu().numpy()
    part_idx = part_idx.cpu().numpy()

    scores = scores.cpu().numpy()
    height = scores.shape[1]
    width = scores.shape[2]
    # change dimensions from (x, h, w) to (x//2, h, w, 2) to allow return of complete coord array
    offsets = offsets.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_fwd = displacements_fwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_bwd = displacements_bwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))

    squared_nms_radius = nms_radius ** 2
    pose_count = 0
    pose_scores = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

    for root_score, (root_id, root_coord_y, root_coord_x) in zip(part_scores, part_idx):
        root_coord = np.array([root_coord_y, root_coord_x])
        root_image_coords = root_coord * output_stride + offsets[root_id, root_coord_y, root_coord_x]

        if within_nms_radius_fast(
                pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords):
            continue

        keypoint_scores, keypoint_coords = decode_pose(
            root_score, root_id, root_image_coords,
            scores, offsets, output_stride,
            displacements_fwd, displacements_bwd)

        pose_score = get_instance_score_fast(
            pose_keypoint_coords[:pose_count, :, :], squared_nms_radius, keypoint_scores, keypoint_coords)

        # NOTE this isn't in the original implementation, but it appears that by initially ordering by
        # part scores, and having a max # of detections, we can end up populating the returned poses with
        # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
        # Set min_pose_score to 0. to revert to original behaviour
        if min_pose_score == 0. or pose_score >= min_pose_score:
            pose_scores[pose_count] = pose_score
            pose_keypoint_scores[pose_count, :] = keypoint_scores
            pose_keypoint_coords[pose_count, :, :] = keypoint_coords
            pose_count += 1

        if pose_count >= max_pose_detections:
            break

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords
