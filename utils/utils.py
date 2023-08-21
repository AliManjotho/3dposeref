import numpy as np

def keypoint_similarity(j_pred, j_gt, std, scale_factor):    
    dist = np.sqrt(np.sum((j_pred - j_gt)**2))
    return np.exp(-dist/2*(scale_factor**2)*(std**2))

def is_joint_good(j_pred, j_gt, std, scale_factor):
    ks = keypoint_similarity(j_pred, j_gt, std, scale_factor)
    return ks > 0.85

def is_joint_jitter(j_pred, j_gt, std, scale_factor):
    ks = keypoint_similarity(j_pred, j_gt, std, scale_factor)
    return ks > 0.5 and ks <= 0.85

def is_joint_miss(j_pred, j_gt, std, scale_factor):
    ks = keypoint_similarity(j_pred, j_gt, std, scale_factor)
    return ks < 0.5

def is_joint_inversion(j_pred, j_gt, j_symm, std, scale_factor):
    ks1 = keypoint_similarity(j_pred, j_gt, std, scale_factor)
    ks2 = keypoint_similarity(j_pred, j_symm, std, scale_factor)
    return ks1 < 0.5 and ks2 >= 0.5

def is_joint_swap(j_pred, j_gt, j_other, std, scale_factor):
    ks1 = keypoint_similarity(j_pred, j_gt, std, scale_factor)
    ks2 = keypoint_similarity(j_pred, j_other, std, scale_factor)
    return ks1 < 0.5 and ks2 >= 0.5
