import os
import cv2
import argparse
import numpy as np

def calc_mean_std_np(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.shape
    C = size[-1]
    feat_var = np.var(feat.reshape(-1, C), axis=0) + eps
    feat_std = np.sqrt(feat_var).reshape(1, 1, C)
    feat_mean = np.mean(feat.reshape(-1, C), axis=0).reshape(1, 1, C)
    return feat_mean, feat_std

def color_stat_calculation(in_path):
    if os.path.isdir(in_path):
        img_list = os.listdir(in_path)
        img_list = [os.path.join(in_path, f) for f in img_list]
    else:
        img_list = [in_path]
    
    total_mean = np.zeros((1,1,3))
    total_std = np.zeros((1,1,3))

    for img_name in img_list:
        img = cv2.imread(img_name).astype(np.float32)[:, :, [2, 1, 0]]/ 255.
        img = (img-0.5)/0.5 # rgb, [-1,1]
        gt_mean, gt_std = calc_mean_std_np(img)
        total_mean += gt_mean
        total_std += gt_std
    avg_mean = total_mean / len(img_list)
    avg_std = total_std / len(img_list)
    print(f"Color statistics for {in_path} are as follows:")
    print(f'- avg mean: {avg_mean}')
    print(f'- avg std: {avg_std}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', type=str, help="Input folder or image for color statistics calculation.")
    args = parser.parse_args()
    color_stat_calculation(args.in_path)