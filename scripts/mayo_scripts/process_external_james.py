import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot as plt

def segment_cardiac_echo(frame, standardized=False, ecg=False, threshold_min=29):
    """
    code to segment just the ultrasound image of the cardiac ultrasound frames
    input:
        frame: the raw ultrasound array frame
        standardized: option for a standardized dataset input
    output:
        frame_crop: segmented cardiac ultrasound
    """

    if standardized:  # if it's a standarized ultrasound, just crop a rectangle
        # adding a normalization step in the middle so that the ultrasound isn't as dark
        frame_h = int(frame.shape[0] * 0.1)
        frame_w = int(frame.shape[1] * 0.25)
        if ecg:
            frame_crop = frame[frame_h:, frame_w:-frame_w]
        else:
            frame_crop = frame[frame_h:-frame_h, frame_w:-frame_w]
    else:  # if not standardized, use James' code - need some cleaning up
        # remove the top 10% to avoid the bar
        # frame_h = int(frame.shape[0] * 0.1)
        # frame = frame[frame_h:, :]
        # James' code
        ret2, threshhold = cv2.threshold(frame, threshold_min, 255, 0)
        contours, hierarchy = cv2.findContours(threshhold, 1, 2)  # need to change this so it picks up more things
        # Approx contour
        cnt = contours[0]
        largest = cv2.contourArea(cnt)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = contours[0]
        # Central points and area
        moments = cv2.moments(cnt)
        cent_x = int(moments['m10'] / moments['m00'])
        cent_y = int(moments['m01'] / moments['m00'])
        shape_area = cv2.contourArea(cnt)
        shape_perim = cv2.arcLength(cnt, True)
        epsilon = 0.01 * shape_perim
        approximation = cv2.approxPolyDP(cnt, epsilon, True)
        convex_hull = cv2.convexHull(cnt)
        contour_mask = np.zeros(frame.shape, np.uint8)
        contour_mask = cv2.drawContours(contour_mask, [convex_hull], 0, 1, -1)  # changed James' code so that multiplying works
        frame_crop = frame * contour_mask

    return contour_mask

# # python process_external_james.py

# # set up paths
# root_paths = ['/media/Datacenter_storage/James_ECHO_LVH/External_Validation/AL_NP', 
#               '/media/Datacenter_storage/James_ECHO_LVH/External_Validation/CP_NP',
#               '/media/Datacenter_storage/James_ECHO_LVH/External_Validation/NL_NP']
              
# save_base = '/media/Datacenter_storage/CardioOncology/AP4_crop_process/'

# # set up a list to make the dataframe
# paths = []
# pt_ids = []
# cls_list = []
# numerical_labels = []

# # for each class
# for i in root_paths:
#     # make new save path
#     cls = i.split('/')[-1]
#     cls_save_dir = os.path.join(save_base, i.split('/')[-1])
#     os.makedirs(cls_save_dir, exist_ok=True)
#     pt_paths = [os.path.join(i, k) for k in os.listdir(i)]
#     # for each patient
#     for j in pt_paths:
#         pt_id = j.split('/')[-1]
#         files = [os.path.join(j, h) for h in os.listdir(j)]
#         # for each file
#         for fp in files:
#             # process the code (using my rectangle crop)
#             orig_img = sitk.ReadImage(fp)
#             frame = sitk.GetArrayFromImage(orig_img)
#             # process with James' code
#             frame_center = segment_cardiac_echo(frame, standardized=False, ecg=True, threshold_min=10)
#             # convert into PIL image to save
#             img = Image.fromarray(frame_center)
#             save_img_path = os.path.join(cls_save_dir, fp.split('/')[-1])
#             img.save(save_img_path)
#             # append to list all the relevant information
#             paths.append(save_img_path)
#             pt_ids.append(pt_id)
#             cls_list.append(cls)
#             if cls == 'AL_NP':
#                 numerical_label = 0
#                 numerical_labels.append(numerical_label)
#             elif cls == 'CP_NP':
#                 numerical_label = 1
#                 numerical_labels.append(numerical_label)
#             else:
#                 numerical_label = 2
#                 numerical_labels.append(numerical_label)
#             print('saved: {}\t{}\t{}\t{}'.format(save_img_path, pt_id, cls, numerical_label), flush=True)

# df = pd.DataFrame({'path': paths, 'pt_id': pt_ids, 'class': cls_list})
# df.to_csv('/media/Datacenter_storage/CardioOncology/External_processed_Jamesprocessed/external_data.csv')