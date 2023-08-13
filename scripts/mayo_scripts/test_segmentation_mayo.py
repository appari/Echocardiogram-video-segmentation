# Notebook whic
# 1. Downloads weights
# 2. Initializes model and imports weights
# 3. Performs test time evaluation of videos (already preprocessed with ConvertDICOMToAVI.ipynb)

import re
import os, os.path
from os.path import splitext
import pydicom as dicom
import numpy as np
from pydicom.uid import UID, generate_uid
import shutil
from multiprocessing import dummy as multiprocessing
import time
import subprocess
import datetime
from datetime import date
import sys
import cv2
import matplotlib.pyplot as plt
import sys
from shutil import copy
import math
import torch
import torchvision
import skimage.draw
import tqdm
import sklearn
import scipy
import echonet
import wget 


torch.cuda.empty_cache()

sys.path.append("..")
        
# Initialize and Run Segmentation model

# This part is the main loop for processing videos and performing the segmentation
# It includes loading the weights, processing each video, and saving the results
def process_videos(destinationFolder, videosFolder, DestinationForWeights, model_name):
    batch_size = 20
    
    # Initialize model
    model = torchvision.models.segmentation.__dict__[model_name](pretrained=False, aux_loss=False)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)
    
    print(f"Loading weights for {model_name} from", os.path.join(DestinationForWeights, f"{model_name}_random"))
    
    if torch.cuda.is_available():
        print("cuda is available, using original weights")
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(os.path.join(DestinationForWeights, f"{model_name}_random.pt"))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("cuda is not available, using cpu weights")
        device = torch.device("cpu")
        checkpoint = torch.load(os.path.join(DestinationForWeights, f"{model_name}_random"), map_location="cpu")
        state_dict_cpu = {k[7:]: v for (k, v) in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict_cpu)
    
    def collate_fn(x):
        x, f = zip(*x)
        i = list(map(lambda t: t.shape[1], x))
        x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
        return x, f, i
    
    mean = np.array([31.834011, 31.95879, 32.082172])
    std = np.array([48.866325, 49.137333, 49.361984])
    dataloader = torch.utils.data.DataLoader(echonet.datasets.Echo(split="external_test", external_test_location=videosFolder, target_type=["Filename"], length=None, period=1, mean=mean, std=std),
                                             batch_size=10, num_workers=0, shuffle=False, pin_memory=(device.type == "cuda"), collate_fn=collate_fn)
    
    output = destinationFolder
    model.eval()
    
    os.makedirs(os.path.join(output, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output, "size"), exist_ok=True)
    echonet.utils.latexify()
    flip = True
    with torch.no_grad():
        with open(os.path.join(output, "size.csv"), "w") as g:
            g.write("Filename,Frame,Size,ComputerSmall\n")
    
            for (x, filenames, length) in tqdm.tqdm(dataloader):
                if flip:
                    x = torch.flip(x, [3])
                y = np.concatenate([model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])
    
                start = 0
    
                x = x.numpy()
                for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                    # Extract one video and segmentation predictions
                    video = x[start:(start + offset), ...]
                    logit = y[start:(start + offset), 0, :, :]

                    # Un-normalize video
                    video *= std.reshape(1, 3, 1, 1)
                    video += mean.reshape(1, 3, 1, 1)

                    # Get frames, channels, height, and width
                    f, c, h, w = video.shape  # pylint: disable=W0612
                    assert c == 3

                    # Put two copies of the video side by side
                    video = np.concatenate((video, video), 3)

                    # If a pixel is in the segmentation, saturate blue channel
                    # Leave alone otherwise
                    video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                    # Add blank canvas under pair of videos
                    video = np.concatenate((video, np.zeros_like(video)), 2)

                    # Compute size of segmentation per frame
                    size = (logit > 0).sum((1, 2))

                    # Identify systole frames with peak detection
                    trim_min = sorted(size)[round(len(size) ** 0.05)]
                    trim_max = sorted(size)[round(len(size) ** 0.95)]
                    trim_range = trim_max - trim_min
                    systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                    # Write sizes and frames to file
                    for (frame, s) in enumerate(size):
                        # g.write("{},{},{},{}\n".format(filename, frame, s, 1 if x in  else 0))
                        g.write("{},{},{},{}\n".format(filename, frame, s, 1 if frame in systole else 0))


                    # Plot sizes
                    # fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                    # plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                    # ylim = plt.ylim()
                    # for s in systole:
                    #     plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                    # plt.ylim(ylim)
                    # plt.title(os.path.splitext(filename)[0])
                    # plt.xlabel("Seconds")
                    # plt.ylabel("Size (pixels)")
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                    # plt.close(fig)

                    # Normalize size to [0, 1]
                    size -= size.min()
                    size = size / size.max()
                    size = 1 - size

                    # Iterate the frames in this video
                    for (f, s) in enumerate(size):

                        # On all frames, mark a pixel for the size of the frame
                        video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                        if f in systole:
                            # If frame is computer-selected systole, mark with a line
                            video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                        def dash(start, stop, on=10, off=10):
                            buf = []
                            x = start
                            while x < stop:
                                buf.extend(range(x, x + on))
                                x += on
                                x += off
                            buf = np.array(buf)
                            buf = buf[buf < stop]
                            return buf
                        d = dash(115, 224)

                        # if f == large_index[i]:
                        #     # If frame is human-selected diastole, mark with green dashed line on all frames
                        #     video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                        # if f == small_index[i]:
                        #     # If frame is human-selected systole, mark with red dashed line on all frames
                        #     video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                        # Get pixels for a circle centered on the pixel
                        r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

                        # On the frame that's being shown, put a circle over the pixel
                        video[f, :, r, c] = 255.

                    # Rearrange dimensions and save
                    video = video.transpose(1, 0, 2, 3)
                    video = video.astype(np.uint8)
                    echonet.utils.savevideo(os.path.join(output, "videos", 'result_'+filename), video, 50)

                    # Move to next video
                    start += offset
            
def main():
    destinationFolder = "/home/lalith/echonet/dynamic/Output_test"
    videosFolder = "/home/lalith/scratch/converted_echo_abnormal_AP4/"
    DestinationForWeights = "/home/lalith/echonet/dynamic/EchoNetDynamic-Weights"

    # Download model weights
    if os.path.exists(DestinationForWeights):
        print("The weights are at", DestinationForWeights)
    else:
        print("Creating folder at ", DestinationForWeights, " to store weights")
        os.mkdir(DestinationForWeights)
        
    segmentationWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt'


    if not os.path.exists(os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL))):
        print("Downloading Segmentation Weights, ", segmentationWeightsURL," to ",os.path.join(DestinationForWeights,os.path.basename(segmentationWeightsURL)))
        filename = wget.download(segmentationWeightsURL, out = DestinationForWeights)
    else:
        print("Segmentation Weights already present")
        
    model_name = 'deeplabv3_resnet50'
    
    # Call the function
    process_videos(destinationFolder, videosFolder, DestinationForWeights, model_name)
 
if __name__ == '__main__':
    main()