import cv2
import numpy as np
from scipy.signal import butter, lfilter
from scipy.ndimage import convolve
import os


# Helper function to build a Laplacian pyramid
def build_laplacian_pyramid(im, filt1=None, filt2=None, edges='reflect'):
    if filt1 is None:
        filt1 = np.array([1, 4, 6, 4, 1]) / 16  # Default filter 'binom5'
    if filt2 is None:
        filt2 = filt1


    height = int(np.floor(np.log2(min(im.shape))))


    pyr = []
    pind = []


    for _ in range(height):
        lo = convolve(im, filt1[:, None], mode=edges)
        lo = convolve(lo, filt1[None, :], mode=edges)
        lo2 = lo[::2, ::2]  # Downsample


        up = np.zeros_like(im)
        up[::2, ::2] = lo2  # Upsample
        up = convolve(up, filt2[:, None], mode=edges)
        up = convolve(up, filt2[None, :], mode=edges)


        hi = im - up
        pyr.append(hi)
        pind.append(hi.shape)


        im = lo2


    pyr.append(im)
    pind.append(im.shape)
    return pyr, pind


# Function to reconstruct the Laplacian pyramid
def reconstruct_laplacian_pyramid(pyr, pind, filt2=None, edges='reflect'):
    if filt2 is None:
        filt2 = np.array([1, 4, 6, 4, 1]) / 16  # Default filter 'binom5'


    im = pyr[-1]
    for i in range(len(pind) - 2, -1, -1):
        up = np.zeros(pind[i])
        up[::2, ::2] = im  # Upsample
        up = convolve(up, filt2[:, None], mode=edges)
        up = convolve(up, filt2[None, :], mode=edges)


        im = up + pyr[i]


    return im


# Temporal filtering using Butterworth filters
def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b_low, a_low = butter(order, low, btype='low')
    b_high, a_high = butter(order, high, btype='low')
    lowpass = lfilter(b_low, a_low, data)
    highpass = lfilter(b_high, a_high, data)
    return lowpass - highpass


# Main function for video processing
def amplify_spatial_lpyr_temporal_butter(vid_file, out_dir, alpha, lambda_c, fl, fh, sampling_rate):
    if not os.path.exists('output_dir'):
        os.makedirs('output_dir')
    vid = cv2.VideoCapture(vid_file)
    if not vid.isOpened():
        print("Error: Could not open video file.")
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = width - (width % 2)
    height = height - (height % 2)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


    out_name = f"{out_dir}/output_alpha-{alpha}_lambda-{lambda_c}_fl-{fl}_fh-{fh}.mp4"
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        print("Error: Could not initialize video writer. Check codec or output file path.")
        return


    _, first_frame = vid.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255


    pyr, pind = build_laplacian_pyramid(first_frame[:, :, 0])
    pyr_prev = [np.copy(band) for band in pyr]
    lowpass1 = [np.copy(band) for band in pyr]
    lowpass2 = [np.copy(band) for band in pyr]


    for frame_idx in range(1, frame_count):
        ret, frame = vid.read()
        if not ret:
            print(f"Error: Could not read frame {frame_idx}")
            break
        print(f"Processing frame {frame_idx}...")


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255
        pyr, _ = build_laplacian_pyramid(frame[:, :, 0])


        # Temporal filtering
        for i in range(len(pyr)):
            lowpass1[i] = butter_bandpass_filter(pyr[i], fl, fh, sampling_rate)
            lowpass2[i] = butter_bandpass_filter(pyr_prev[i], fl, fh, sampling_rate)


        filtered = [lowpass1[i] - lowpass2[i] for i in range(len(pyr))]
        pyr_prev = [np.copy(band) for band in pyr]


        # Amplify spatial frequencies
        delta = lambda_c / 8 / (1 + alpha)
        exaggeration_factor = 2
        lambda_min = np.sqrt(height ** 2 + width ** 2) / 3


        for i, band in enumerate(filtered):
            lambda_band = lambda_min / (2 ** i)
            curr_alpha = (lambda_band / delta / 8 - 1) * exaggeration_factor
            curr_alpha = min(curr_alpha, alpha)
            filtered[i] *= curr_alpha


        # Reconstruct and save
        amplified_frame = reconstruct_laplacian_pyramid(filtered, pind)
        output_frame = frame[:, :, 0] + amplified_frame
        output_frame = np.clip(output_frame, 0, 1)


        # Combine processed luminance channel with original Cr and Cb channels
        y_channel = output_frame
        cr_channel = frame[:, :, 1]  # Cr channel from original frame
        cb_channel = frame[:, :, 2]  # Cb channel from original frame
        cr_channel = np.clip(cr_channel, 0, 1)
        cb_channel = np.clip(cb_channel, 0, 1)
        output_frame_ycrcb = np.stack((y_channel, cr_channel, cb_channel), axis=-1)


        # Convert YCrCb to BGR
        output_frame_bgr = cv2.cvtColor((output_frame_ycrcb * 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


        out.write(output_frame_bgr)


    vid.release()
    out.release()


amplify_spatial_lpyr_temporal_butter('./data/baby.mp4', 'output_dir', alpha=10, lambda_c=16, fl=0.4, fh=3, sampling_rate=30)
