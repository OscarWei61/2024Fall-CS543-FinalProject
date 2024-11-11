import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

#build laplacian pyramid for video
def laplacian_filter(frame_list, levels=3):
    filtered_result = []

    for i in range(0, frame_list.shape[0]):
        frame = frame_list[i]
        tmp = frame.copy()

        gaussian_pyramid = [tmp]
        for _ in range(levels):
            tmp = cv2.pyrDown(tmp)
            gaussian_pyramid.append(tmp)
        
        laplacian_pyramid = []
        for j in range(levels, 0, -1):
            guassian_expanded = cv2.pyrUp(gaussian_pyramid[j])
            laplacian_layer = cv2.subtract(gaussian_pyramid[j - 1], guassian_expanded)
            laplacian_pyramid.append(laplacian_layer)
        
        if i == 0:
            for k in range(levels):
                filtered_result.append(np.zeros((frame_list.shape[0], laplacian_pyramid[k].shape[0], laplacian_pyramid[k].shape[1], 3)))

        for n in range(levels):
            filtered_result[n][i] = laplacian_pyramid[n]

    return filtered_result

#reconstract video from laplacian pyramid
def reconstruct_from_frame_list(filter_tensor_list,levels=3):
    final = np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up = cv2.pyrUp(up) + filter_tensor_list[n + 1][i]
        final[i] = up
    return final

#manify motion
def motion_magnification(file_name, lowcut, highcut, levels=3, amplification_factor=30):
    # Load video
    capture = cv2.VideoCapture(file_name)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_list = np.zeros((frame_count, height, width, 3), dtype='float')
    i = 0
    while capture.isOpened():
        ret,frame = capture.read()
        if ret is True:
            frame = frame / 255
            frame_list[i] = frame
            i += 1
        else:
            break
    
    # Construct laplacian pyramid
    frame_list_laplacian = laplacian_filter(frame_list, levels=levels)

    # Apply low-pass Butterworth filter to find the motion with lower frequency and magnify the motion by the amplification factor
    filtered_frame_list = []
    omega = 0.5 * fps
    low = lowcut / omega
    high = highcut / omega
    order = 5
    for j in range(levels):
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_frame = signal.lfilter(b, a, frame_list_laplacian[j], axis=0)
        filtered_frame *= amplification_factor
        filtered_frame_list.append(filtered_frame)
    
    # Reconstruct video frame from filtered laplacian pyramid
    magnified_frame_list = reconstruct_from_frame_list(filtered_frame_list)
    final = frame_list + magnified_frame_list
    final *= 255

    # Save final result
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter("out.avi", fourcc, 30, (width, height), 1)
    for k in range(0, final.shape[0]):
        writer.write(cv2.convertScaleAbs(final[k]))
    writer.release()

if __name__=="__main__":
    motion_magnification("./data/face2.mp4", 0.4, 3)