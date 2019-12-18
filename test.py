import numpy as np
import cv2


# cam_urls = ['http://192.168.43.171:8080/video', 'http://192.168.43.96:8080/video']
# caps = [cv2.VideoCapture(cam_url) for cam_url in cam_urls]

def update(val=0):
    #	disparity	range	is	tuned	for	'aloe'	image	pair
    # frames = [cap.read()[1] for cap in caps]
    #         # frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    imgL = cv2.imread('imL.png')
    imgR = cv2.imread('imR.png')
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio',
                                                 'disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize',
                                                   'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff',
                                               'disparity'))
    stereo.setMinDisparity(cv2.getTrackbarPos('minDisparity', 'disparity'))
    stereo.setNumDisparities(cv2.getTrackbarPos('numDisparities', 'disparity'))
    print('computing	disparity…')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp - min_disp) / num_disp)


if __name__ == "__main__":
    window_size = 9
    min_disp = 16
    num_disp = 64 - min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 8 * 3 * window_size ** 2
    P2 = 32 * 3 * window_size ** 2
    imgL = cv2.imread('imL.png')
    imgR = cv2.imread('imR.png')
    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50,
                       update)
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize,
                       200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50,
                       update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250,
                       update)
    cv2.createTrackbar('minDisparity', 'disparity', min_disp, 250,
                       update)
    cv2.createTrackbar('numDisparities', 'disparity', num_disp, 250,
                       update)
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    update()
    cv2.waitKey()
