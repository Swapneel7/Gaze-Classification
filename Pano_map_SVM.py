# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import imutils
import cv2
import csv
from datetime import datetime, timezone
from scipy.cluster.vq import kmeans, whiten

THRESH_VAL = 90
THRESH_TYPE = cv2.THRESH_BINARY_INV
AD_THRESH = cv2.ADAPTIVE_THRESH_MEAN_C
FRAME_SZ = 560
BLUR = 3
BLUR_SZ = (BLUR, BLUR)
CONT_RNG = (0, 2)
MERGE_WEIGHT = (0.95, 0.05)
VID_START = 0
VID_END = 260000
SEAM_THRESH = 100
NEW_REF_THRESH = 0.03
REF_MATCH_THRESH = 0.7

# fisheye correction params
DIM = (FRAME_SZ, int(0.75 * FRAME_SZ))
K = np.array([[869.729, 0, 487.388], [0, 869.203, 358.695], [0, 0, 1]])
D = np.array([[0.1024684, -0.8466225, 0.000369681421, -0.00057014884, 1.43796465]])


# Remove distortion from image given camera params calculated earlier
def remove_distortion(img, h, w, new_mtx):
    return cv2.undistort(img, K, D, None, new_mtx)


# Generate a blurriness value for an image
def blurriness(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    m, s = cv2.meanStdDev(lap)
    return (s[0] ** 2)[0]


if __name__ == '__main__':
    # Params
    vid_file = 'recording.avi'
    ref_img_file = 'ref_img.jpg'
    data_file = 'RawETData.csv'
    homCache = np.identity(3)

    # Video feed
    vidcap = cv2.VideoCapture('D:\\Users\\svekhande\\downloads\\10.20.P01 - hdd-3-recording\\' + vid_file)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, VID_START)

    # Load eye tracking data
    data = open('C:\\Users\\svekhand\\Downloads\\' + data_file, 'r')
    reader = csv.reader(data)
    headers = next(data, None)
    print(datetime.strptime('00:00:00:000', '%H:%M:%S:%f').replace(tzinfo=timezone.utc).timestamp())
    start_time = datetime.strptime('00:00:00:000', '%H:%M:%S:%f').replace(tzinfo=timezone.utc).timestamp() * 1000
    points = [(float(row[20]), float(row[21]), datetime.strptime(row[18], '%H:%M:%S:%f'),
               datetime.strptime(row[19], '%H:%M:%S:%f')) for row in reader if
              row[10] == 'Visual Intake' and row[0] == 'Trial001']
    x_pts = [p[0] for p in points if p[2].replace(tzinfo=timezone.utc).timestamp() * 1000 - start_time > VID_START]
    y_pts = [p[1] for p in points if p[2].replace(tzinfo=timezone.utc).timestamp() * 1000 - start_time > VID_START]
    start_times = [p[2].replace(tzinfo=timezone.utc).timestamp() * 1000 - start_time for p in points if
                   p[2].replace(tzinfo=timezone.utc).timestamp() * 1000 - start_time >= VID_START]
    end_times = [p[3].replace(tzinfo=timezone.utc).timestamp() * 1000 - start_time for p in points if
                 p[2].replace(tzinfo=timezone.utc).timestamp() * 1000 - start_time >= VID_START]
    st_ti = iter(start_times)
    ti = iter(end_times)
    xi = iter(x_pts)
    yi = iter(y_pts)
    gaze_st_ts = next(st_ti, -1)
    gaze_ts = next(ti, -1)
    gaze_x = next(xi)
    gaze_y = next(yi)
    # pano_pts = np.zeros(len(points))
    pano_pts = []
    gaze_times = []
    # print(times)
    # print(min(x_pts))
    # print(max(x_pts))
    # print(min(y_pts))
    # print(max(y_pts))

    # Initialize feature detector and matcher
    orb = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=False)

    # Empty image that will become panorama built from video
    pano = np.zeros([1080, 1920, 3], dtype=np.uint8)

    # Initialize panorama with first frame
    s1, frame1_c = vidcap.read()
    RATIO = FRAME_SZ / frame1_c.shape[1]
    frame1_c = imutils.resize(frame1_c, width=FRAME_SZ)

    # Get undistortion parameters for use with each frame
    h, w = frame1_c.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    frame1 = remove_distortion(frame1_c, h, w, new_mtx)
    ts1 = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    kp1, des1 = orb.detectAndCompute(frame1, None)

    # Initialize lists used for storing reference frames
    ref_frames, ref_kps, ref_dess = ([frame1.copy()], [kp1.copy()], [des1.copy()])
    ref_homs = [np.eye(3)]

    # Extra value for restraining number of homography resets
    rflag = 0

    # Iterate through video frames
    while True:

        s2, frame2_c = vidcap.read()
        frame2_c = imutils.resize(frame2_c, width=FRAME_SZ)
        frame2 = remove_distortion(frame2_c, h, w, new_mtx)
        blur2 = blurriness(frame2)
        ts2 = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        kp2, des2 = orb.detectAndCompute(frame2, None)

        # Get matches between current and last frame
        matches = bf.knnMatch(des1, des2, k=2)

        # Get matches for all currently stored reference frames
        ref_matches = [bf.knnMatch(ref_des, des2, k=2) for ref_des in ref_dess]

        # Remove outliers with Lowe's ratio test
        matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
        ref_matches = [[m for m, n in ref_match if m.distance < 0.5 * n.distance] for ref_match in ref_matches]

        # Find best match among reference images
        ref_match_fits = [len(r) for r in ref_matches]
        best_match_idx = ref_match_fits.index(max(ref_match_fits))
        best_match = ref_matches[best_match_idx]
        best_kp = ref_kps[best_match_idx]
        best_des = ref_dess[best_match_idx]

        # Check for adequate number of matches
        if len(matches) > 10:


            # Point pairs between frames
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            src_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Point pairs between current frame and best reference match
            ref_dst = np.float32([best_kp[m.queryIdx].pt for m in best_match])
            ref_src = np.float32([kp2[m.trainIdx].pt for m in best_match])

            # Get homography between current and previous video frame
            hom, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                           maxIters=5000)
            # print('%d, %d -- %d -- %f' % (len(matches), len(best_match), rflag, blur2))

            # Check if addition of new reference frame is needed
            if len(best_match) < NEW_REF_THRESH * len(matches) and blur2 >= 450:
                # print('ADDING NEW REF FRAME')
                ref_frames += [frame2.copy()]
                ref_kps += [kp2.copy()]
                ref_dess += [des2.copy()]
                ref_homs += [homCache]

            # Check if reference is suitable replacement
            # for between-frame homography
            if len(best_match) >= REF_MATCH_THRESH * len(matches) and rflag <= 0:
                ref_hom, ref_mask = cv2.findHomography(ref_src, ref_dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                                       maxIters=5000)
                # print('matching reference frame %d' % (best_match_idx))
                if ref_hom is not None:
                    homCache = ref_hom.dot(ref_homs[best_match_idx])
                else:
                    homCache = hom.dot(homCache)
                rflag = 30

                # Update best ref frame to be this frame
                # if len(best_match) >= 0.8 * len(matches):
                ref_frames[best_match_idx] = frame2.copy()
                ref_kps[best_match_idx] = kp2.copy()
                ref_dess[best_match_idx] = des2.copy()
                ref_homs[best_match_idx] = homCache
            else:
                homCache = hom.dot(homCache)
                rflag -= 1

            result = cv2.warpPerspective(frame2, homCache, (pano.shape[1], pano.shape[0]))

            # Add temporally relevant gaze fixations
            while gaze_ts <= ts2:
                # print('adding gaze')
                # print(RATIO)
                p = np.array([gaze_x * RATIO, gaze_y * RATIO, 1], dtype=np.float32)
                p = homCache.dot(p)
                pano_pts += [cv2.KeyPoint(p[0] / p[2], p[1] / p[2], 4)]
                gaze_times += [gaze_ts - gaze_st_ts]
                # pano_pts += [cv2.KeyPoint(p[0]*RATIO, p[1]*RATIO, 4)]
                # print(p)
                gaze_st_ts = next(st_ti)
                gaze_ts = next(ti)
                gaze_x = next(xi)
                gaze_y = next(yi)
                if gaze_ts > VID_END:
                    break

            if gaze_ts > VID_END:
                break

            try:
                to_add = cv2.addWeighted(pano, MERGE_WEIGHT[0], result, MERGE_WEIGHT[1], 0)
                # to_add = result
                np.copyto(pano, to_add, where=(result > 0).astype(bool))
            except:
                print('busted')

            # Draw mapped gaze points onto pano
            img_kp = cv2.drawKeypoints(pano, pano_pts, outImage=np.array([]), color=(0, 255, 0))
            cv2.imshow('kp', img_kp)
            # cv2.imshow('pano', pano)

            # Store current frame as previous frame for next comparison
            frame1 = frame2.copy()
            kp1, des1 = (kp2.copy(), des2.copy())
            # kp1, des1 = orb.detectAndCompute(frame1, None)
            ts1 = ts2
        else:
            # frame1 = frame2.copy()
            # frame1_h = frame2_h.copy()
            # kp1, des1 = (kp2.copy(), des2.copy())
            # ts1 = ts2
            continue

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        # For changing values of D
        if k == ord('u'):
            homCache[0, 0] = 1
        if k == ord('i'):
            homCache[0, 1] = 1
        if k == ord('o'):
            homCache[0, 2] = 1
        if k == ord('p'):
            homCache[1, 0] = 1
        if k == ord('h'):
            homCache[1, 1] = 1
        if k == ord('j'):
            homCache[1, 2] = 1
        if k == ord('k'):
            homCache[2, 0] = 1
        if k == ord('l'):
            homCache[2, 1] = 1

        # For changing values of K
        if k == ord('t'):
            K[0, 0] += 50
            print(K[0])
        if k == ord('y'):
            K[1, 1] += 50
            print(K[1])
        if k == ord('f'):
            K[0, 0] -= 50
            print(K[0])
        if k == ord('g'):
            K[1, 1] -= 50
            print(K[1])
        if k == ord('e'):
            K[0, 2] += 1
            print(K[0])
        if k == ord('s'):
            K[0, 2] -= 1
            print(K[0])
        if k == ord('r'):
            K[1, 2] += 1
            print(K[1])
        if k == ord('d'):
            K[1, 2] -= 1
            print(K[1])

    print(ts2)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setDegree(0.0)
    # svm.setGamma(0.0)
    # svm.setCoef0(0.0)
    # svm.setC(0)
    # svm.setNu(0.0)
    # svm.setP(0.0)
    # svm.setClassWeights(None)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))

    # params = dict( kernel_type = cv2.ml.SVM_LINEAR,
    #                       svm_type = cv2.ml.SVM_C_SVC,
    #                       C = 1 )
    train_pano_pts_xy = []
    samples = []
    responses = np.array([1, 0], 'int32')
    pano_pts_1 = cv2.KeyPoint(228, 312, 4);
    pano_pts_2 = cv2.KeyPoint(440, 534, 4);
    pano_pts_3 = cv2.KeyPoint(208, 324, 4);
    pano_pts_4 = cv2.KeyPoint(450, 524, 4);

    train_pano_pts_xy.append(pano_pts_1)
    train_pano_pts_xy.append(pano_pts_2)

    Train_pano_pts = np.array([p.pt for p in train_pano_pts_xy], dtype=np.float32)
    #predictions=[]
    # samples.append(pano_pts_3)
    pano_pts_xy = np.array([p.pt for p in pano_pts], dtype=np.float32)
    svm.train(Train_pano_pts, cv2.ml.ROW_SAMPLE, responses)
    predictions=svm.predict(pano_pts_xy);
    colors = [(255, 255, 0), (0, 255, 255)]
    color_names = ['cyan', 'yellow']

    #(svm.predict_all(pano_pts_xy))
    #for s in pano_pts_xy:
      #  print(svm.predict(s))
      #  predictions.append(np.float32( [svm.predict(s)]));

    # Cluster mapped gaze points using k-means
    k = 2
    pano_pts_xy = np.array([p.pt for p in pano_pts], dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
    ret, labels, centers = cv2.kmeans(pano_pts_xy, k, None, crit, 10, 0)
    colors = [(255, 0, 0), (0, 0, 255)]
    color_names = ['blue', 'red']
    for clr, c in zip(colors, centers):
        x = int(c[0])
        y = int(c[1])
        cv2.rectangle(pano, (x - 5, y - 5), (x + 5, y + 5), clr, thickness=cv2.FILLED)

    # Change color of mapped gaze points to visually associate with k-cluster
    kp_clusters = [[] for _ in range(k)]
    kp_times = [0 for _ in range(k)]
    for i, p in enumerate(labels):
        kp_clusters[labels[i][0]] += [pano_pts[i]]
        kp_times[labels[i][0]] += gaze_times[i]

    for i, c in enumerate(kp_clusters):
        pano = cv2.drawKeypoints(pano, c, outImage=np.array([]), color=colors[i])

    print('\n\
			#############################\n\
			########## METRICS ##########\n\
			#############################\n\n')

    # Number of fixations per AOI (k-means cluster)
    print('Total run time: %.3f sec  Time spent fixating: %.3f sec' % (
    (ts2 - VID_START) / 1000.0, sum(gaze_times) / 1000.0))
    print('Fixations per AOI:\n\t' + '\n\t'.join(('Cluster %d (%s): %d/%d  Time: %.3f sec' % \
                                                  (i, color_names[i], len(c), len(pano_pts), kp_times[i] / 1000.0)) \
                                                 for i, c in enumerate(kp_clusters)))

    # Time to fixation on a specific AOI from event start point

    # Ratio of time compared to total time in a given period (e.g. critical event period)

    cv2.imshow('kp', pano)

    cv2.waitKey(0)
