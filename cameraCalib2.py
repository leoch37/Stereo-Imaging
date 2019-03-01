#Jared Waters and Leo Chen
#CE264 Project 2: Plane Sweeping Sterio
#Camera calibration and image undistortion
#Primary Contributor: Jared Waters
#Code adapted from: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

#---------Imports-----------
import numpy as np
import cv2 as cv
import glob
#-------end imports----------

#-------containers for the points and data-----------
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#----------end variable initialization

#-------------Process the images for corner points--------------
images = glob.glob('*.png') #get all of the matching file names.
#for every file name of a png in the directory
for fname in images:
    #open the file
    img = cv.imread(fname)
    #convert it to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        #add the world points to thier container
        objpoints.append(objp)
        #use a square of 22x22 pixels around each rough corner. Where the gradient changes rapidly is the corner
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        #append the accurate corner coordinates
        imgpoints.append(corners)
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (7,6), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)
#cv.destroyAllWindows()
#calcualte the camera matrix mtx and distortion coeffcients dist
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#-------------------end processing-------------------

#------------undistort image1 and image2--------------
# undistort imgage1
img = cv.imread('image1.jpeg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('image1undist.png', dst)

# undistort imgage2
img = cv.imread('image2.jpeg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('image2undist.png', dst)
#---------------end undistortion----------------------------

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

np.savetxt('mtx.txt', mtx)
np.savetxt('dist.txt', dist)


