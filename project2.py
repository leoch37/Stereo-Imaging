#!/home/lche49/.virtualenvs/cv/bin/python
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
		# finding random colors for the keypoints
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1) # the 1 is for line thickness
        img1 = cv.circle(img1,tuple(pt1),5,color,-1) # -1 is the thickness of circle
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	

img1 = cv.imread('IMG_0849.JPG',0)
img2 = cv.imread('IMG_0850.JPG',0)

img1 = cv.resize(img1,None,fx=0.5,fy=0.5)
img2 = cv.resize(img2,None,fx=0.5,fy=0.5)

K = np.array([[1644, 0, 1027], [0, 1638, 748], [0, 0, 1]])
distor = np.array([0.2004, -1.196, -0.00393, 0.00001873, 2.013])

focal = 1644
pp = (1027, 748)

# Using SIFT function
sift = cv.xfeatures2d.SIFT_create()

#find Keypoints
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None) 

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) #5)
search_params = dict(checks=50) #50)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
	# this changes the amount of good points, towards 1 its more lenient and towards 0 more picky
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F) # just changed that for both
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img3)
plt.subplot(122),plt.imshow(img5)
plt.savefig('img_epilines1.png')
plt.savefig('img_epilines2.png')
plt.show()

E, mask = cv.findEssentialMat(pts1, pts2, focal, pp, cv.RANSAC)

R1, R2, r = cv.decomposeEssentialMat(E)

#R1_d = np.c_[R1, np.zeros(3)]
R1_t = np.c_[R1, -r]

P1 = np.c_[K, np.zeros(3)]
P2 = np.matmul(K, R1_t)

R1 = np.array(R1, dtype='f')
R2 = np.array(R2, dtype='f')

pts1 = cv.transpose(pts1)
pts2 = cv.transpose(pts2)
pts1 = np.array(pts1, dtype='f')
pts2 = np.array(pts2, dtype='f')

tri_output1 = cv.triangulatePoints(P1, P2, pts1, pts2)

print(R1_t)
print(r)
print(P1)
print(P2)

tri_output1 = tri_output1/tri_output1[3]
z_min_max = tri_output1[2]
print(z_min_max)

tri_output1 = np.matmul(P1,tri_output1)
tri_output1 = tri_output1/tri_output1[2]

tri_output_x = tri_output1[0]
tri_output_y = tri_output1[1]

pts1_x = pts1[0,:]
pts1_y = pts1[1,:]

plot = plt.imshow(img1, cmap='gray')

plt.scatter(tri_output_x, tri_output_y, s=30, c='b', alpha=0.5)
plt.scatter(pts1_x, pts1_y, s=30, c='r', alpha=0.5)
plt.savefig('img1_projected_pts.jpg')

#part 4:
min = np.min(z_min_max)
max = np.max(z_min_max)
print ('max=')
print (max)
print ('min=')
print (min)

####### PART 4 ###########
d = np.linspace(min, max, 20)
homographies = np.full((20,4,4), 0, 'float')
for i in range(20):
	n = ([0, 0, -1, d[i]])
	P1_aug = P1
	P2_aug = P2
	P1_aug = np.vstack([P1_aug, n])
	P2_aug = np.vstack([P2_aug, n])
	P2_inv = np.linalg.inv(P2_aug)
	H = np.matmul(P1_aug,P2_inv)
	homographies[i,:,:] = H

#homographies = homographies.reshape(:,3,3)
homog = homographies[:,0:3,0:3]
#print(homog[5])
#print(np.shape(homog[5]))
#print(type(homog[5,1,1]))
#img_warp = []
#for i in range(20):
	# We wanted to use warp the image here but its not working
	#img_warp[i,:,:] = cv.warpPerspective(img2, homog[i,:,:], None)
	
for i in range(20):	
	abs_diff = img1 # - img_warp[i,:,:];	#this is supposed to subtract the warped image from the first
	abs_diff = np.abs(abs_diff)
	
blur = cv.blur(abs_diff, (15,15))
cv.imwrite('blur.jpg',blur)



