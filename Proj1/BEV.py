import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#Import imgs
img1 = cv2.imread(ROOT/"img/car1.png")
img2 = cv2.imread(ROOT/"img/car2.png")
#img1 = cv2.resize(img1,(900,900))
#img2 = cv2.resize(img2,(900,900))
if img1 is None or img2 is None:
    raise FileNotFoundError("Make sure import path is correct")

g1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=2000)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

kp1,des1 = orb.detectAndCompute(g1,None)
kp2,des2 = orb.detectAndCompute(g2,None)
if des1 is None or des2 is None:
    raise ValueError("No SIFT features were detected")
matches = matcher.knnMatch(des1,des2,k=2)

# Lowe's ratio test
good = []
for match in matches:
    if(len(match)==2):
        m,n = match
        if m.distance < 0.75*n.distance:
            good.append(m)
print(len(good))

result = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Feature match",result)
cv2.waitKey(0)

pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
if(len(good)<4):
    raise ValueError("Homography estimation needs at least 4 good points.")

H, mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,2.0)

if H is None or mask is None:
    raise RuntimeError("Homography Estimation failed")

#Let's separate inliers and outliers
inliers = [m for m,keep in zip(good,mask.ravel().astype(bool)) if keep]
outliers = [m for m,keep in zip(good,mask.ravel().astype(bool)) if not keep]

print(f"Inliers: {len(inliers)}")
print(f"Outliers: {len(outliers)}")
inlier_vis = cv2.drawMatches(img1,kp1,img2,kp2,inliers,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow(" ",inlier_vis)
cv2.waitKey(0)
H = H/H[2,2]
#Warp images
h2,w2 = img2.shape[:2]
#Refine Homography
inlier_mask = mask.ravel().astype(bool)
p1 = pt1[inlier_mask]
p2 = pt2[inlier_mask]
if(len(p1)<4):
    raise RuntimeError("Homography Estimation requires at least 4 points")

H_refined,_ = cv2.findHomography(p1,p2,method=0)
print(H_refined)
if (H_refined is not None):
    H = H_refined
    H = H/H[2,2]
#Error analysis
projected = cv2.perspectiveTransform(p1,H)
error = np.linalg.norm(projected-p2,axis=1)
print(f"Max error: {error.max()}")
print(f"Mean Reprojection error: {error.mean()}")
warped_img1 = cv2.warpPerspective(img1,H,(w2,h2))

cv2.imshow("warped image",warped_img1)
cv2.waitKey(0)
blend = cv2.addWeighted(img2,0.5,warped_img1,0.5,0)
cv2.imshow("Blended",blend)
cv2.waitKey(0)
cv2.imwrite(ROOT/"img/warped.jpg",warped_img1)
cv2.destroyAllWindows()