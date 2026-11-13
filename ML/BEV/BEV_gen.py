import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from simpleCNN import SimpleCNN
XMAX,XMIN = 50,0
YMAX,YMIN = 25,-25
Z=0
DX = 0.5
DY = 0.5

Nx = int((XMAX-XMIN)/DX)
Ny = int((YMAX-YMIN)/DY)

#Generate coordinates
x_coords = torch.linspace(XMIN + DX/2,XMAX-DX/2,Nx)
y_coords = torch.linspace(YMIN + DY/2,YMAX-DY/2,Ny)

x_grid, y_grid = torch.meshgrid(x_coords,y_coords,indexing="ij")
z_grid = torch.full_like(x_grid,Z)
#Combine all grids

bev_grid = torch.stack([x_grid,y_grid,z_grid],dim=-1)
print(f"BEV grid shape: {bev_grid.shape}")

#Flatten to 3D points
bev = bev_grid.reshape(-1,3)
print("BEV points shape:", bev.shape)
print("First point:", bev[0])
print("Last point:", bev[-1])
#Visualise

plt.figure(figsize=(8, 8))

plt.scatter(bev[:,1].numpy(),bev[:,0].numpy(),s=2)
plt.xlabel("Y (metres) - Left / Right")
plt.ylabel("X (metres) - Forward")
plt.title("BEV Coordinate Grid")

plt.xlim(YMIN, YMAX)
plt.ylim(XMIN, XMAX)

plt.grid(True)
plt.axis("equal")
plt.show()


#Camera intrinsic 
K = np.array([[800.0,0.0,640.0],[0.0,800.0,360.0],[0.0,0.0,1.0]],dtype=np.float64)
ImgW = 1280
ImgH = 720

R = np.array([[0.0,-1.0,0.0],[0.0,0.0,-1.0],[1.0,0.0,0.0]],dtype=np.float64)

cam_height = 1.5
C = np.array([[0.0],[0.0],[cam_height]])
t = -R@C

#Convert Rotation to Rodrigues

rvec,_ = cv2.Rodrigues(R)
tvec = t.reshape(3,1)

#Convert BEV pts to numpy
points_vehicle = bev.numpy().astype(np.float64)
dist_coeff = np.zeros((5,1))#Distortion coeffs

img_pts,_ = cv2.projectPoints(points_vehicle,rvec,tvec,K,dist_coeff)

img_pts = img_pts.reshape(-1,2)

print(f"Projected points: {img_pts.shape}")
#Compute camera coordinates

points_cam = (R@points_vehicle.T +t).T
#Validity check
valid_mask = (
    (points_cam[:, 2] > 0) &
    (img_pts[:, 0] >= 0) &
    (img_pts[:, 0] < ImgW) &
    (img_pts[:, 1] >= 0) &
    (img_pts[:, 1] < ImgH)
)

print("Valid points:", valid_mask.sum())
print("Total points:", len(points_vehicle))
plt.figure(figsize=(12, 6))

plt.scatter(
    img_pts[valid_mask, 0],
    img_pts[valid_mask, 1],
    s=2
)

plt.xlim(0, ImgW)
plt.ylim(ImgH, 0)

plt.xlabel("Image u (pixels)")
plt.ylabel("Image v (pixels)")
plt.title("BEV Points Projected onto Camera Image")

plt.grid(True)
plt.show()

#With model
model = SimpleCNN()
img = torch.rand(1,3,720,1280)
features = model(img)

print(features.shape)

B,C,Hf,Wf = features.shape

stride=8

uv_feature = img_pts/stride
u = uv_feature[:,0]
v = uv_feature[:,1]

#Normalisation
u_norm = 2.0*u/(Wf-1) -1.0 
v_norm = 2.0*v/(Hf-1) -1.0 

sampling_grid = np.stack([u_norm, v_norm],axis=-1)
#Convert to tensor
sampling_grid = torch.tensor(sampling_grid,dtype=torch.float32)
sampling_grid = sampling_grid.reshape(1,100,100,2)
#Move invalid points outside the sampling grid

valid = torch.tensor(valid_mask, dtype=torch.bool).reshape(1,100,100)

sampling_grid[~valid] = 2.0

#Sample CNN features
F = torch.nn.functional
bev_features = F.grid_sample(
    features,
    sampling_grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True
)

print("BEV features:", bev_features.shape)
#Visualise one feature
bev_visual = bev_features[0].mean(dim=0)

plt.figure(figsize=(8, 8))

plt.imshow(
    bev_visual.detach().numpy(),
    origin="lower",
    extent=[YMIN, YMAX, XMIN, XMAX],
    aspect="equal")

plt.xlabel("Y (metres)")
plt.ylabel("X (metres)")
plt.title("Sampled BEV Feature Map")
plt.colorbar(label="Mean feature activation")
plt.show()
