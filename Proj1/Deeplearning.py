import numpy as np

features = np.array([
    [[2,4],[3,1]], [[5,2],[1,6]]
])

print(features.shape)
depth_probs = np.array([
    [[0.8, 0.3],
     [0.6, 0.9]],

    [[0.2, 0.7],
     [0.4, 0.1]]
])

print(depth_probs.shape)

lifted = depth_probs[:,None,:,:] * features[None,:,:,:]
print(lifted.shape)
# Inspect feature vectors:
print(f"5m: Feature at (0,0): {lifted[0,:,0,0]}")
print(f"10m: Feature at (0,0): {lifted[1,:,0,0]}")
#SPLAT

bev = np.zeros((2,2,2))

destinations = np.array([
    [[(0,0),(0,1)],[(1,0),(1,1)]],
    [[(0,0),(0,1)],[(1,0),(1,1)]]

])
print(destinations.shape)

for d in range(2):
    for h in range(2):
        for w in range(2):

            row,col = destinations[d,h,w]
            feature_vec = lifted[d,:,h,w]
            bev[row,col] += feature_vec

print(bev)