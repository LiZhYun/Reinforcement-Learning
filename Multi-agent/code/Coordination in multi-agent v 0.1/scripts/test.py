import numpy as np

# print( np.eye(5)[[1,1,1,1,1,1,1]])
active_masks = np.ones((5, 3, 1), dtype=np.float32)

arrays = [[True, False, True] for _ in range(5)]
dones = np.stack(arrays)
dones_env = np.all(dones, axis=1)
print(dones == True)
print((dones == True).sum())
active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
print(active_masks)