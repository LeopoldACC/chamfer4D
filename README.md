# chamfer4D
 chamfer distance for 4D point cloud dataset,such as Semantic KITTI.
change from the [implement](https://github.com/chrdiller/pyTorchChamferDistance)
Due to JIT compile,you can use the code without installing it.
```python
from chamfer4D import ChamferDistance
chamfer_dist = ChamferDistance()

dist1, dist2 = chamfer_dist(points, points_reconstructed)
loss = (torch.mean(dist1)) + (torch.mean(dist2))
```
