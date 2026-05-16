import torch
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from pytorch3d.loss import chamfer_distance as cd
from torchmetrics.segmentation.hausdorff_distance import HausdorffDistance
import open3d as o3d

def sample_surface_points_o3d(points_np: np.ndarray, num_points: int = 512) -> np.ndarray:
    assert points_np.ndim == 2 and points_np.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    down_pcd = pcd.farthest_point_down_sample(min(num_points, len(points_np)))
    down_np = np.asarray(down_pcd.points)
    cur = down_np.shape[0]
    if cur < num_points:
        pad_idx = np.random.randint(0, cur, size=(num_points - cur,))
        pad_pts = down_np[pad_idx]
        down_np = np.vstack([down_np, pad_pts])
    return down_np

def voxel_to_pointcloud(voxel_set, num_points=512, device='cuda'):
    point_clouds = []
    for voxel in voxel_set:
        try:
            if isinstance(voxel, torch.Tensor):
                voxel = voxel.cpu().numpy()
            level = 0.00
            min_val, max_val = voxel.min(), voxel.max()
            if not (min_val <= level <= max_val):
                level = (min_val + max_val) / 2
            verts, faces, normals, values = marching_cubes(voxel, level=level, spacing=(1,1,1))
            pts_post_sample = sample_surface_points_o3d(verts, num_points=num_points)
            pts_post_sample = torch.from_numpy(pts_post_sample).float().to(device)
            point_clouds.append(pts_post_sample)
        except Exception as e:
            print(f"Error processing voxel: {e}")

            point_clouds.append(torch.zeros(num_points, 3).to(device))
    point_clouds = torch.stack(point_clouds, dim=0)
    print(f"Point clouds shape: {point_clouds.shape}")
    return point_clouds

def cdist_blockwise(x, y, block_size=10240):
    min_dists = []
    for i in range(0, x.size(0), block_size):
        end = min(i + block_size, x.size(0))
        block = x[i:end]
        dists = torch.cdist(block, y)
        min_dists.append(dists.min(dim=1)[0])
    return torch.cat(min_dists, dim=0)

def chamfer_distance(x, y, block_size=10240):
    assert x.size(-1) == 3 and y.size(-1) == 3
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0).to(x.device)
    dist_x_to_y = cdist_blockwise(x, y, block_size).mean()
    dist_y_to_x = cdist_blockwise(y, x, block_size).mean()
    return dist_x_to_y + dist_y_to_x

def compute_mmd(generated_clouds, real_clouds, block_size=10240):
    mmd_total = 0.0
    for gen in generated_clouds:
        min_dist = float('inf')
        for real in real_clouds:
            dist = chamfer_distance(gen, real, block_size).item()
            if dist < min_dist:
                min_dist = dist
        mmd_total += min_dist
    return mmd_total / len(generated_clouds) if generated_clouds else 0.0

def compute_tmd(generated_clouds, block_size=10240):
    tmd_total = 0.0
    count = 0
    num_samples = len(generated_clouds)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist = chamfer_distance(generated_clouds[i], generated_clouds[j], block_size).item()
            tmd_total += dist
            count += 1
    return tmd_total / count if count > 0 else 0.0

def hausdorff_distance_unidirectional(x, y, block_size=10240):
    min_dists_xy = cdist_blockwise(x, y, block_size)
    hxy = min_dists_xy.max().item()
    min_dists_ys = cdist_blockwise(y, x, block_size)
    hyx = min_dists_ys.max().item()
    return max(hxy, hyx)

def compute_uhd(real_cloud, generated_clouds, block_size=10240):
    uhd = hausdorff_distance_unidirectional(real_cloud, generated_clouds, block_size)
    return uhd

@torch.no_grad()
def compute_metrics_uncond(gen_voxel, real_voxel, resolution=64):

    gen_pc = voxel_to_pointcloud(gen_voxel) / float(resolution)
    real_pc = voxel_to_pointcloud(real_voxel) / float(resolution)
    gen_and_real = torch.cat([gen_pc, real_pc], dim=0)

    m = gen_pc.size(0)
    n = real_pc.size(0)
    tot = (m+n) * (m+n)

    a = gen_and_real.unsqueeze(1).expand(-1, m+n, -1, -1)
    b = gen_and_real.unsqueeze(0).expand(m+n, -1, -1, -1)
    a_flat = a.flatten(0, 1)
    b_flat = b.flatten(0, 1)

    dist_list = []
    minibatch_size = 512
    for i in range(0, tot, minibatch_size):
        a_batch = a_flat[i:i + minibatch_size]
        b_batch = b_flat[i:i + minibatch_size]
        dist_batch = cd(a_batch, b_batch, batch_reduction=None)[0]
        dist_list.append(dist_batch)
    dist_matrix = torch.cat(dist_list, dim=0).view(m+n, m+n)

    min_dist, min_ind = dist_matrix[:m, m:].min(dim=1)

    mmd = min_dist.mean().item()

    cov = torch.unique(min_ind).shape[0] / n * 100

    _, second_min_inds = dist_matrix.kthvalue(
        k=2,
        dim=1,
        keepdim=False,
    )

    gen_in_gen = (second_min_inds[:m] < m).sum().item()
    real_in_real = (second_min_inds[m:] >= m).sum().item()
    nna = (gen_in_gen + real_in_real) / (m + n) * 100
    metrics = {
        'mmd': mmd,
        'cov': cov,
        'nna': nna
    }
    del gen_pc, real_pc, gen_and_real, dist_matrix
    return metrics

@torch.no_grad()
def batch_uhd_manual(gen: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    B, K, _ = gen.shape
    hds = torch.zeros(B, device=gen.device)
    for i in range(B):
        A, Bp = gen[i], real[i]
        if A.numel() == 0 or Bp.numel() == 0:
            continue
        D = torch.cdist(A[None], Bp[None], p=2.0).squeeze(0)
        d_ab = D.min(dim=1).values.max()
        d_ba = D.min(dim=0).values.max()
        hds[i] = torch.max(d_ab, d_ba)
    return hds

@torch.no_grad()
def compute_metrics_cond(gen_voxel, real_voxel, resolution=64):
    assert len(gen_voxel) == len(real_voxel), 'The number of generated and real voxels must be the same.'
    gen_pc = voxel_to_pointcloud(gen_voxel) / float(resolution)
    real_pc = voxel_to_pointcloud(real_voxel) / float(resolution)

    dist = cd(gen_pc, real_pc, batch_reduction=None)[0]
    print('cd:\n', dist)

    uhd = batch_uhd_manual(gen_pc, real_pc)
    print('uhd:\n', uhd)

    metrics = {
        'cd': dist.mean().item(),
        'uhd': uhd.mean().item()
    }
    return metrics
