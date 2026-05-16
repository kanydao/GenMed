import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom, median_filter
from scipy.ndimage import binary_fill_holes, label
import torch
import torch.nn.functional as F

def compute_sdf(voxel_matrix):
    inside_dist = distance_transform_edt(voxel_matrix)
    outside_dist = distance_transform_edt(1 - voxel_matrix)
    sdf = outside_dist - inside_dist

    sdf = sdf / 32.0
    sdf = sdf.clip(-0.2, 0.2)
    return sdf

def visualize_sdf(voxel_matrix, sdf, slice_index, prefix='slice'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Voxel Matrix Slice")
    plt.imshow(voxel_matrix[slice_index], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("SDF Slice")
    plt.imshow(sdf[slice_index], cmap='jet')
    plt.colorbar()
    plt.show()
    plt.savefig(f'{prefix}_{slice_index}.png')
    plt.close()

def extract_surface(sdf):
    from skimage.measure import marching_cubes
    vertices, faces, normals, values = marching_cubes(sdf, level=0)
    return vertices, faces

def visualize_surface(vertices, faces):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(vertices[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    plt.show()
    plt.savefig('surface.png')
    plt.close()

def create_3d_mask(vertices, faces, shape):
    from scipy.spatial import Delaunay
    mask = np.zeros(shape, dtype=np.uint8)

    delaunay = Delaunay(vertices)

    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    z = np.arange(0, shape[2])
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    inside = delaunay.find_simplex(grid) >= 0

    grid_points = grid[inside]

    mask[grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]] = 1

    return mask

def fill_holes_along_axis(volume, axis):
    filled_volume = np.copy(volume)
    for i in range(volume.shape[axis]):
        if axis == 0:
            slice_2d = volume[i, :, :]
        elif axis == 1:
            slice_2d = volume[:, i, :]
        else:
            slice_2d = volume[:, :, i]

        filled_slice = binary_fill_holes(slice_2d)

        labeled_array, num_features = label(filled_slice)
        if num_features > 1:
            max_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
            filled_slice = labeled_array == max_label

        if axis == 0:
            filled_volume[i, :, :] = filled_slice
        elif axis == 1:
            filled_volume[:, i, :] = filled_slice
        else:
            filled_volume[:, :, i] = filled_slice

    return filled_volume

def mesh_to_voxel(vertices, faces):
    import pyvista as pv
    shape = [64, 64, 64]
    faces = np.hstack([[3] + list(face) for face in faces])
    mesh = pv.PolyData(vertices, faces)
    grid = pv.voxelize(mesh, density=1.0)
    voxel_points = np.array(grid.points)
    filled_voxel = np.zeros(shape)

    for voxel in voxel_points:
        x, y, z = voxel
        x, y, z = int(np.round(x)), int(np.round(y)), int(np.round(z))
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            filled_voxel[x, y, z] = 1
    return filled_voxel

def sdf_to_filled_mesh(sdf, level=0.02):

    from skimage.measure import marching_cubes
    import open3d as o3d
    vertices_filled, faces_filled, normals_filled, values_filled = marching_cubes(sdf, level=level)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_filled)
    mesh.triangles = o3d.utility.Vector3iVector(faces_filled)
    mesh.compute_vertex_normals()

    mesh = mesh.filter_smooth_simple(number_of_iterations=20)
    mesh.compute_vertex_normals()

    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

def sdf_to_voxel(sdf, level=0.02):

    sdf = gaussian_filter(sdf, sigma=3.0)
    sdf = median_filter(sdf, size=5)
    filled_voxel = binary_fill_holes(sdf <= level).astype(np.uint8)

    z_inds, y_inds, x_inds = np.where(filled_voxel > 0)

    if len(z_inds) == 0:
        return filled_voxel

    z1, z2 = z_inds.min(), z_inds.max()
    y1, y2 = y_inds.min(), y_inds.max()
    x1, x2 = x_inds.min(), x_inds.max()

    d = z2 - z1 + 1
    h = y2 - y1 + 1
    w = x2 - x1 + 1

    D, H, W = filled_voxel.shape
    shifted_voxel = np.zeros_like(filled_voxel)
    shifted_voxel[D//2-d//2:D//2+d-d//2, H//2-h//2:H//2+h-h//2, W//2-w//2:W//2+w-w//2] = filled_voxel[z1:z2+1, y1:y2+1, x1:x2+1]
    filled_voxel = shifted_voxel

    scale = 56 / max(d, h, w)
    scale = max(scale, 1.0)
    filled_voxel = zoom(filled_voxel, scale, order=1)

    D, H, W = filled_voxel.shape
    filled_voxel = filled_voxel[D//2-32: D//2+32, H//2-32: H//2+32, W//2-32: W//2+32]

    for _ in range(10):
        filled_voxel = fill_holes_along_axis(filled_voxel, 1)
        filled_voxel = fill_holes_along_axis(filled_voxel, 2)
        filled_voxel = fill_holes_along_axis(filled_voxel, 0)

    filled_voxel = gaussian_filter(filled_voxel.astype(np.float32), sigma=1.0)
    filled_voxel = filled_voxel > 0.5

    return filled_voxel

def compute_gradients(sdf, spacing=1.0):
    kernel_dx = torch.tensor([[[[-1, 0, 1]]]], device=sdf.device, dtype=sdf.dtype).unsqueeze(0) / (2 * spacing)
    kernel_dy = torch.tensor([[[[-1], [0], [1]]]], device=sdf.device, dtype=sdf.dtype).unsqueeze(0) / (2 * spacing)
    kernel_dz = torch.tensor([[[[-1]], [[0]], [[1]]]], device=sdf.device, dtype=sdf.dtype).unsqueeze(0) / (2 * spacing)

    grad_x = F.conv3d(sdf, kernel_dx, padding="same")
    grad_y = F.conv3d(sdf, kernel_dy, padding="same")
    grad_z = F.conv3d(sdf, kernel_dz, padding="same")

    gradients = torch.cat([grad_x, grad_y, grad_z], dim=1)
    return gradients

def compute_mean_curvature(sdf, spacing=1.0, bound=0.1, epsilon=1e-5):
    gradients = compute_gradients(sdf, spacing=spacing)
    grad_norm = torch.sqrt((gradients ** 2).sum(dim=1, keepdim=True) + epsilon)
    grad_normalized = gradients / grad_norm

    kernel_dx = torch.tensor([[[[-1, 0, 1]]]], device=sdf.device, dtype=sdf.dtype).unsqueeze(0) / (2 * spacing)
    kernel_dy = torch.tensor([[[[-1], [0], [1]]]], device=sdf.device, dtype=sdf.dtype).unsqueeze(0) / (2 * spacing)
    kernel_dz = torch.tensor([[[[-1]], [[0]], [[1]]]], device=sdf.device, dtype=sdf.dtype).unsqueeze(0) / (2 * spacing)

    dn_x_dx = F.conv3d(grad_normalized[:, 0:1, :, :, :], kernel_dx, padding="same")
    dn_x_dy = F.conv3d(grad_normalized[:, 0:1, :, :, :], kernel_dy, padding="same")
    dn_x_dz = F.conv3d(grad_normalized[:, 0:1, :, :, :], kernel_dz, padding="same")

    dn_y_dx = F.conv3d(grad_normalized[:, 1:2, :, :, :], kernel_dx, padding="same")
    dn_y_dy = F.conv3d(grad_normalized[:, 1:2, :, :, :], kernel_dy, padding="same")
    dn_y_dz = F.conv3d(grad_normalized[:, 1:2, :, :, :], kernel_dz, padding="same")

    dn_z_dx = F.conv3d(grad_normalized[:, 2:3, :, :, :], kernel_dx, padding="same")
    dn_z_dy = F.conv3d(grad_normalized[:, 2:3, :, :, :], kernel_dy, padding="same")
    dn_z_dz = F.conv3d(grad_normalized[:, 2:3, :, :, :], kernel_dz, padding="same")

    curvature = (dn_x_dx ** 2 + dn_x_dy ** 2 + dn_x_dz ** 2 +
                 dn_y_dx ** 2 + dn_y_dy ** 2 + dn_y_dz ** 2 +
                 dn_z_dx ** 2 + dn_z_dy ** 2 + dn_z_dz ** 2)

    mask = (sdf.abs() < bound).float()
    mean_curvature_surface = torch.sum(curvature * mask, dim=(2,3,4)) / (torch.sum(mask, dim=(2,3,4)) + epsilon)
    return mean_curvature_surface

def sdf_to_mesh_with_flexicubes(
    sdf: torch.Tensor,
    resolution: int = 64,
    device: str = 'cuda',
    save_path: str = None
):
    import trimesh
    from kaolin.non_commercial import FlexiCubes
    from scipy.interpolate import griddata
    if isinstance(sdf, np.ndarray):
        sdf = torch.from_numpy(sdf).float()
    assert sdf.shape == (resolution, resolution, resolution),\
        f"SDF shape must be ({resolution}, {resolution}, {resolution})"

    sdf = sdf.to(device)
    sdf = F.pad(sdf, (1, 0, 1, 0, 1, 0), mode='constant', value=0.2)
    scalar_field = sdf.reshape(-1)

    flexicubes = FlexiCubes(device=device)
    voxelgrid_vertices, cube_idx = flexicubes.construct_voxel_grid(resolution)

    assert scalar_field.shape[0] == voxelgrid_vertices.shape[0],\
        f"scalar_field must match the number of voxelgrid_vertices ({voxelgrid_vertices.shape[0]})"

    vertices, faces, l_dev = flexicubes(
        voxelgrid_vertices=voxelgrid_vertices,
        scalar_field=scalar_field,
        cube_idx=cube_idx,
        resolution=resolution,
        training=False,
        output_tetmesh=False,
        grad_func=None
    )

    if save_path is not None:
        mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(),
                               faces=faces.detach().cpu().numpy())
        mesh.export(save_path)

    vertices = (vertices + 0.5) * resolution
    vertices = vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    return vertices, faces

if __name__ == "__main__":
    def generate_sphere_sdf(resolution=64, radius=0.5):
        """Build an SDF tensor for a sphere centered at the origin over [-1, 1]^3."""
        lin = torch.linspace(-1, 1, resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        sdf = torch.norm(grid, dim=-1) - radius
        return sdf
    resolution = 64
    sdf = generate_sphere_sdf(resolution=resolution, radius=0.6)

    vertices, faces = sdf_to_mesh_with_flexicubes(
        sdf=sdf,
        resolution=resolution,
        device='cuda',
        save_path=None
    )

    print(f"Mesh extracted: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
