import numpy as np
import torch
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
import io
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.sdf_fn import sdf_to_filled_mesh, mesh_to_voxel, sdf_to_voxel, sdf_to_mesh_with_flexicubes
import os
import glob

def generate_rotating_mesh_gif(mat, gif_name='rotating_mesh.gif'):
    """
    Generate a rotating 3D mesh GIF from a 3D tensor or numpy array.

    Parameters:
    - mat: The input 3D data, either as a PyTorch tensor or a numpy array.
    - gif_name: The filename for the output GIF.
    """

    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()

    b = mat.shape[0]
    meshes = [sdf_to_filled_mesh(mat[j, 0], level=0.02) for j in range(b)]

    angles = np.linspace(0, 360, 36)
    images = []

    axes = []
    fig = plt.figure(figsize=(10, 5))
    for j in range(b):
        verts, faces = meshes[j]
        ax = fig.add_subplot(1, b, j + 1, projection='3d')
        axes.append(ax)
        mesh = Poly3DCollection(verts[faces], facecolor='lightgrey', edgecolor='dimgray', alpha=0.9)
        ax.add_collection3d(mesh)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        cmap='gray', lw=0.5, edgecolor='dimgray', alpha=0.7)
        ax.set_xlim(np.min(verts[:, 0]), np.max(verts[:, 0]))
        ax.set_ylim(np.min(verts[:, 1]), np.max(verts[:, 1]))
        ax.set_zlim(np.min(verts[:, 2]), np.max(verts[:, 2]))
        ax.axis('off')

    plt.tight_layout()

    for angle in angles:
        for ax in axes:
            ax.view_init(elev=20., azim=angle)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()

    plt.close(fig)
    imageio.mimsave(gif_name, images, duration=0.1)
    print(f"GIF generated: {gif_name}")

def generate_rotating_voxel_gif(mat, gif_name='rotating_mesh.gif'):
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()

    b = mat.shape[0]
    voxels = [sdf_to_voxel(mat[j, 0], level=0.02) for j in range(b)]

    angles = np.linspace(0, 360, 72)
    images = []
    axes = []

    fig = plt.figure(figsize=(10, 5))
    for j in range(b):
        voxel = voxels[j]
        ax = fig.add_subplot(1, b, j + 1, projection='3d')
        axes.append(ax)
        filled = np.argwhere(voxel)
        ax.scatter(filled[:, 0], filled[:, 1], filled[:, 2], color='gray', edgecolors='k', alpha=0.9)
        ax.set_xlim(0, voxel.shape[0])
        ax.set_ylim(0, voxel.shape[1])
        ax.set_zlim(0, voxel.shape[2])
        ax.axis('off')

    plt.tight_layout()

    for angle in angles:
        for ax in axes:
            ax.view_init(elev=20., azim=angle)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        images.append(imageio.v3.imread(buf))
        buf.close()

    plt.close(fig)
    imageio.mimsave(gif_name, images, duration=0.1)

    print(f"GIF generated: {gif_name}")
    return np.stack(voxels)

def draw_gif(image, name, idx, output_dir):
    verts, faces, normals, values = marching_cubes(image, 0.02, spacing=(1, 1, 1))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidths=0.1)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    plt.tight_layout()

    compare_dir = f"{output_dir}/compare_{idx}"
    for view in range(0, 360, 30):
        ax.view_init(30, view)
        plt.show()
        plt.savefig(f"{compare_dir}/mesh_{view:03d}.png")

    plt.close()

    os.system(f"convert -delay 50 -loop 0 {compare_dir}/mesh_*.png {compare_dir}/mesh-{idx:04d}-{name}.gif")
    os.system(f"rm {compare_dir}/mesh_*.png")

def create_mesh_gif(volume, output_dir, prefix, threshold=0.02, title=None):
    """
    Creates a rotating 3D mesh GIF from a volume.

    Args:
        volume: 3D numpy array
        output_dir: Directory to save images and GIF
        prefix: Filename prefix for the output
        threshold: Threshold for marching cubes algorithm
        title: Optional title for the plot
    """
    verts, faces, normals, values = marching_cubes(volume, threshold, spacing=(1, 1, 1))

    fig = plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidths=0.1)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    plt.tight_layout()

    for view in range(0, 360, 30):
        ax.view_init(30, view)
        plt.savefig(f"{output_dir}/{prefix}_{view:03d}.png")

    plt.close()

    os.system(f"convert -delay 50 -loop 0 {output_dir}/{prefix}_*.png '{output_dir}/{prefix}.gif'")
    for file in glob.glob(f"{output_dir}/{prefix}_*.png"):
        os.remove(file)

def create_side_by_side_mesh_gif(orig_volume, recon_volume, output_dir, prefix, threshold=0.02, title=None):
    """
    Creates a rotating 3D mesh GIF with original and reconstructed volumes side by side.
    Uses Open3D OffscreenRenderer (EGL) for fast GPU-accelerated rendering.
    Frames are composited in memory and written directly to GIF via imageio (no temp PNG files).

    Args:
        orig_volume:  Original 3D numpy array
        recon_volume: Reconstructed 3D numpy array
        output_dir:   Directory to save the GIF
        prefix:       Filename prefix
        threshold:    Threshold for mesh extraction (unused; kept for API compat)
        title:        Optional title text burned into the first frame
    """
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    os.makedirs(output_dir, exist_ok=True)

    RES = 64
    W, H = 512, 512

    orig_verts,  orig_faces  = sdf_to_mesh_with_flexicubes(orig_volume,  resolution=RES)
    recon_verts, recon_faces = sdf_to_mesh_with_flexicubes(recon_volume, resolution=RES)

    def make_o3d_mesh(verts, faces):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        return mesh

    orig_mesh  = make_o3d_mesh(orig_verts,  orig_faces)
    recon_mesh = make_o3d_mesh(recon_verts, recon_faces)

    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    mat.base_color = np.array([0.75, 0.75, 0.75, 1.0])

    center = np.array([RES / 2, RES / 2, RES / 2], dtype=np.float32)
    radius = float(RES * 1.5)
    fov    = 60.0

    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

    def render_at_angle(mesh_obj, azim_deg, elev_deg=25.0):
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry('mesh', mesh_obj, mat)
        azim = np.radians(azim_deg)
        elev = np.radians(elev_deg)
        eye = center + radius * np.array([
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
            np.cos(elev) * np.cos(azim),
        ], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        renderer.setup_camera(fov, center, eye, up)
        return np.asarray(renderer.render_to_image())

    angles = list(range(0, 360, 30))
    frames = []
    for azim in angles:
        left  = render_at_angle(orig_mesh,  azim)
        right = render_at_angle(recon_mesh, azim)
        frame = np.concatenate([left, right], axis=1)
        frames.append(frame)

    from PIL import Image
    gif_path = f'{output_dir}/{prefix}.gif'
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=500,
        loop=0,
    )
    print(f"GIF saved: {gif_path}")

def create_four_panel_mesh_gif(
    orig_volume, decoded_volume,
    pad_orig_volume, pad_decoded_volume,
    output_dir, prefix, threshold=0.02, title=None
):
    """
    Creates a rotating 3D mesh GIF with a 2×2 grid layout:
        Top-left:    orig_volume      (no pad original)
        Top-right:   decoded_volume   (no pad decoded)
        Bottom-left: pad_orig_volume  (after pad_and_resize_sdf)
        Bottom-right:pad_decoded_volume

    Args:
        orig_volume:       Original SDF, no padding
        decoded_volume:    Decoded SDF from original (no padding)
        pad_orig_volume:   Original SDF after pad_and_resize_sdf
        pad_decoded_volume:Decoded SDF from padded version
        output_dir:        Directory to save the GIF
        prefix:            Filename prefix
        threshold:         Unused; kept for API compatibility
        title:             Optional title
    """
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)

    RES = 64
    W, H = 512, 512
    LABEL_H = 36

    def extract_mesh(vol):
        return sdf_to_mesh_with_flexicubes(vol, resolution=RES)

    orig_verts,     orig_faces      = extract_mesh(orig_volume)
    decoded_verts,  decoded_faces   = extract_mesh(decoded_volume)
    pad_orig_verts, pad_orig_faces  = extract_mesh(pad_orig_volume)
    pad_dec_verts,  pad_dec_faces   = extract_mesh(pad_decoded_volume)

    def make_o3d_mesh(verts, faces):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        return mesh

    orig_mesh     = make_o3d_mesh(orig_verts,     orig_faces)
    decoded_mesh  = make_o3d_mesh(decoded_verts,  decoded_faces)
    pad_orig_mesh = make_o3d_mesh(pad_orig_verts, pad_orig_faces)
    pad_dec_mesh  = make_o3d_mesh(pad_dec_verts,  pad_dec_faces)

    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    mat.base_color = np.array([0.75, 0.75, 0.75, 1.0])

    center = np.array([RES / 2, RES / 2, RES / 2], dtype=np.float32)
    radius = float(RES * 1.5)
    fov    = 60.0

    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

    def render_at_angle(mesh_obj, azim_deg, elev_deg=25.0):
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry('mesh', mesh_obj, mat)
        azim = np.radians(azim_deg)
        elev = np.radians(elev_deg)
        eye = center + radius * np.array([
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
            np.cos(elev) * np.cos(azim),
        ], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        renderer.setup_camera(fov, center, eye, up)
        return np.asarray(renderer.render_to_image())

    def make_label_bar(panel_w, label_texts, bar_h=LABEL_H):
        bar = Image.new("RGB", (panel_w * len(label_texts), bar_h), color=(240, 240, 240))
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        for i, txt in enumerate(label_texts):
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = i * panel_w + (panel_w - tw) // 2
            y = (bar_h - th) // 2
            draw.text((x, y), txt, fill=(30, 30, 30), font=font)
        return np.array(bar)

    top_label    = make_label_bar(W, ["Original (no pad)", "Decoded (no pad)"])
    bottom_label = make_label_bar(W, ["Original (pad)",    "Decoded (pad)"])

    angles = list(range(0, 360, 30))
    frames = []
    for azim in angles:
        tl = render_at_angle(orig_mesh,     azim)
        tr = render_at_angle(decoded_mesh,  azim)
        bl = render_at_angle(pad_orig_mesh, azim)
        br = render_at_angle(pad_dec_mesh,  azim)
        top_row    = np.concatenate([top_label,    np.concatenate([tl, tr], axis=1)], axis=0)
        bottom_row = np.concatenate([bottom_label, np.concatenate([bl, br], axis=1)], axis=0)
        frame = np.concatenate([top_row, bottom_row], axis=0)
        frames.append(frame)

    gif_path = f'{output_dir}/{prefix}.gif'
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=500,
        loop=0,
    )
    print(f"GIF saved: {gif_path}")

def create_triple_mesh_gif(orig_volume, no_adapt_volume, adapt_volume,
                           output_dir, prefix, threshold=0.02, title=None):
    """
    Creates a rotating 3D mesh GIF with three volumes side by side:
        left: original  |  center: no-adapt recon  |  right: adapt recon

    Args:
        orig_volume:      Original 3D numpy array
        no_adapt_volume:  VAE decode without adaptor
        adapt_volume:     VAE decode + adaptor
        output_dir:       Directory to save the GIF
        prefix:           Filename prefix
        threshold:        Unused; kept for API compatibility
        title:            Optional title (written as label row above GIF)
    """
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)

    RES = 64
    W, H = 512, 512

    def extract_mesh(vol):
        return sdf_to_mesh_with_flexicubes(vol, resolution=RES)

    orig_verts,     orig_faces     = extract_mesh(orig_volume)
    no_adapt_verts, no_adapt_faces = extract_mesh(no_adapt_volume)
    adapt_verts,    adapt_faces    = extract_mesh(adapt_volume)

    def make_o3d_mesh(verts, faces):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        return mesh

    orig_mesh     = make_o3d_mesh(orig_verts,     orig_faces)
    no_adapt_mesh = make_o3d_mesh(no_adapt_verts, no_adapt_faces)
    adapt_mesh    = make_o3d_mesh(adapt_verts,    adapt_faces)

    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    mat.base_color = np.array([0.75, 0.75, 0.75, 1.0])

    center = np.array([RES / 2, RES / 2, RES / 2], dtype=np.float32)
    radius = float(RES * 1.5)
    fov    = 60.0

    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

    def render_at_angle(mesh_obj, azim_deg, elev_deg=25.0):
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry('mesh', mesh_obj, mat)
        azim = np.radians(azim_deg)
        elev = np.radians(elev_deg)
        eye = center + radius * np.array([
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
            np.cos(elev) * np.cos(azim),
        ], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        renderer.setup_camera(fov, center, eye, up)
        return np.asarray(renderer.render_to_image())

    LABEL_H = 36
    labels = ["Original", "No Adaptor", "Adaptor"]

    def make_label_bar(panel_w, panel_h_unused, label_texts, bar_h=LABEL_H):
        """Returns an RGB uint8 array of shape [bar_h, 3*panel_w, 3]."""
        bar = Image.new("RGB", (panel_w * len(label_texts), bar_h), color=(240, 240, 240))
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        for i, txt in enumerate(label_texts):
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = i * panel_w + (panel_w - tw) // 2
            y = (bar_h - th) // 2
            draw.text((x, y), txt, fill=(30, 30, 30), font=font)
        return np.array(bar)

    angles = list(range(0, 360, 30))
    label_bar = make_label_bar(W, H, labels)
    frames = []
    for azim in angles:
        left   = render_at_angle(orig_mesh,     azim)
        center_ = render_at_angle(no_adapt_mesh, azim)
        right  = render_at_angle(adapt_mesh,    azim)
        row = np.concatenate([left, center_, right], axis=1)

        frame = np.concatenate([label_bar, row], axis=0)
        frames.append(frame)

    from PIL import Image as _Image
    gif_path = f'{output_dir}/{prefix}.gif'
    pil_frames = [_Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=500,
        loop=0,
    )
    print(f"GIF saved: {gif_path}")
