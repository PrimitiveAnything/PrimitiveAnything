import imageio
import pytorch3d.renderer
from pytorch3d.structures import Meshes
import torch


def render_mesh(vertices, faces, device):
    vertices, faces = [torch.from_numpy(x).to(device=device, dtype=torch.float) for x in (vertices, faces)]

    # Define texture
    textures_rgb = torch.ones_like(vertices, device=device) * torch.tensor([0.6, 0.6, 0.3], device=device)
    textures = pytorch3d.renderer.TexturesVertex([textures_rgb])

    # Assemble Mesh
    mesh = Meshes(
        [vertices], [faces], textures
    )

    # Define cameras
    n_views = 10
    distance = 100
    object_center = vertices.mean(dim=-2)
    elevations = 30 # torch.linspace(0, 2 * torch.pi, n_views, device=device).sin() * 30
    rotation_degrees = torch.linspace(-180, 180, n_views, device=device)
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=distance, elev=elevations, azim=rotation_degrees, device=device, at=object_center.unsqueeze(0),
    ) # (N, 3, 3), (N, 3)
    cameras = pytorch3d.renderer.cameras.FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device,
    )

    # Define lights
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, distance]], device=device)

    # Initialize renderer
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=256, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings),
        shader=pytorch3d.renderer.HardPhongShader(device=device, lights=lights),
    )
    views = renderer(mesh.extend(n_views), cameras=cameras, lights=lights)
    views = (views[:, :, :, :3] * 255).to(torch.uint8)
    views = views.cpu().numpy()

    imageio.mimwrite(
        'test_output.gif', [img for img in views], frame_duration=80, loop=0
    )
