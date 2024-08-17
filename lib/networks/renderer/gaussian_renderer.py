import torch
import torch.nn.functional as F
# from lib.utils.vis_3d import write_ply
# from .. import embedder
from lib.utils.blend_utils import *
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh
from lib.utils.blend_utils import pose_dir_to_can_dir


class Renderer:
    # def __init__(self):

  
    def render(self, cfg, viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, pxyz, scaling_modifier = 1.0, override_color = None, override_rotation = None, override_scale = None, override_opacity = None):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pxyz, dtype=pxyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=cfg.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pxyz
        means2D = screenspace_points
        if override_opacity is not None:
            opacity = override_opacity
        else:
            opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None 
        cov3D_precomp = None
        if cfg.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        elif cfg.rotate_gaussian:
            scales = override_scale
            rotations = override_rotation
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if cfg.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pxyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp = pose_dir_to_can_dir(dir_pp, override_rotation)
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        if cfg.mod == "train" and cfg.mask_loss:
            rendered_mask, _= rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = torch.ones_like(colors_precomp),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        else: 
            rendered_mask = None
            

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "rendered_mask": rendered_mask,
                "wiewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
