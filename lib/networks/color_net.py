import torch.nn as nn
import os
import sys

MODULE_PATH = os.path.abspath(".")
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
#import spconv
import torch.nn.functional as F
import numpy as np
from lib.networks import embedder
from lib.utils.blend_utils import *
from lib.utils.vis_3d import *
import torch
import tinycudann as tcnn
import commentjson as json

def normalize(pts, bbox):
    c = (bbox[0] + bbox[1]) / 2
    s = (bbox[1] - bbox[0])
    center = c
    scale = s
    bbox = bbox
    pts = (pts - center) / scale + 0.5
    return pts


class ColorNetwork(nn.Module):
    def __init__(self, cfg):
        super(ColorNetwork, self).__init__()
        self.color_latent = nn.Embedding(cfg.num_latent_code, 128)
        self.view_encoder =  tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 3,
                        },
                    ],
                }
        )

        self.xyz_encoder = tcnn.Encoding(n_input_dims=3, 
                                            encoding_config={
                                             "otype": "HashGrid",
                                            "n_levels": 16,
                                            "n_features_per_level": 2,
                                            "log2_hashmap_size": 19,
                                            "base_resolution": 16,
                                            "per_level_scale": 1.5
                                            })
        
        self.color_net = tcnn.Network(
                n_input_dims = (self.view_encoder.n_output_dims if cfg.view_dependent else 0)
                    + (cfg.rgb_feature_dim - 3 if cfg.delta_rgb else cfg.rgb_feature_dim)
                    + (128 if cfg.latent_t else 0)
                    + (128 if cfg.pose_color else 0)
                    + (self.xyz_encoder.n_output_dims if cfg.txyz and not cfg.pose_color else 0),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": cfg.color_net_width,
                    "n_hidden_layers": 2,
                },
            )

        self.basis_net = tcnn.Network(
            n_input_dims = self.view_encoder.n_output_dims + 128,
            n_output_dims= 8,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 4,
            },
        )


    def forward(self, batch, cfg, feature, viewdir, txyz, tbounds, pose_latent=None):
        if cfg.rgb_only:
            return torch.sigmoid(feature[:, -3:])
        num_latent = torch.tensor(batch['latent_index'], device=feature.device)
        if cfg.rgb_basis:
            num, _ = feature.shape
            #n x 8 x 3
            view_embedding = self.view_encoder(viewdir)
            feature = feature.view(num, -1, 3)
            latent = self.color_latent(num_latent)
            latent_feature = latent.repeat(feature.shape[0], 1)
            input = torch.cat([view_embedding, latent_feature], dim = -1)
            #n x 8
            coefficient = torch.sigmoid(self.basis_net(input)).float()
            rgb = coefficient.view(num, 1, 8) @ feature
            rgb = torch.sigmoid(rgb.squeeze())
            return rgb
        else:
            input = feature

        if(cfg.pose_color):
            input = torch.cat([input, pose_latent[0].transpose(0, 1)], dim = -1)
        elif cfg.txyz:
            txyz = normalize(txyz, tbounds)
            txyz_embbeding = self.xyz_encoder(txyz)
            input = torch.cat([input, txyz_embbeding], dim = -1)

        if(cfg.view_dependent):
            view_embedding = self.view_encoder(viewdir)
            input = torch.cat([input, view_embedding], dim = -1)
        if(cfg.latent_t):
            latent = self.color_latent(num_latent)
            latent_feature = latent.repeat(feature.shape[0], 1)
            input = torch.cat([input, latent_feature], dim = -1)
   
        rgb = self.color_net(input.contiguous()).reshape(-1, 3).to(feature)
        if cfg.delta_rgb: 
            rgb = torch.sigmoid(feature[:, -3:] + rgb)
        else:
            rgb = torch.sigmoid(rgb)
        return rgb


if __name__ == "__main__":
    config_pth = os.path.join("config/config.json")
    with open(config_pth) as f:
        config = json.load(f)
    cl_net = ColorNetwork(config)

    