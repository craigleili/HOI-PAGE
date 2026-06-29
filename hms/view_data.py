import os.path as osp
import sys
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.config import ViewDataConfig, parse_config

class ViewData(Dataset):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(ViewDataConfig, cfg)

        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs):
        up_sign = -1
        azimuth_offset = 180

        if (self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0]) % 360 == 0:
            azimuth = torch.linspace(
                self.cfg.azimuth_range[0],
                self.cfg.azimuth_range[1],
                self.cfg.n_azimuths + 1,
            )[: self.cfg.n_azimuths]
        else:
            azimuth = torch.linspace(
                self.cfg.azimuth_range[0],
                self.cfg.azimuth_range[1],
                self.cfg.n_azimuths,
            )
        elevation = torch.linspace(
            self.cfg.elevation_range[0],
            self.cfg.elevation_range[1],
            self.cfg.n_elevations,
        )
        n_azi = azimuth.shape[0]
        n_ele = elevation.shape[0]
        azimuth = azimuth.view(1, -1).repeat(n_ele, 1).flatten()
        azimuth = (azimuth + azimuth_offset) % 360
        azimuth = azimuth * math.pi / 180
        elevation = elevation.view(-1, 1).repeat(1, n_azi).flatten()
        elevation = elevation * math.pi / 180
        elevation *= up_sign

        fovy = torch.full_like(azimuth, self.cfg.fovy_deg)
        fovy = fovy * math.pi / 180

        camera_distances = torch.full_like(azimuth, self.cfg.camera_distance)

        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            ],
            dim=-1,
        )

        center = torch.as_tensor([0, 0, 0]).float().view(1, 3).repeat(camera_positions.shape[0], 1)
        up = torch.as_tensor([0, up_sign, 0]).float().view(1, 3).repeat(camera_positions.shape[0], 1)

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.linalg.cross(lookat, up, dim=-1), dim=-1)
        up = F.normalize(torch.linalg.cross(right, lookat, dim=-1), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        self.azimuth = azimuth
        self.elevation = elevation
        self.camera_positions = camera_positions
        self.camera_distances = camera_distances
        self.fovy = fovy
        self.c2w = c2w

    def __len__(self):
        return self.camera_positions.shape[0]

    def __getitem__(self, index):
        return {
            "index": index,
            "azimuth": self.azimuth[index],
            "elevation": self.elevation[index],
            "camera_positions": self.camera_positions[index],
            "camera_distances": self.camera_distances[index],
            "fovy": self.fovy[index],
            "c2w": self.c2w[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch
