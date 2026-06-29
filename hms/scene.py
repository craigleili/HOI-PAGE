import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import roma
import pickle

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.config import MeshScenesConfig, parse_config
from hms.common import (
    may_create_folder,
    read_json,
    to_numpy,
    to_torch,
    get_device,
    load_o3d_pcd,
)
import nvdiffrec.render as nvdr

class MeshScenes(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(MeshScenesConfig, cfg)
        self.device = get_device()

        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs):
        self.name = self.cfg.obj_name.strip()

        obj_path = self.cfg.obj_path
        self.mesh = nvdr.obj.load_obj(obj_path)

        pcd_path = f"{obj_path[:-4]}_pcd.ply"
        pcd = load_o3d_pcd(pcd_path)["V"]
        self.pcd = to_torch(pcd).float().to(self.device)

        bbox_min = torch.min(self.pcd, dim=0)[0]
        bbox_max = torch.max(self.pcd, dim=0)[0]
        bcorners = torch.stack(
            [
                torch.tensor([bbox_min[0], bbox_min[1], bbox_min[2]]),
                torch.tensor([bbox_min[0], bbox_max[1], bbox_min[2]]),
                torch.tensor([bbox_min[0], bbox_min[1], bbox_max[2]]),
                torch.tensor([bbox_min[0], bbox_max[1], bbox_max[2]]),
                torch.tensor([bbox_max[0], bbox_min[1], bbox_min[2]]),
                torch.tensor([bbox_max[0], bbox_max[1], bbox_min[2]]),
                torch.tensor([bbox_max[0], bbox_min[1], bbox_max[2]]),
                torch.tensor([bbox_max[0], bbox_max[1], bbox_max[2]]),
            ]
        )
        self.bcorners = bcorners.float().to(self.device)

        sdf_path = f"{obj_path[:-4]}.json"
        sdf_json_path = sdf_path
        sdf_npy_path = sdf_path[:-5] + "_sdf.npy"
        sdf_meta = read_json(sdf_json_path)
        sdf_min = np.array(sdf_meta["min"]).astype(np.float32)
        sdf_min = to_torch(np.reshape(sdf_min, (1, 1, 3))).to(self.device)
        sdf_max = np.array(sdf_meta["max"]).astype(np.float32)
        sdf_max = to_torch(np.reshape(sdf_max, (1, 1, 3))).to(self.device)
        sdf_dim = sdf_meta["dim"]
        sdf = np.load(sdf_npy_path).astype(np.float32)
        sdf = np.reshape(sdf, (1, 1, sdf_dim, sdf_dim, sdf_dim))
        self.sdf = to_torch(sdf).to(self.device)
        self.sdf_min = sdf_min
        self.sdf_max = sdf_max

        self.register_buffer("scales", torch.zeros(1).float().to(self.device))

        rotation_shape = (self.cfg.n_frames, 6 if self.cfg.use_continous_rot_repr else 3)
        self.rotations = nn.Parameter(torch.zeros(*rotation_shape).float().to(self.device))

        self.translations = nn.Parameter(torch.zeros(self.cfg.n_frames, 3).float().to(self.device))

        self.reset_xforms()

    def reset_xforms(self, scales=None, rotations=None, translations=None):

        if scales is None:
            scales = 0.0
        self.reset_scales(scales)

        if rotations is None:
            rotations = torch.eye(3)
        self.reset_rotations(rotations)

        if translations is None:
            translations = [0, 0, 0]
        self.reset_translations(translations)

    @torch.no_grad()
    def reset_scales(self, scales):
        if not torch.is_tensor(scales):
            scales = torch.as_tensor([scales])

        assert scales.ndim == 1
        scales = scales.float().to(self.device)

        self.scales.copy_(scales)

    @torch.no_grad()
    def reset_rotations(self, rotations):
        if not torch.is_tensor(rotations):
            rotations = torch.as_tensor(rotations)
        rotations = rotations.float().to(self.device)

        if rotations.ndim == 2:
            rotations = rotations.unsqueeze(0).repeat(self.cfg.n_frames, 1, 1)
        assert rotations.ndim == 3

        if self.cfg.use_continous_rot_repr:
            rotations = rotations[..., :2].contiguous().view(self.cfg.n_frames, 3 * 2)
        else:
            rotations = roma.rotmat_to_rotvec(rotations)

        self.rotations.copy_(rotations)

    @torch.no_grad()
    def reset_translations(self, translations):
        if not torch.is_tensor(translations):
            translations = torch.as_tensor(translations)
        translations = translations.float().to(self.device)

        if translations.ndim == 1:
            translations = translations.unsqueeze(0).repeat(self.cfg.n_frames, 1)
        assert translations.ndim == 2

        self.translations.copy_(translations)

    def aux_render_data(self):

        mesh = self.mesh
        faces = mesh.t_pos_idx.long().to(self.device)
        uvs = mesh.v_tex.to(self.device)
        face_uv_indices = mesh.t_tex_idx.long().to(self.device)
        kd = mesh.material.kd
        kd = nvdr.texture.rgb_to_srgb(kd)
        ks = mesh.material.ks
        return [
            {
                "uvs": uvs,
                "faces": faces,
                "face_uv_indices": face_uv_indices,
                "kd": kd,
                "ks": ks,
            }
        ]

    def get_xforms(self):
        scales = torch.exp(self.scales).view(1, 1, 1)
        translations = self.translations.unsqueeze(2)
        rotations = self.rotations
        if self.cfg.use_continous_rot_repr:
            rotations = rotations.view(rotations.shape[:-1] + (3, 2))
            rotations = roma.special_gramschmidt(rotations)
        else:
            rotations = roma.rotvec_to_rotmat(rotations)
        pads = torch.as_tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(rotations.shape[0], 1, 1).to(rotations)
        xform = torch.cat((scales * rotations, translations), dim=2)
        xform = torch.cat((xform, pads), dim=1)
        return xform

    def forward(self, *args, **kwargs):
        xform = self.get_xforms()

        pcds = self.pcd.to(xform)
        pcds = torch.cat((pcds, torch.ones_like(pcds[:, :1])), dim=-1)
        pcds = pcds.unsqueeze(0)
        pcds = pcds @ xform.transpose(1, 2)
        pcds = pcds[..., :3]

        bcorners = self.bcorners.to(xform)
        bcorners = torch.cat((bcorners, torch.ones_like(bcorners[:, :1])), dim=-1)
        bcorners = bcorners.unsqueeze(0)
        bcorners = bcorners @ xform.transpose(1, 2)
        bcorners = bcorners[..., :3]

        mesh = self.mesh
        vertices = mesh.v_pos.to(xform)
        vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
        vertices = vertices.unsqueeze(0)
        vertices = vertices @ xform.transpose(1, 2)
        vertices = vertices[..., :3]

        faces = mesh.t_pos_idx.long().to(self.device)

        return {
            "pcds": pcds,
            "bcorners": bcorners,
            "vertices": vertices,
            "faces": faces,
            "scales": torch.exp(self.scales),
            "rotations": self.rotations,
            "translations": self.translations,
            "xforms": xform,
            "name": self.name,
        }

    def to_canonical(self, vertices):
        xform = self.get_xforms()
        xform_inv = torch.linalg.inv(xform)

        assert vertices.ndim == 3
        vertices = F.pad(vertices, (0, 1), "constant", 1.0)
        vertices = vertices @ xform_inv.transpose(1, 2)
        vertices = vertices[..., :3]

        return vertices

    def get_sdf(self, vertices, is_canonical=False):
        if not is_canonical:
            vertices = self.to_canonical(vertices)

        sdf_min = self.sdf_min
        sdf_max = self.sdf_max
        sdf_vol = self.sdf

        batch_size, num_vertices, _ = vertices.shape
        vertices = (vertices - sdf_min) / (sdf_max - sdf_min) * 2 - 1

        sdf_values = [
            F.grid_sample(
                sdf_vol,
                vertices[i, :, [2, 1, 0]].view(1, num_vertices, 1, 1, 3),
                padding_mode="border",
                align_corners=True,
            ).reshape(1, num_vertices)
            for i in range(batch_size)
        ]
        sdf_values = torch.cat(sdf_values, dim=0)
        return sdf_values

    def export(self, output_dir, interval=1, **kwargs):
        sdict = self(**kwargs)
        adata = self.aux_render_data()
        mid = 0

        may_create_folder(output_dir)

        vertices = sdict["vertices"]
        faces = sdict["faces"]

        uvs_tmp = adata[mid]["uvs"]
        face_uv_indices_tmp = adata[mid]["face_uv_indices"]
        kd_tmp = adata[mid]["kd"]
        kd_tmp = nvdr.texture.srgb_to_rgb(kd_tmp)
        ks_tmp = adata[mid]["ks"]

        with open(osp.join(output_dir, "scene_poses.pkl"), "wb") as fh:
            pickle.dump(
                {
                    "xforms": to_numpy(sdict["xforms"]),
                    "name": sdict["name"],
                    "mesh_path": self.cfg.obj_path,
                },
                fh,
            )

        indices = list(range(0, vertices.shape[0], interval))
        if len(indices) == 0:
            indices = [0]
        elif indices[-1] != vertices.shape[0] - 1:
            indices.append(vertices.shape[0] - 1)

        mesh_paths = list()
        for fid in indices:
            mesh = nvdr.mesh.Mesh(
                vertices[fid],
                faces,
                v_tex=uvs_tmp,
                t_tex_idx=face_uv_indices_tmp,
                material=nvdr.material.Material(
                    {
                        "name": "mat",
                        "bsdf": "pbr",
                        "kd": kd_tmp,
                        "ks": ks_tmp,
                    }
                ),
            )
            nvdr.obj.write_obj(output_dir, f"scene_{fid:03d}", mesh)
            mesh_paths.append(osp.join(output_dir, f"scene_{fid:03d}.obj"))

        return mesh_paths
