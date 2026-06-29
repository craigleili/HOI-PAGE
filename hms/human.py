import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import roma
import smplx
import pickle
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from dataclasses import dataclass

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.config import SmplxHumansConfig, parse_config
from hms.common import (
    may_create_folder,
    read_json,
    to_numpy,
    to_torch,
    to_list,
    get_rotation_matrix,
    get_device,
    load_smplx_uv,
)
import nvdiffrec.render as nvdr

class SmplxHumans(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(SmplxHumansConfig, cfg)
        self.device = get_device()

        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs):
        self.name = self.cfg.human_name.strip()

        smplx_model = smplx.create(
            model_path=self.cfg.smplx_path,
            model_type=self.cfg.smplx_type,
            batch_size=self.cfg.n_frames,
            gender=self.cfg.gender,
        )
        smplx_model.to(self.device)
        smplx_model.requires_grad_(False)
        n_joints = smplx_model.NUM_BODY_JOINTS

        if self.cfg.smplx_type == "smplx":
            self.n_vertices = 10475
        elif self.cfg.smplx_type == "smpl" or self.cfg.smplx_type == "smplh":
            self.n_vertices = 6890
        else:
            raise RuntimeError(f"Unknown smplx_type: {self.cfg.smplx_type}")

        vposer_model, _ = load_model(
            self.cfg.vposer_path,
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        vposer_model.to(self.device)

        self.render_data = load_smplx_uv(self.cfg.uv_path, self.cfg.tex_path)

        segmentations = read_json(self.cfg.segmentation_path)
        for k, v in segmentations.items():
            segmentations[k] = torch.as_tensor(v).long().to(self.device)
        self.segmentations = segmentations

        @dataclass
        class SubModules:
            smplx_model: nn.Module
            vposer_model: nn.Module

        self.sub_modules = SubModules(smplx_model, vposer_model)

        global_orient = np.identity(4, dtype=np.float32)
        for idx, axis in enumerate(self.cfg.rotation_axes):
            global_orient = get_rotation_matrix(axis, self.cfg.rotation_angles[idx]) @ global_orient
        global_orient = to_torch(global_orient[:3, :3]).to(self.device)
        global_orient = roma.rotmat_to_rotvec(global_orient)
        global_orient = global_orient.unsqueeze(0).repeat(self.cfg.n_frames, 1)

        transl = torch.as_tensor(to_list(self.cfg.translation)).float().to(self.device)
        transl = transl.unsqueeze(0).repeat(self.cfg.n_frames, 1)
        betas = torch.zeros(self.cfg.n_frames, 10).to(self.device)
        body_pose = torch.zeros(self.cfg.n_frames, 32).to(self.device)
        if not self.cfg.use_latent_pose:
            body_pose = self.vposer_model.decode(body_pose)["pose_body"].contiguous().view(self.cfg.n_frames, 21 * 3)
            if n_joints > 21:
                body_pose = torch.cat(
                    (
                        body_pose,
                        torch.zeros_like(body_pose[:, : (n_joints - 21) * 3]),
                    ),
                    dim=1,
                )

        if self.cfg.use_continous_rot_repr:
            global_orient = roma.rotvec_to_rotmat(global_orient)
            global_orient = global_orient[..., :2].contiguous().view(self.cfg.n_frames, 3 * 2)
            if not self.cfg.use_latent_pose:
                body_pose = roma.rotvec_to_rotmat(body_pose.view(self.cfg.n_frames, n_joints, 3))
                body_pose = body_pose[..., :2].contiguous().view(self.cfg.n_frames, n_joints, 3 * 2)

        self.transl = nn.Parameter(transl)
        self.global_orient = nn.Parameter(global_orient)
        self.betas = nn.Parameter(betas)
        self.body_pose = nn.Parameter(body_pose)

    @property
    def smplx_model(self):
        return self.sub_modules.smplx_model

    def set_smplx_model(self, model):
        self.sub_modules.smplx_model = model

    @property
    def vposer_model(self):
        return self.sub_modules.vposer_model

    def set_vposer_model(self, model):
        self.sub_modules.vposer_model = model

    def get_body_parts(self):
        return sorted(list(set(self.segmentations.keys())))

    def get_body_segmentations(self, part_labels):
        if isinstance(part_labels, str):
            part_labels = [part_labels]
        assert isinstance(part_labels, (list, tuple))
        supported_labels = self.get_body_parts()
        labels = torch.zeros(self.n_vertices).int().to(self.device)
        for pl in part_labels:
            if pl not in supported_labels:
                continue
            labels[self.segmentations[pl]] = 1
        return labels > 0

    def aux_render_data(self):

        uv_dict = self.render_data
        uvs = to_torch(uv_dict["uvs"]).to(self.device)
        faces = to_torch(uv_dict["faces"]).to(self.device)
        face_uv_indices = to_torch(uv_dict["face_uv_indices"]).to(self.device)
        kd = to_torch(uv_dict["map_Kd"]).flip(0).float().to(self.device) / 255.0
        kd = nvdr.texture.Texture2D(kd)
        ks = to_torch(uv_dict["map_Ks"]).float().to(self.device)
        ks = nvdr.texture.Texture2D(ks)
        return [
            {
                "uvs": uvs,
                "faces": faces,
                "face_uv_indices": face_uv_indices,
                "kd": kd,
                "ks": ks,
            }
        ]

    @torch.no_grad()
    def update(
        self,
        transl=None,
        global_orient=None,
        betas=None,
        body_pose=None,
    ):
        n_frames = self.cfg.n_frames
        n_joints = self.smplx_model.NUM_BODY_JOINTS

        if transl is not None:
            assert transl.shape == (n_frames, 3)
            self.transl.copy_(transl)

        if global_orient is not None:
            assert global_orient.shape == (n_frames, 3)
            if self.cfg.use_continous_rot_repr:
                global_orient = roma.rotvec_to_rotmat(global_orient)
                global_orient = global_orient[..., :2].contiguous().view(n_frames, 3 * 2)
            self.global_orient.copy_(global_orient)

        if betas is not None:
            assert betas.shape == (n_frames, 10)
            self.betas.copy_(betas)

        if body_pose is not None:
            assert body_pose.shape == (n_frames, n_joints * 3)
            if self.cfg.use_latent_pose:
                body_pose = self.vposer_model.encode(body_pose[..., : 21 * 3]).mean
                assert body_pose.shape == (n_frames, 32)
            else:
                if self.cfg.use_continous_rot_repr:
                    body_pose = body_pose.view(n_frames, n_joints, 3)
                    body_pose = roma.rotvec_to_rotmat(body_pose)
                    body_pose = body_pose[..., :2].contiguous().view(n_frames, -1, 3 * 2)
            self.body_pose.copy_(body_pose)

    def forward(self, *args, **kwargs):
        n_frames = self.cfg.n_frames
        n_joints = self.smplx_model.NUM_BODY_JOINTS

        transl = self.transl
        global_orient = self.global_orient
        betas = self.betas
        body_pose = self.body_pose

        if self.cfg.use_continous_rot_repr:
            global_orient = global_orient.view(global_orient.shape[:-1] + (3, 2))
            global_orient = roma.special_gramschmidt(global_orient)
            global_orient = roma.rotmat_to_rotvec(global_orient)

        if self.cfg.use_latent_pose:
            body_pose = self.vposer_model.decode(body_pose)["pose_body"].contiguous().view(n_frames, 21 * 3)
            if n_joints > 21:
                body_pose = torch.cat(
                    (
                        body_pose,
                        torch.zeros_like(body_pose[:, : (n_joints - 21) * 3]),
                    ),
                    dim=1,
                )
        else:
            if self.cfg.use_continous_rot_repr:
                body_pose = roma.special_gramschmidt(body_pose.view(n_frames, -1, 3, 2))
                body_pose = roma.rotmat_to_rotvec(body_pose)
                body_pose = body_pose.view(n_frames, -1)

        sdict = self.smplx_model(
            transl=transl,
            global_orient=global_orient,
            betas=betas,
            body_pose=body_pose,

        )

        vertices = sdict.vertices
        joints = sdict.joints
        faces = self.smplx_model.faces_tensor.long()

        return {
            "vertices": vertices,
            "joints": joints,
            "faces": faces,
            "transl": transl,
            "global_orient": global_orient,
            "betas": betas,
            "body_pose": body_pose,

            "gender": self.smplx_model.gender,
            "name": self.name,
        }

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

        with open(osp.join(output_dir, "human_poses.pkl"), "wb") as fh:
            pickle.dump(
                {
                    "transl": to_numpy(sdict["transl"]),
                    "global_orient": to_numpy(sdict["global_orient"]),
                    "betas": to_numpy(sdict["betas"]),
                    "body_pose": to_numpy(sdict["body_pose"]),
                    "name": sdict["name"],
                    "model_path": self.cfg.smplx_path,
                    "model_type": self.cfg.smplx_type,
                    "batch_size": self.cfg.n_frames,
                    "gender": self.cfg.gender,
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
            nvdr.obj.write_obj(output_dir, f"human_{fid:03d}", mesh)
            mesh_paths.append(osp.join(output_dir, f"human_{fid:03d}.obj"))

        return mesh_paths
