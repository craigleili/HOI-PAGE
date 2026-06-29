import os.path as osp
import sys
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.config import NVRendererConfig, parse_config
from hms.common import get_rotation_matrix, to_numpy, to_list, get_device

import nvdiffrec.render as nvdr

def load_envmap(filepath, device="cpu"):

    envmap = torch.tensor(imageio.imread(filepath, format="HDR-FI"), device=device)

    alpha = torch.ones((*envmap.shape[:2], 1), device=device)
    envmap = torch.cat((envmap, alpha), dim=-1)
    return envmap

@torch.no_grad()
def merge_texture_data(materials, texcoords, face_uv_indices, max_res):

    device = texcoords[0].device
    num_faces = [len(fti) for fti in face_uv_indices]
    num_texcoords = [len(tc) for tc in texcoords]

    texcoords = to_numpy(torch.cat(texcoords, axis=0)).tolist()

    face_uv_indices = [fti + offset for fti, offset in zip(face_uv_indices, [0] + np.cumsum(num_texcoords).tolist()[:-1])]
    face_uv_indices = to_numpy(torch.cat(face_uv_indices, axis=0)).tolist()

    face_mat_indices = list()
    for mid in range(len(materials)):
        face_mat_indices.extend([mid] * num_faces[mid])

    material, texcoords, face_uv_indices = nvdr.material.merge_materials(materials, texcoords, face_uv_indices, face_mat_indices, max_res)
    texcoords = torch.as_tensor(texcoords, dtype=torch.float32, device=device)
    face_uv_indices = torch.as_tensor(face_uv_indices, dtype=torch.int64, device=device)

    return material, texcoords, face_uv_indices

def compute_vertex_normals(v_pos, t_pos_idx):
    i0 = t_pos_idx[:, 0]
    i1 = t_pos_idx[:, 1]
    i2 = t_pos_idx[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.linalg.cross(v1 - v0, v2 - v0)

    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    v_nrm = torch.where(
        nvdr.util.dot(v_nrm, v_nrm) > 1e-20,
        v_nrm,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda"),
    )
    v_nrm = nvdr.util.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm

def get_projection_matrix(fx, fy, width, height, near=0.1, far=100.0):

    return torch.as_tensor(
        [
            [2 * fx / width, 0, 0, 0],
            [0, 2 * fy / height, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ]
    ).float()

def get_focal_length(width, height):

    focal_length = (width**2 + height**2) ** 0.5

    return focal_length

def render_backgrounds(envmap, mvp_mats, h, w, device="cuda"):

    assert mvp_mats.ndim == 3
    nviews = mvp_mats.shape[0]
    pos_int = torch.arange(w * h, device=device).int()
    pos = torch.stack((pos_int % w, pos_int // w), dim=1) / torch.tensor((w, h), device=device).float()
    pos[..., 0] = 2 * pos[..., 0] - 1
    pos[..., 1] = 1 - 2 * pos[..., 1]
    ones = torch.ones((w * h, 1), device=device).float()
    pos_near = torch.cat((pos, -ones, ones), dim=1).view(1, w * h, 4, 1)
    pos_far = torch.cat((pos, ones, ones), dim=1).view(1, w * h, 4, 1)
    inv_mvp = torch.linalg.inv(mvp_mats).view(nviews, 1, 4, 4)
    pos_near = (inv_mvp @ pos_near).squeeze(-1)
    pos_near = pos_near[..., :3] / pos_near[..., 3:]
    pos_far = (inv_mvp @ pos_far).squeeze(-1)
    pos_far = pos_far[..., :3] / pos_far[..., 3:]
    rays = pos_far - pos_near
    rays_view = rays / torch.norm(rays, dim=2, keepdim=True)
    rays_view = rays_view.reshape((nviews, h, w, -1))
    theta = torch.acos(rays_view[..., 1])
    phi = torch.atan2(rays_view[..., 0], rays_view[..., 2])
    envmap_uvs = torch.stack([1 - phi / (2 * np.pi), theta / np.pi], dim=-1)
    bgs = dr.texture(envmap[None, ...], envmap_uvs, filter_mode="linear")
    bgs = bgs.flip(dims=(1,))
    bgs[..., -1] = 0
    return bgs

def render_mesh(
    glctx,
    vertices,
    vertex_lights,
    faces,
    uvs,
    uv_indices,
    proj_mat,
    texture,
    backgrounds,
    width,
    height,
    boost=1.0,
):

    vertices_hom = F.pad(vertices, (0, 1), "constant", 1.0)
    vertices_clip = vertices_hom @ proj_mat.transpose(1, 2)

    rast, _ = dr.rasterize(glctx, vertices_clip, faces, (height, width))

    texc, _ = dr.interpolate(uvs.unsqueeze(0), rast, uv_indices)
    colors = dr.texture(texture, texc, filter_mode="linear")
    colors = F.pad(colors, (0, 1), "constant", 1.0)

    if vertex_lights is not None:
        lgtc, _ = dr.interpolate(vertex_lights, rast, faces)
        lgtc = F.pad(lgtc / np.pi, (0, 1), "constant", 1.0)
        lgtc = lgtc.clamp(0.0, 1.0).pow(1 / 2.2)
        colors = colors * lgtc

    cam_coords, _ = dr.interpolate(vertices, rast, faces)

    face_ids = rast[..., -1:].int() - 1

    mask = face_ids >= 0
    if backgrounds is not None:
        rgb = torch.where(mask, colors, backgrounds)
    else:
        rgb = colors
    rgb = dr.antialias(rgb, rast, vertices_clip, faces, pos_gradient_boost=boost)
    rgb = rgb[..., :-1]
    rgb_nobg = colors[..., :-1]

    return {
        "rgb": rgb.clamp(0, 1).flip(1),
        "rgb_nobg": rgb_nobg.clamp(0, 1).flip(1),
        "cam_coords": cam_coords.flip(1),
        "face_ids": face_ids.flip(1),
        "mask": mask.flip(1).float(),
    }

class SphericalHarmonics:

    def __init__(self, envmap, device):

        self.device = device

        h, w = envmap.shape[:2]

        theta = (torch.linspace(0, np.pi, h, device=device)).repeat(w, 1).t()
        phi = (torch.linspace(3 * np.pi, np.pi, w, device=device)).repeat(h, 1)

        sin_theta = torch.sin(theta)

        x = sin_theta * torch.cos(phi)
        z = -sin_theta * torch.sin(phi)
        y = torch.cos(theta)

        Y_0 = 0.282095

        Y_1 = [0.488603 * z, 0.488603 * x, 0.488603 * y]
        Y_2 = [
            0.315392 * (3 * z.square() - 1),
            1.092548 * x * z,
            0.546274 * (x.square() - y.square()),
            1.092548 * x * y,
            1.092548 * y * z,
        ]

        area = w * h
        radiance = envmap[..., :3]
        dt_dp = 2.0 * np.pi**2 / area

        L = [
            [(radiance * Y_0 * (sin_theta)[..., None] * dt_dp).sum(dim=(0, 1))],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0, 1)) for y in Y_1],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0, 1)) for y in Y_2],
        ]

        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        self.M = torch.stack(
            [
                torch.stack([c1 * L[2][2], c1 * L[2][-2], c1 * L[2][1], c2 * L[1][1]]),
                torch.stack([c1 * L[2][-2], -c1 * L[2][2], c1 * L[2][-1], c2 * L[1][-1]]),
                torch.stack([c1 * L[2][1], c1 * L[2][-1], c3 * L[2][0], c2 * L[1][0]]),
                torch.stack([c2 * L[1][1], c2 * L[1][-1], c2 * L[1][0], c4 * L[0][0] - c5 * L[2][0]]),
            ]
        ).movedim(2, 0)

    def eval(self, n):

        normal_array = n.view((-1, 3))
        h_n = F.pad(normal_array, (0, 1), "constant", 1.0)
        l = (h_n.t() * (self.M @ h_n.t())).sum(dim=1)
        return l.t().view(n.shape)

class NVRenderer(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = parse_config(NVRendererConfig, cfg)
        self.device = get_device()

        self.configure(*args, **kwargs)

    def configure(self, glctx, objects_to_render, *args, **kwargs):
        assert isinstance(objects_to_render, (list, tuple))
        self.objects_to_render = objects_to_render

        envmap = load_envmap(self.cfg.envmap_path).to(self.device)
        self.register_buffer("envmap", envmap)

        self.sh = SphericalHarmonics(envmap * self.cfg.envmap_sh_scale, self.device)

        if glctx is None:
            self.glctx = dr.RasterizeCudaContext(self.device)
        else:
            self.glctx = glctx

        ardata = [o2r.aux_render_data() for o2r in objects_to_render]

        face_ranges = list()
        fcount = 0
        for olist in ardata:
            nf = sum([odict["faces"].shape[0] for odict in olist])
            face_ranges.append((fcount, fcount + nf))
            fcount += nf
        self.face_ranges = face_ranges

        materials_arg = list()
        texcoords_arg = list()
        face_uv_indices_arg = list()
        for olist in ardata:
            for odict in olist:
                materials_arg.append(
                    {
                        "name": "mat",
                        "bsdf": "pbr",
                        "kd": odict["kd"],
                        "ks": odict["ks"],
                    }
                )
                texcoords_arg.append(odict["uvs"])
                face_uv_indices_arg.append(odict["face_uv_indices"])

        material, texcoords, face_uv_indices = merge_texture_data(
            materials=materials_arg,
            texcoords=texcoords_arg,
            face_uv_indices=face_uv_indices_arg,
            max_res=to_list(self.cfg.texture_size),
        )
        texcoords = texcoords.to(self.device)
        face_uv_indices = face_uv_indices.to(self.device)

        self.register_buffer("texcoords", texcoords)
        self.register_buffer("face_uv_indices", face_uv_indices)

        self.material = nvdr.material.Material(
            {
                "name": material["name"],
                "bsdf": material["bsdf"],

                "kd": nvdr.texture.create_trainable(material["kd"], auto_mipmaps=True),
                "ks": nvdr.texture.create_trainable(material["ks"], auto_mipmaps=True),
            }
        )

        env_xform = np.identity(4, dtype=np.float32)
        for axis, angle in zip("xy", [180, -90]):
            env_xform = get_rotation_matrix(axis, angle) @ env_xform
        env_xform = torch.as_tensor(env_xform).float().to(self.device)
        self.register_buffer("env_xform", env_xform)

        normal_xform = np.identity(4, dtype=np.float32)
        for axis, angle in zip("xy", [180, 90]):
            normal_xform = get_rotation_matrix(axis, angle) @ normal_xform
        normal_xform = torch.as_tensor(normal_xform[:3, :3]).float().to(self.device)
        self.register_buffer("normal_xform", normal_xform)

    def forward(self, c2w, center, **kwargs):
        width, height = self.cfg.viewport_width, self.cfg.viewport_height

        focal_length = get_focal_length(width=width, height=height)
        proj_mat = get_projection_matrix(focal_length, focal_length, width, height)
        proj_mat = proj_mat.unsqueeze(0).to(self.device)
        mv_mat = torch.linalg.inv(c2w)

        bgs = render_backgrounds(
            envmap=self.envmap * self.cfg.envmap_bg_scale,
            mvp_mats=proj_mat @ mv_mat @ self.env_xform.unsqueeze(0),
            h=height,
            w=width,
            device=self.device,
        )
        bgs = bgs.clamp(0.0, 1.0).pow(1 / 2.2)

        vertices = list()
        faces = list()
        vcount = 0
        for oid in range(len(self.objects_to_render)):
            gdict = self.objects_to_render[oid]()
            vs = gdict["vertices"]
            fs = gdict["faces"]
            start, end = self.face_ranges[oid]
            assert end - start == fs.shape[0]
            vertices.append(vs)
            faces.append(fs + vcount)
            vcount += vs.shape[1]
        vertices = torch.cat(vertices, dim=1)
        faces = torch.cat(faces, dim=0).int()
        face_uv_indices = self.face_uv_indices.int()
        texcoords = self.texcoords

        vertex_normals = [compute_vertex_normals(vertices[idx], faces.long()) for idx in range(vertices.shape[0])]
        vertex_normals = torch.stack(vertex_normals, dim=0)
        normal_xform = self.normal_xform.unsqueeze(0)
        vertex_normals = vertex_normals @ normal_xform.transpose(1, 2)
        vertex_lights = self.sh.eval(vertex_normals).contiguous()

        if center is None:
            center = torch.zeros(3)
        trans_mat = torch.eye(4).to(proj_mat)
        trans_mat[:3, 3] = -center.to(proj_mat)
        trans_mat = trans_mat.unsqueeze(0)

        mv_mat = mv_mat @ trans_mat
        vertices = F.pad(vertices, (0, 1), "constant", 1.0)
        vertices = vertices @ mv_mat.transpose(1, 2)
        vertices = vertices[..., :3].contiguous()

        rdict = render_mesh(
            glctx=self.glctx,
            vertices=vertices,
            vertex_lights=vertex_lights,
            faces=faces,
            uvs=texcoords,
            uv_indices=face_uv_indices,
            proj_mat=proj_mat,
            texture=self.material["kd"].data,
            backgrounds=bgs,
            width=width,
            height=height,
            boost=self.cfg.boost,
        )
        cam_coords = F.pad(rdict["cam_coords"], (0, 1), "constant", 1.0)
        mv_mat_inv = torch.linalg.inv(mv_mat).view(-1, 1, 4, 4)
        world_coords = cam_coords @ mv_mat_inv.transpose(2, 3)
        world_coords = world_coords[..., :3]

        obj_masks = list()
        for oid in range(len(self.objects_to_render)):
            start, end = self.face_ranges[oid]
            omask = torch.logical_and(rdict["face_ids"] >= start, rdict["face_ids"] < end)
            omask = torch.logical_and(omask, rdict["mask"] > 0)
            obj_masks.append(omask.float())

        res = {
            "comp_rgb": rdict["rgb"],
            "comp_rgb_nobg": rdict["rgb_nobg"],
            "comp_mask": rdict["mask"],
            "comp_cam_coords": rdict["cam_coords"],
            "comp_world_coords": world_coords,
            "comp_face_ids": rdict["face_ids"],
            "mv_mat": mv_mat.squeeze(0),
            "proj_mat": proj_mat.squeeze(0),
            "obj_masks": obj_masks,
        }

        return res
