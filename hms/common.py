import json
import os
import os.path as osp
import re
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import scipy
import scipy.sparse
import open3d as o3d
import imageio
import hashlib
from pathlib import Path
from scipy.spatial import cKDTree

class KNNSearch(object):
    DTYPE = np.float32
    WORKERS = 4

    def __init__(self, data):

        self.data = np.asarray(data, dtype=self.DTYPE)
        self.kdtree = cKDTree(self.data)

    def query(self, kpts, k, return_dists=False):

        kpts = np.asarray(kpts, dtype=self.DTYPE)
        nndists, nnindices = self.kdtree.query(kpts, k=k, workers=self.WORKERS)
        if return_dists:
            return nnindices, nndists
        else:
            return nnindices

    def query_ball(self, kpt, radius):

        kpt = np.asarray(kpt, dtype=self.DTYPE)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, workers=self.WORKERS)
        return nnindices

def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False

def parent_folder(file_path):
    return str(Path(file_path).parent)

def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)

def list_files(folder_path, name_filter, alphanum_sort=False, full_path=False):
    file_list = [p.name for p in list(Path(folder_path).glob(name_filter)) if p.is_file()]
    if alphanum_sort:
        file_list = sorted_alphanum(file_list)
    else:
        file_list = sorted(file_list)
    if full_path:
        file_list = [osp.join(folder_path, fn) for fn in file_list]
    return file_list

def read_lines(file_path):
    with open(file_path, "r") as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    return lines

def read_json(filepath):
    with open(filepath, "r") as fh:
        ret = json.load(fh)
    return ret

def write_json(filepath, data):
    assert isinstance(data, (dict, tuple, list))
    with open(filepath, "w") as fh:
        fh.write(json.dumps(data, indent=4))

def write_yaml(filepath, data, flow_style=False):
    assert isinstance(data, (dict, tuple, list))
    with open(filepath, "w") as fh:
        yaml.dump(data, fh, default_flow_style=flow_style)

def valid_str(x):
    return x is not None and isinstance(x, str) and x != ""

def strip_str(x):
    assert isinstance(x, str)
    res = list()
    for i in x.strip():
        if i.isalnum():
            res.append(i)
        else:
            if len(res) != 0 and res[-1] != "_":
                res.append("_")
    return "".join(res)

def hash_str(x):
    assert isinstance(x, str)
    x = x.encode("utf-8")
    sha256_hash = hashlib.sha256()
    sha256_hash.update(x)
    return sha256_hash.hexdigest()

def linear_weights(w_start, w_end, steps, output_type="list", device="cpu"):
    weights = np.linspace(w_start, w_end, steps + 1)
    weights = weights[:-1]
    if output_type == "pt":
        return torch.as_tensor(weights).float().to(device)
    elif output_type == "np":
        return weights.astype(np.float32)
    elif output_type == "list":
        return weights.tolist()
    else:
        raise RuntimeError(f"[!] {output_type} is not supported!")

def normalize(x, axis, eps=1e-6):
    if isinstance(x, np.ndarray):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        norm = np.clip(norm, a_min=eps, a_max=None)
        return x / norm
    elif isinstance(x, torch.Tensor):
        return F.normalize(x, dim=axis, eps=eps)
    else:
        raise RuntimeError(f"[!] input type is not supported!")

def get_translate_matrix(*args):
    if len(args) == 1:
        if isinstance(args[0], float):
            tx, ty, tz = args[0], args[0], args[0]
        elif isinstance(args[0], (np.ndarray, list, tuple)):
            assert len(args[0]) == 3
            tx, ty, tz = args[0][0], args[0][1], args[0][2]
        else:
            raise RuntimeError("[!] Wrong input arguments!")
    elif len(args) == 3:
        assert isinstance(args[0], float)
        tx, ty, tz = args
    else:
        raise RuntimeError("[!] Wrong input arguments!")
    res = np.identity(4, dtype=np.float32)
    res[0, 3] = tx
    res[1, 3] = ty
    res[2, 3] = tz
    return res

def get_rotation_matrix(axis, theta):
    theta = theta * math.pi / 180.0
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    if axis == "x":
        rot = np.asarray(
            [
                [1, 0, 0],
                [0, costheta, -sintheta],
                [0, sintheta, costheta],
            ]
        )
    elif axis == "y":
        rot = np.asarray(
            [
                [costheta, 0, sintheta],
                [0, 1, 0],
                [-sintheta, 0, costheta],
            ]
        )
    elif axis == "z":
        rot = np.asarray(
            [
                [costheta, -sintheta, 0],
                [sintheta, costheta, 0],
                [0, 0, 1],
            ]
        )
    else:
        raise RuntimeError(f"[!] axis {axis} is not supported!")
    res = np.identity(4, dtype=np.float32)
    res[:3, :3] = rot
    return res

def apply_transform3d(x, xform):
    if torch.is_tensor(x):
        assert torch.is_tensor(xform)
        if x.numel() == 0:
            return x
        assert x.ndim == 2 and x.shape[-1] == 3
        assert xform.ndim == 2
        x = F.pad(x, (0, 1), "constant", 1.0)
        x = x @ xform.transpose(0, 1)
        x = x[..., :3].contiguous()
        return x
    elif isinstance(x, np.ndarray):
        assert isinstance(xform, np.ndarray)
        if x.size == 0:
            return x
        assert x.ndim == 2 and x.shape[-1] == 3
        assert xform.ndim == 2 and xform.shape == (4, 4)
        x = np.concatenate([x, np.ones_like(x[:, :1])], axis=1)
        x = x @ xform.T
        x = x[:, :3]
        return x
    else:
        raise RuntimeError("[!] x and xform must be either torch.Tensor or np.ndarray!")

def apply_transforms3d(x, xforms):
    if torch.is_tensor(x):
        assert torch.is_tensor(xforms)
        if x.numel() == 0:
            return x
        assert x.ndim == 3 and x.shape[-1] == 3
        if xforms.ndim == 2:
            xforms = xforms.unsqueeze(0)
        assert xforms.ndim == 3
        x = F.pad(x, (0, 1), "constant", 1.0)
        x = x @ xforms.transpose(1, 2)
        x = x[..., :3].contiguous()
        return x
    elif isinstance(x, np.ndarray):
        assert isinstance(xforms, np.ndarray)
        if x.size == 0:
            return x
        assert x.ndim == 3 and x.shape[-1] == 3
        if xforms.ndim == 2:
            xforms = np.expand_dims(xforms, axis=0)
        assert xforms.ndim == 3
        x = np.concatenate([x, np.ones_like(x[:, :, :1])], axis=2)
        x = x @ np.transpose(xforms, (0, 2, 1))
        x = x[..., :3]
        return x
    else:
        raise RuntimeError("[!] x and xforms must be either torch.Tensor or np.ndarray!")

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

def get_rank():

    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

def get_device():
    return torch.device(f"cuda:{get_rank()}")

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, (scipy.sparse.coo_matrix, scipy.sparse.csc_matrix)):
        return sparse_np_to_torch(x)
    elif isinstance(x, (list, tuple)):
        return [to_torch(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return torch.as_tensor(x)
    else:
        return x

def to_cuda(x, device="cuda"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return [to_cuda(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return [to_numpy(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return np.asarray(x)
    else:
        return x

def to_list(x):
    if len(x) == 0:
        return list()
    if isinstance(x[0], (int, float, bool)):
        return [item for item in x]
    else:
        return [to_list(item) for item in x]

def seeding(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def to_o3d_mesh(V, F, VC=None):
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.copy(V)),
        o3d.utility.Vector3iVector(np.copy(F)),
    )
    if VC is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(np.copy(VC))
    return m

def load_o3d_pcd(filepath):
    pcd = o3d.t.io.read_point_cloud(filepath)
    out = dict()
    for k, v in pcd.point.items():
        if k == "positions":
            V = v.numpy().astype(np.float32)
            out["V"] = V
        elif k == "colors":
            VC = v.numpy()
            if VC.dtype != np.uint8:
                VC = np.clip(VC, 0, 1) * 255
            out["VC"] = VC.astype(np.uint8)
        elif k == "normals":
            VN = v.numpy().astype(np.float32)
            out["VN"] = VN
    return out

def save_o3d_pcd(filepath, V, VN=None, VC=None):
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(V, dtype=dtype)
    if VN is not None:
        pcd.point.normals = o3d.core.Tensor(VN, dtype=dtype)
    if VC is not None:
        if VC.dtype != np.uint8:
            VC = np.clip(VC, 0, 1) * 255
        VC = VC.astype(np.uint8)
        pcd.point.colors = o3d.core.Tensor(VC, dtype=o3d.core.uint8)
    return o3d.t.io.write_point_cloud(filepath, pcd)

def save_o3d_mesh(filepath, V, F, VC=None):
    dtype = o3d.core.float32
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(V, dtype=dtype)
    mesh.triangle.indices = o3d.core.Tensor(F, dtype=o3d.core.int32)
    if VC is not None:
        if VC.dtype != np.uint8:
            VC = np.clip(VC, 0, 1) * 255
        VC = VC.astype(np.uint8)
        mesh.vertex.colors = o3d.core.Tensor(VC, dtype=o3d.core.uint8)
    return o3d.t.io.write_triangle_mesh(filepath, mesh)

def load_mtl(filepath, verbose=False):
    folder = parent_folder(filepath)
    lines = read_lines(filepath)
    materials = list()
    for line in lines:
        if line.startswith("#"):
            continue
        items = line.split()
        if items[0] == "newmtl":
            materials.append({"material_name": items[1]})
        elif items[0] in ("Ka", "Kd", "Ks", "Ke"):
            materials[-1][items[0]] = np.asarray(list(map(float, items[1:4])), dtype=np.float32)
        elif items[0] in ("Ns", "Ni", "d"):
            materials[-1][items[0]] = float(items[1])
        elif items[0] == "Tr":
            materials[-1]["d"] = 1 - float(items[1])
        elif items[0] == "illum":
            materials[-1][items[0]] = int(items[1])
        elif items[0] in (
            "map_Ka",
            "map_Kd",
            "map_Ks",
            "map_Ke",
            "map_Ns",
            "map_Bump",
            "bump",
        ):
            image = imageio.imread(osp.join(folder, items[-1]))

            materials[-1][items[0]] = image
        else:
            if verbose:
                print(f"[*] load_mtl: skipping {line}")
    return materials

def load_obj(filepath, verbose=False):
    lines = read_lines(filepath)

    vertices = list()
    vertex_normals = list()
    uvs = list()
    faces = list()
    face_uv_indices = list()
    face_normal_indices = list()
    material_assigns = list()
    materials = list()

    def get_material_id(name):
        for idx, mat in enumerate(materials):
            if mat["material_name"] == name:
                return idx
        return None

    for line in lines:
        if line.startswith("mtllib"):
            mats = load_mtl(osp.join(parent_folder(filepath), line.split()[-1]), verbose=verbose)
            materials += mats

    material_id = None
    face_count = 0
    for line in lines:
        if line.startswith("#"):
            continue
        items = line.split()
        if items[0] == "v":
            vertices.append(list(map(float, items[1:4])))
        elif items[0] == "vn":
            vertex_normals.append(list(map(float, items[1:4])))
        elif items[0] == "vt":
            uvs.append(list(map(float, items[1:3])))
        elif items[0] == "f":
            indices = [list(), list(), list()]
            for item in items[1:4]:
                for idx, val in enumerate(item.split("/")):
                    if len(val) == 0:
                        continue
                    indices[idx].append(int(val) - 1)
            faces.append(indices[0])
            face_uv_indices.append(indices[1])
            face_normal_indices.append(indices[2])
            face_count += 1
        elif items[0] == "usemtl":
            if face_count > 0:
                material_assigns.extend([material_id] * face_count)
            material_id = get_material_id(items[1])
            face_count = 0
        else:
            if verbose:
                print(f"[*] load_obj: skipping {line}")
    if face_count > 0:
        material_assigns.extend([material_id] * face_count)

    vertices = np.asarray(vertices, dtype=np.float32)
    vertex_normals = np.asarray(vertex_normals, dtype=np.float32)
    if vertex_normals.size > 0:
        vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    uvs = np.asarray(uvs, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    face_uv_indices = np.asarray(face_uv_indices, dtype=np.int32)
    face_normal_indices = np.asarray(face_normal_indices, dtype=np.int32)
    material_assigns = np.asarray(material_assigns, dtype=np.int32)

    return (
        vertices,
        vertex_normals,
        uvs,
        faces,
        face_uv_indices,
        face_normal_indices,
        material_assigns,
        materials,
    )

def load_smplx_uv(template_path, texture_path, verbose=False):
    lines = read_lines(template_path)
    uvs = list()
    faces = list()
    face_uv_indices = list()
    face_normal_indices = list()
    for line in lines:
        if line.startswith("#"):
            continue
        items = line.split()
        if items[0] == "vt":
            uvs.append(list(map(float, items[1:3])))
        elif items[0] == "f":
            indices = [list(), list(), list()]
            for item in items[1:4]:
                for idx, val in enumerate(item.split("/")):
                    if len(val) == 0:
                        continue
                    indices[idx].append(int(val) - 1)
            faces.append(indices[0])
            face_uv_indices.append(indices[1])
            face_normal_indices.append(indices[2])
        else:
            if verbose:
                print(f"[*] load_obj: skipping {line}")
    uvs = np.asarray(uvs, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    face_uv_indices = np.asarray(face_uv_indices, dtype=np.int32)
    face_normal_indices = np.asarray(face_normal_indices, dtype=np.int32)

    texture = imageio.imread(texture_path)[..., :3]

    return {
        "uvs": uvs,
        "faces": faces,
        "face_uv_indices": face_uv_indices,
        "face_normal_indices": face_normal_indices,
        "map_Kd": texture,
        "map_Ks": np.asarray((0.5, 0.5, 0.5)),
    }
