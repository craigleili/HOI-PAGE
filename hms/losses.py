import os.path as osp
import sys
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.common import to_torch, get_device, apply_transform3d, apply_transforms3d, linear_weights
from hms.config import LiftingLossConfig, parse_config

def l2_loss(x, y, dim=-1):
    return ((x - y) ** 2).sum(dim).mean()

def simple_static_loss(x):

    return l2_loss(x[..., 1:, :], x[..., :-1, :])

def simple_smoothness_loss(x):

    return l2_loss(x[..., 1:-1, :], 0.5 * (x[..., :-2, :] + x[..., 2:, :]))

def rotation_static_loss(x):

    x = roma.rotmat_to_rotvec(x)
    return simple_static_loss(x)

def rotation_smoothness_loss(x):

    interp = roma.rotmat_slerp(x[:-2], x[2:], torch.tensor(0.5).to(x))
    diff = roma.rotmat_geodesic_distance(interp, x[1:-1])
    return (diff**2).mean()

def geman_mcclure_func(residual, rho=0.2):
    squared_res = residual**2
    dist = torch.div(squared_res, squared_res + rho**2)
    return rho**2 * dist

def pcd_distance(p1, p2, reduction="min", error_func=None):
    if p1 is None or p2 is None:
        return None
    assert p1.ndim == p2.ndim == 3
    nnres = knn_points(p1=p1, p2=p2, norm=2, K=1)
    nndists = nnres.dists[..., 0]
    if error_func is not None:
        nndists = error_func(nndists)
    if reduction == "min":
        dists = torch.min(nndists, dim=1)[0]
    elif reduction == "mean":
        dists = torch.mean(nndists, dim=1)
    else:
        raise RuntimeError(f"Unknown reduction: {reduction}")
    return dists

def project_points_with_intrinsics(points, intrinsics):
    assert points.ndim == 2
    assert intrinsics.ndim == 2
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    px = points[:, 0] * fx / points[:, 2] + cx
    py = points[:, 1] * fy / points[:, 2] + cy
    d = points[:, 2]
    return torch.stack([px, py, d], dim=-1)

def project_points_gl(points, mv_mat, proj_mat, image_height, image_width):
    assert points.ndim == 2
    assert mv_mat.ndim == 2
    assert proj_mat.ndim == 2
    points = F.pad(points, (0, 1), "constant", 1.0)
    points_cam = points @ mv_mat.T
    points = points_cam @ proj_mat.T
    points = points[:, :3] / torch.clamp(points[:, 3:], min=1e-6)
    px = (points[:, 0] + 1) * 0.5 * image_width
    py = (1 - points[:, 1]) * 0.5 * image_height
    d = -points_cam[:, 2]
    return torch.stack([px, py, d], dim=-1)

class Video2DData(nn.Module):
    def __init__(
        self,
        object_masks,
        object_part_masks,
        human_masks,
        c2w,
        mv_mat,
        proj_mat,
        **kwargs,
    ):
        super().__init__()
        for k, w in kwargs.items():
            setattr(self, k, w)

        device = get_device()
        height, width = human_masks[0].shape[-2:]

        object_pixels = list()
        for oid in range(len(object_masks)):

            omasks = to_torch(object_masks[oid]).bool().to(device)
            opixels = [

                torch.nonzero(omasks[i], as_tuple=False)[:, [1, 0]].float()
                for i in range(omasks.shape[0])
            ]
            object_pixels.append(opixels)

        object_part_pixels = list()
        for oid in range(len(object_part_masks)):
            oppixels = dict()
            for pname, pmasks in object_part_masks[oid].items():

                pmasks = to_torch(pmasks).bool().to(device)
                ppixels = [

                    torch.nonzero(pmasks[i], as_tuple=False)[:, [1, 0]].float()
                    for i in range(pmasks.shape[0])
                ]
                oppixels[pname] = ppixels
            object_part_pixels.append(oppixels)

        human_pixels = list()
        for i in range(len(human_masks)):

            hmasks = to_torch(human_masks[i]).bool().to(device)
            hpixels = [

                torch.nonzero(hmasks[i], as_tuple=False)[:, [1, 0]].float()
                for i in range(hmasks.shape[0])
            ]
            human_pixels.append(hpixels)

        c2w = to_torch(c2w).float().to(device)
        mv_mat = to_torch(mv_mat).float().to(device)
        proj_mat = to_torch(proj_mat).float().to(device)

        self.object_pixels = object_pixels
        self.object_part_pixels = object_part_pixels
        self.human_pixels = human_pixels
        self.c2w = c2w
        self.mv_mat = mv_mat
        self.proj_mat = proj_mat
        self.height = height
        self.width = width
        self.device = device

    def forward(self):
        return {
            "object_pixels": self.object_pixels,
            "object_part_pixels": self.object_part_pixels,
            "human_pixels": self.human_pixels,
            "c2w": self.c2w,
            "mv_mat": self.mv_mat,
            "proj_mat": self.proj_mat,
            "height": self.height,
            "width": self.width,
        }

class Video3DData(Video2DData):
    def __init__(
        self,
        intrinsics,
        object_points,
        object_masks,
        object_part_points,
        object_part_masks,
        human_points,
        human_masks,
        human_poses,
        c2w,
        mv_mat,
        proj_mat,
        n_joints,
        **kwargs,
    ):
        super().__init__(object_masks, object_part_masks, human_masks, c2w, mv_mat, proj_mat, **kwargs)

        self.n_joints = n_joints
        device = self.device

        intrinsics = to_torch(intrinsics).float().to(device)

        object_points = [[to_torch(op).float().to(device) for op in opoints] for opoints in object_points]

        object_part_points = [
            {opname: [to_torch(opf).float().to(device) for opf in opframes] for opname, opframes in opdict.items()} for opdict in object_part_points
        ]

        human_points = [[to_torch(hp).float().to(device) for hp in hpoints] for hpoints in human_points]

        humans_transl = list()
        humans_global_orient = list()
        humans_betas = list()
        humans_body_pose = list()
        for hpdict in human_poses:
            humans_transl.append(to_torch(hpdict["transl"]).float().to(device))
            humans_global_orient.append(to_torch(hpdict["global_orient"]).float().to(device))
            humans_betas.append(to_torch(hpdict["betas"]).float().to(device))
            humans_body_pose.append(to_torch(hpdict["body_pose"][..., : n_joints * 3]).to(device))

        self.intrinsics = intrinsics
        self.object_points = object_points
        self.object_part_points = object_part_points
        self.human_points = human_points
        self.humans_transl = humans_transl
        self.humans_global_orient = humans_global_orient
        self.humans_betas = humans_betas
        self.humans_body_pose = humans_body_pose

        n_frames = len(human_points[0])
        rotations = torch.eye(3).unsqueeze(0).repeat(n_frames, 1, 1).float().to(device)
        rotations = rotations[..., :2].contiguous().view(n_frames, 3 * 2)
        self.pmap_scales = nn.Parameter(torch.zeros(n_frames).float().to(device))
        self.pmap_rotations = nn.Parameter(rotations)
        self.pmap_translations = nn.Parameter(torch.zeros(n_frames, 3).float().to(device))

    def get_pmap_xforms(self):
        scales = torch.exp(self.pmap_scales).view(-1, 1, 1)
        rotations = self.pmap_rotations.view(-1, 3, 2)
        rotations = roma.special_gramschmidt(rotations)
        translations = self.pmap_translations.unsqueeze(2)
        pads = torch.as_tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(rotations.shape[0], 1, 1).to(rotations)
        xform = torch.cat((scales * rotations, translations), dim=2)
        xform = torch.cat((xform, pads), dim=1)
        return xform

    def forward(self):
        pmap_xform = self.get_pmap_xforms()
        pmap_xform_inv = torch.linalg.inv(pmap_xform)
        object_points = [[apply_transform3d(op, pmap_xform[i]) for i, op in enumerate(opoints)] for opoints in self.object_points]
        object_part_points = [
            {opname: [apply_transform3d(opf, pmap_xform[i]) for i, opf in enumerate(opframes)] for opname, opframes in opdict.items()}
            for opdict in self.object_part_points
        ]
        human_points = [[apply_transform3d(hp, pmap_xform[i]) for i, hp in enumerate(hpoints)] for hpoints in self.human_points]
        return {
            "intrinsics": self.intrinsics,
            "object_points": object_points,
            "object_pixels": self.object_pixels,
            "object_part_points": object_part_points,
            "object_part_pixels": self.object_part_pixels,
            "human_points": human_points,
            "human_pixels": self.human_pixels,
            "humans_transl": self.humans_transl,
            "humans_global_orient": self.humans_global_orient,
            "humans_betas": self.humans_betas,
            "humans_body_pose": self.humans_body_pose,
            "pmap_xform": pmap_xform,
            "pmap_xform_inv": pmap_xform_inv,
            "c2w": self.c2w,
            "mv_mat": self.mv_mat,
            "proj_mat": self.proj_mat,
            "height": self.height,
            "width": self.width,
        }

class LiftingLoss(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(LiftingLossConfig, cfg)
        self.device = get_device()

        self.configure(*args, **kwargs)

    def configure(self, humans, scenes, igraph, obj_segments, glctx, *args, **kwargs):
        self.humans = humans
        self.scenes = scenes
        self.igraph = igraph
        self.obj_segments = obj_segments
        self.glctx = glctx

        self.n_frames = self.humans[0].cfg.n_frames
        self.n_joints = self.humans[0].smplx_model.NUM_BODY_JOINTS

    def fit_humans(
        self,
        step_id,
        n_steps,
        video_data,
        use_human_cd3d=True,
        use_human_cd2d=True,
    ):
        def lw(name):
            weights = getattr(self.cfg, f"{name}_weight")[:2]
            return linear_weights(weights[0], weights[1], n_steps)[step_id]

        assert "intrinsics" in video_data

        human_data = list()
        hvertices_proj = list()
        for hid in range(len(self.humans)):
            hdata = self.humans[hid]()
            human_data.append(hdata)

            hvrast = apply_transforms3d(hdata["vertices"], video_data["pmap_xform_inv"])
            hvproj = [project_points_with_intrinsics(hvrast[j], video_data["intrinsics"][j]) for j in range(self.n_frames)]
            hvproj = torch.stack(hvproj, dim=0)
            hvertices_proj.append(hvproj)
        loss_terms = list()

        if use_human_cd3d:
            loss_human_cd3d = list()
            for hid in range(len(self.humans)):
                lh_cd3d = list()
                for j in range(self.n_frames):
                    hpoints = video_data["human_points"][hid][j]
                    if hpoints.shape[0] == 0:
                        continue
                    cdist = pcd_distance(hpoints.unsqueeze(0), human_data[hid]["vertices"][j].unsqueeze(0), reduction="mean")
                    cdist = cdist.squeeze(0)
                    cdist = geman_mcclure_func(cdist)
                    lh_cd3d.append(cdist)
                lh_cd3d = torch.stack(lh_cd3d, dim=0).mean()
                loss_human_cd3d.append(lh_cd3d)
            loss_human_cd3d = torch.stack(loss_human_cd3d, dim=0).mean()
            loss_terms.append([lw("human_cd3d"), loss_human_cd3d, "human_cd3d"])

        if use_human_cd2d:
            loss_human_cd2d = list()
            for hid in range(len(self.humans)):
                lh_cd2d = list()
                for j in range(self.n_frames):
                    hpixels = video_data["human_pixels"][hid][j]
                    if hpixels.shape[0] == 0:
                        continue
                    cdist = pcd_distance(hpixels.unsqueeze(0), hvertices_proj[hid][j, :, :2].unsqueeze(0), reduction="mean")
                    cdist = cdist.squeeze(0)
                    lh_cd2d.append(cdist)
                lh_cd2d = torch.stack(lh_cd2d, dim=0).mean()
                loss_human_cd2d.append(lh_cd2d)
            loss_human_cd2d = torch.stack(loss_human_cd2d, dim=0).mean()
            loss_terms.append([lw("human_cd2d"), loss_human_cd2d, "human_cd2d"])

        return loss_terms

    def fit_objects(
        self,
        step_id,
        n_steps,
        video_data=None,
        use_object_cd3d=True,
        use_object_cd2d=True,
        use_object_part_cd3d=True,
        use_object_part_cd2d=True,
        use_object_smooth_trans=True,
        use_object_smooth_rot=True,
        use_object_scale=True,
        use_intersect=True,
        use_nocontact=True,
        use_contact_drift=True,
    ):
        def lw(name):
            weights = getattr(self.cfg, f"{name}_weight")[:2]
            return linear_weights(weights[0], weights[1], n_steps)[step_id]

        has_intrinsics = "intrinsics" in video_data

        human_data = list()
        for hid in range(len(self.humans)):
            hdata = self.humans[hid]()
            human_data.append(hdata)

        scene_data = list()
        spcds_proj = list()
        for sid in range(len(self.scenes)):
            sdata = self.scenes[sid]()
            scene_data.append(sdata)

            svproj = list()
            for j in range(self.n_frames):
                spoints = sdata["pcds"][j]
                if has_intrinsics:
                    spoints = apply_transform3d(spoints, video_data["pmap_xform"][j])
                    spoints = project_points_with_intrinsics(spoints, video_data["intrinsics"][j])
                else:
                    spoints = project_points_gl(
                        points=spoints,
                        mv_mat=video_data["mv_mat"],
                        proj_mat=video_data["proj_mat"],
                        image_height=video_data["height"],
                        image_width=video_data["width"],
                    )
                svproj.append(spoints)
            svproj = torch.stack(svproj, dim=0)
            spcds_proj.append(svproj)

        def get_scene_id(name):
            for sid in range(len(self.scenes)):
                if scene_data[sid]["name"] == name:
                    return sid
            return None

        def get_human_id(name):
            return int(name.split(" ")[-1].strip()) - 1

        def get_reduction(nodes):
            for node in nodes:
                if node.obj.dtype == "human" and node.name.split(" ")[-1] in ["hand", "foot"]:
                    return "mean"
            return "min"

        def crop_human_part(human_name, part_name):
            hid = get_human_id(human_name)
            hpart_labeling = self.humans[hid].get_body_segmentations(part_name)
            if (~hpart_labeling).all().item():
                return None
            part_pcd = human_data[hid]["vertices"][:, hpart_labeling, :]
            return part_pcd

        def get_object_3d(object_name, frame_id):
            sid = get_scene_id(object_name)
            pcd3d = scene_data[sid]["pcds"][frame_id]
            return pcd3d

        def get_object_2d(object_name, frame_id):
            sid = get_scene_id(object_name)
            pcd2d = spcds_proj[sid][frame_id, :, :2]
            return pcd2d

        def crop_object_part_3d(object_name, part_name):
            sid = get_scene_id(object_name)
            object_parts = self.obj_segments[object_name]["object_part_labels"]
            object_labeling = self.obj_segments[object_name]["object_labeling"]
            opart_labeling = object_labeling == object_parts.index(part_name)
            if (~opart_labeling).all().item():
                return None
            part_pcd = scene_data[sid]["pcds"][:, opart_labeling, :]
            return part_pcd

        def crop_object_part_2d(object_name, part_name):
            sid = get_scene_id(object_name)
            object_parts = self.obj_segments[object_name]["object_part_labels"]
            object_labeling = self.obj_segments[object_name]["object_labeling"]
            opart_labeling = object_labeling == object_parts.index(part_name)
            if (~opart_labeling).all().item():
                return None
            part_pcd = spcds_proj[sid][:, opart_labeling, :2]
            return part_pcd

        object_names = self.igraph.get_objects()

        loss_terms = list()

        if use_object_cd3d:
            loss_object_cd3d = list()
            for oid, oname in enumerate(object_names):
                lo_cd3d = list()
                for j in range(self.n_frames):
                    opoints = video_data["object_points"][oid][j]
                    if opoints.shape[0] == 0:
                        continue
                    fpoints = get_object_3d(oname, j)
                    cdist = pcd_distance(opoints.unsqueeze(0), fpoints.unsqueeze(0), reduction="mean")
                    cdist = cdist.squeeze(0)
                    cdist = geman_mcclure_func(cdist)
                    lo_cd3d.append(cdist)
                if len(lo_cd3d) != 0:
                    lo_cd3d = torch.stack(lo_cd3d, dim=0).mean()
                    loss_object_cd3d.append(lo_cd3d)
            if len(loss_object_cd3d) != 0:
                loss_object_cd3d = torch.stack(loss_object_cd3d, dim=0).mean()
                loss_terms.append([lw("object_cd3d"), loss_object_cd3d, "object_cd3d"])

        if use_object_cd2d:
            loss_object_cd2d = list()
            for oid, oname in enumerate(object_names):
                lo_cd2d = list()
                for j in range(self.n_frames):
                    opixels = video_data["object_pixels"][oid][j]
                    if opixels.shape[0] == 0:
                        continue
                    fpixels = get_object_2d(oname, j)
                    cdist = pcd_distance(opixels.unsqueeze(0), fpixels.unsqueeze(0), reduction="mean")
                    cdist = cdist.squeeze(0)
                    lo_cd2d.append(cdist)
                if len(lo_cd2d) != 0:
                    lo_cd2d = torch.stack(lo_cd2d, dim=0).mean()
                    loss_object_cd2d.append(lo_cd2d)
            if len(loss_object_cd2d) != 0:
                loss_object_cd2d = torch.stack(loss_object_cd2d, dim=0).mean()
                loss_terms.append([lw("object_cd2d"), loss_object_cd2d, "object_cd2d"])

        if use_object_part_cd3d:
            loss_object_part_cd3d = list()
            for oid, oname in enumerate(object_names):
                lop_cd3d = list()
                for pname, ppoints in video_data["object_part_points"][oid].items():
                    spoints = crop_object_part_3d(oname, pname)
                    if spoints is None:
                        continue
                    for j in range(self.n_frames):
                        if ppoints[j].shape[0] == 0:
                            continue
                        cdist = pcd_distance(ppoints[j].unsqueeze(0), spoints[j].unsqueeze(0), reduction="mean")
                        cdist = cdist.squeeze(0)
                        cdist = geman_mcclure_func(cdist)
                        lop_cd3d.append(cdist)
                if len(lop_cd3d) != 0:
                    lop_cd3d = torch.stack(lop_cd3d, dim=0).mean()
                    loss_object_part_cd3d.append(lop_cd3d)
            if len(loss_object_part_cd3d) != 0:
                loss_object_part_cd3d = torch.stack(loss_object_part_cd3d, dim=0).mean()
                loss_terms.append([lw("object_part_cd3d"), loss_object_part_cd3d, "object_part_cd3d"])

        if use_object_part_cd2d:
            loss_object_part_cd2d = list()
            for oid, oname in enumerate(object_names):
                lop_cd2d = list()
                for pname, ppixels in video_data["object_part_pixels"][oid].items():
                    spoints = crop_object_part_2d(oname, pname)
                    if spoints is None:
                        continue
                    for j in range(self.n_frames):
                        if ppixels[j].shape[0] == 0:
                            continue
                        cdist = pcd_distance(ppixels[j].unsqueeze(0), spoints[j].unsqueeze(0), reduction="mean")
                        cdist = cdist.squeeze(0)
                        lop_cd2d.append(cdist)
                if len(lop_cd2d) != 0:
                    lop_cd2d = torch.stack(lop_cd2d, dim=0).mean()
                    loss_object_part_cd2d.append(lop_cd2d)
            if len(loss_object_part_cd2d) != 0:
                loss_object_part_cd2d = torch.stack(loss_object_part_cd2d, dim=0).mean()
                loss_terms.append([lw("object_part_cd2d"), loss_object_part_cd2d, "object_part_cd2d"])

        if use_object_smooth_trans:
            loss_object_smooth_trans = list()
            for sid in range(len(self.scenes)):
                ostate = self.igraph.get_object(scene_data[sid]["name"])
                if ostate is None:
                    continue
                object_trans = scene_data[sid]["translations"]
                if ostate.translational:
                    loss_object_smooth_trans.append(simple_smoothness_loss(object_trans))
                else:
                    loss_object_smooth_trans.append(simple_static_loss(object_trans) * 10)
            loss_object_smooth_trans = torch.stack(loss_object_smooth_trans, dim=0).mean()
            loss_terms.append([lw("object_smooth_trans"), loss_object_smooth_trans, "object_smooth_trans"])

        if use_object_smooth_rot:
            loss_object_smooth_rot = list()
            for sid in range(len(self.scenes)):
                ostate = self.igraph.get_object(scene_data[sid]["name"])
                if ostate is None:
                    continue
                object_rot = scene_data[sid]["rotations"]
                if self.scenes[sid].cfg.use_continous_rot_repr:
                    object_rot = object_rot.view(object_rot.shape[:-1] + (3, 2))
                    object_rot = roma.special_gramschmidt(object_rot)
                else:
                    object_rot = roma.rotvec_to_rotmat(object_rot)
                if ostate.rotational:
                    loss_object_smooth_rot.append(rotation_smoothness_loss(object_rot))
                else:
                    loss_object_smooth_rot.append(rotation_static_loss(object_rot) * 10)
            loss_object_smooth_rot = torch.stack(loss_object_smooth_rot, dim=0).mean()
            loss_terms.append([lw("object_smooth_rot"), loss_object_smooth_rot, "object_smooth_rot"])

        if use_object_scale:

            loss_object_scale = list()
            for sid in range(len(self.scenes)):
                object_scale = scene_data[sid]["scales"]
                loss_object_scale.append(F.relu(torch.abs(object_scale - 1) - 0.1))
            loss_object_scale = torch.stack(loss_object_scale, dim=0).mean()
            loss_terms.append([lw("object_scale"), loss_object_scale, "object_scale"])

        if use_intersect:
            loss_intersect = list()
            for hid in range(len(self.humans)):
                for sid in range(len(self.scenes)):
                    sdf_values = self.scenes[sid].get_sdf(human_data[hid]["vertices"], is_canonical=False)

                    intersects = F.relu(-sdf_values.flatten())
                    icount = (intersects > 0).sum()
                    if icount.item() == 0:
                        continue
                    linters = intersects.sum() / icount
                    loss_intersect.append(linters)
            if len(loss_intersect) != 0:
                loss_intersect = torch.stack(loss_intersect, dim=0).mean()
                loss_terms.append([lw("intersect"), loss_intersect, "intersect"])

        if use_nocontact or use_contact_drift:
            loss_nocontact = list()
            loss_contact_drift = list()
            visited = set()
            for iedge in self.igraph.edges:
                inodes = iedge.nodes
                has_hpart = inodes[0].obj.dtype == "human" or inodes[1].obj.dtype == "human"
                if has_hpart and inodes[0].obj.dtype != "human":
                    inodes = [inodes[1], inodes[0]]
                reduction = get_reduction(inodes)
                if (inodes[0].obj.name, inodes[0].name, inodes[1].obj.name, inodes[1].name) in visited:
                    continue
                if inodes[0].obj.dtype == "object":
                    scene_id = get_scene_id(inodes[0].obj.name)
                else:
                    scene_id = get_scene_id(inodes[1].obj.name)

                if has_hpart:

                    assert inodes[0].obj.dtype == "human"
                    assert inodes[1].obj.dtype == "object"
                    hname = inodes[0].obj.name
                    hpart = inodes[0].name.split(" ")[-1]
                    oname = inodes[1].obj.name
                    opart = inodes[1].name
                    if hpart in ["head", "hips"]:
                        visited.add((hname, hpart, oname, opart))
                        visited.add((oname, opart, hname, hpart))

                        pdists = pcd_distance(crop_human_part(hname, hpart), crop_object_part_3d(oname, opart), reduction=reduction)
                        if pdists is None:
                            continue
                        pcano = self.scenes[scene_id].to_canonical(crop_human_part(hname, hpart))
                    else:
                        visited.add((hname, f"left {hpart}", oname, opart))
                        visited.add((hname, f"right {hpart}", oname, opart))
                        visited.add((oname, opart, hname, f"left {hpart}"))
                        visited.add((oname, opart, hname, f"right {hpart}"))

                        has_left = self.igraph.has_interaction(hname, f"left {hpart}", oname, opart)
                        has_right = self.igraph.has_interaction(hname, f"right {hpart}", oname, opart)
                        if has_left and has_right:

                            pdists = pcd_distance(
                                crop_human_part(hname, [f"left {hpart}", f"right {hpart}"]), crop_object_part_3d(oname, opart), reduction=reduction
                            )
                            if pdists is None:
                                continue
                            pcano = self.scenes[scene_id].to_canonical(crop_human_part(hname, [f"left {hpart}", f"right {hpart}"]))
                        else:

                            pdists_left = pcd_distance(crop_human_part(hname, f"left {hpart}"), crop_object_part_3d(oname, opart), reduction=reduction)
                            pdists_right = pcd_distance(crop_human_part(hname, f"right {hpart}"), crop_object_part_3d(oname, opart), reduction=reduction)
                            if pdists_left is None or pdists_right is None:
                                continue
                            if iedge.continuous:
                                sel_left = pdists_left.mean().item() < pdists_right.mean().item()
                            else:
                                sel_left = pdists_left.min().item() < pdists_right.min().item()
                            if sel_left:
                                pdists = pdists_left
                                pcano = self.scenes[scene_id].to_canonical(crop_human_part(hname, f"left {hpart}"))
                            else:
                                pdists = pdists_right
                                pcano = self.scenes[scene_id].to_canonical(crop_human_part(hname, f"right {hpart}"))
                else:
                    visited.add((inodes[0].obj.name, inodes[0].name, inodes[1].obj.name, inodes[1].name))
                    visited.add((inodes[1].obj.name, inodes[1].name, inodes[0].obj.name, inodes[0].name))

                    part_pcds = [
                        crop_object_part_3d(inodes[0].obj.name, inodes[0].name),
                        crop_object_part_3d(inodes[1].obj.name, inodes[1].name),
                    ]
                    if part_pcds[0] is None or part_pcds[1] is None:
                        continue
                    part_diags = [torch.linalg.norm(ppcd[0, :, :].max(dim=0)[0] - ppcd[0, :, :].min(dim=0)[0]).item() for ppcd in part_pcds]
                    if part_diags[0] < part_diags[1]:
                        pdists = pcd_distance(part_pcds[0], part_pcds[1], reduction=reduction)
                    else:
                        pdists = pcd_distance(part_pcds[1], part_pcds[0], reduction=reduction)
                    pcano = self.scenes[scene_id].to_canonical(part_pcds[1])

                if iedge.continuous:
                    loss_nocontact.append(pdists.mean())
                else:
                    loss_nocontact.append(pdists.min())
                if iedge.rel_static:
                    loss_contact_drift.append(simple_static_loss(pcano.permute(1, 0, 2).contiguous()))
                else:
                    loss_contact_drift.append(simple_smoothness_loss(pcano.permute(1, 0, 2).contiguous()))

            if use_nocontact and len(loss_nocontact) != 0:
                loss_nocontact = torch.stack(loss_nocontact, dim=0).mean()
                loss_terms.append([lw("nocontact"), loss_nocontact, "nocontact"])
            if use_contact_drift and len(loss_contact_drift) != 0:
                loss_contact_drift = torch.stack(loss_contact_drift, dim=0).mean()
                loss_terms.append([lw("contact_drift"), loss_contact_drift, "contact_drift"])

        return loss_terms
