import os
import os.path as osp
import sys
import numpy as np
import pickle
import ast
import gc
import open3d as o3d
import cv2
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import smplx
from PIL import Image
from tqdm import tqdm

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.common import get_device, to_numpy, to_torch, list_files, may_create_folder, strip_str, to_o3d_mesh, save_o3d_pcd
from hms.config import VideoProcessorConfig, parse_config, THIRD_PARTY_DIR

def parse_det_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1 :])
            json_output = json_output.split("```")[0]
            break
    try:
        json_output = ast.literal_eval(json_output)
    except Exception as e:
        end_idx = json_output.rfind('"}') + len('"}')
        truncated_text = json_output[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)
    return json_output

def show_mask(mask, ax, random_color=False, borders=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 0 / 255, 110 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))

class VideoProcessor(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(VideoProcessorConfig, cfg)
        self.device = get_device()

        self.configure(*args, **kwargs)

    def configure(self, mllm, *args, **kwargs):
        self.mllm = mllm

    def get_paths(self, video_path):
        if video_path.endswith(".mp4"):
            video_dir = video_path[:-4]
            if not osp.exists(video_dir):
                may_create_folder(video_dir)
                os.system(f"/bin/ffmpeg -hide_banner -loglevel error -i {video_path} -q:v 2 -start_number 0 {video_dir}/'%05d.jpg'")
        else:
            video_dir = video_path
            video_path = video_path + ".mp4"
            if not osp.exists(video_path):
                os.system(f"/bin/ffmpeg -hide_banner -loglevel error -y -framerate 8 -pattern_type glob -i {video_dir}/*.jpg {video_path}")
        out_dir = video_dir + "_processed"
        may_create_folder(out_dir)
        return video_dir, video_path, out_dir

    def open_detect(self, image_path, text_prompt):
        cfg = self.cfg

        det_res = self.mllm.chat_images(
            provider=cfg.det_provider,
            model_name=cfg.det_model,
            system_prompt=cfg.det_system_prompt,
            user_prompt=cfg.det_user_prompt.format(text_prompt),
            images=[image_path],
            task_name=cfg.det_task,
            temperature=cfg.det_temperature,
        )

        det_response = det_res["response"].strip()

        ow, oh = det_res["original_width"], det_res["original_height"]
        nw, nh = det_res["new_width"], det_res["new_height"]
        if not det_response.startswith("```json"):
            return list()
        json_output = parse_det_json(det_response)

        res = list()
        for idx, bounding_box in enumerate(json_output):
            abs_x1 = int(bounding_box["bbox_2d"][0] / nw * ow)
            abs_y1 = int(bounding_box["bbox_2d"][1] / nh * oh)
            abs_x2 = int(bounding_box["bbox_2d"][2] / nw * ow)
            abs_y2 = int(bounding_box["bbox_2d"][3] / nh * oh)
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            res.append((abs_x1, abs_y1, abs_x2, abs_y2))

        return res

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def segment_objects(self, video_path, bboxes, frame_id):
        from sam2.build_sam import build_sam2_video_predictor

        video_dir, video_path, out_dir = self.get_paths(video_path)
        frame_names = list_files(video_dir, "*.jpg", alphanum_sort=True)

        sam2_predictor = build_sam2_video_predictor(self.cfg.segmenter_model_cfg, self.cfg.segmenter_model_path, device=self.device)

        res = list()
        for bid in range(len(bboxes)):
            inference_state = sam2_predictor.init_state(video_path=video_dir)
            sam2_predictor.reset_state(inference_state)

            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_id,
                obj_id=0,
                box=np.asarray(bboxes[bid]),
            )

            video_segments = dict()
            propagate_res = sam2_predictor.propagate_in_video(inference_state)
            for out_frame_idx, out_obj_ids, out_mask_logits in propagate_res:
                video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}

            frame_masks = list()
            for out_frame_idx in range(0, len(frame_names)):
                frame = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
                frame = np.array(frame.convert("RGB")).astype(np.float32) / 255.0
                if out_frame_idx in video_segments and 0 in video_segments[out_frame_idx]:
                    mask = np.squeeze(video_segments[out_frame_idx][0], axis=0)
                else:

                    mask = np.zeros_like(frame[..., 0]) > 0
                frame_masks.append(mask)
            frame_masks = np.stack(frame_masks, axis=0)
            res.append(frame_masks)
        res = np.stack(res, axis=0)

        del sam2_predictor
        torch.cuda.empty_cache()
        gc.collect()
        return res

    def detect_and_segment_objects(self, video_path, object_name, object_prompt, seg_all=False):
        video_dir, video_path, out_dir = self.get_paths(video_path)
        frame_names = list_files(video_dir, "*.jpg", alphanum_sort=True)

        for frame_name in frame_names:
            frame_id = int(frame_name[:-4])
            box_coords = self.open_detect(image_path=osp.join(video_dir, frame_name), text_prompt=object_prompt)
            if len(box_coords) > 0:
                break

        if not seg_all:
            box_coords = box_coords[:1]

        if len(box_coords) == 0:
            masks = None
        else:
            masks = self.segment_objects(video_path=video_path, bboxes=box_coords, frame_id=frame_id)

        detseg = {
            "name": object_name,
            "prompt": object_prompt,
            "frame_id": frame_id,
            "bboxes": box_coords,
            "masks": masks,
            "video_path": video_path,
            "video_dir": video_dir,
            "frame_names": frame_names,
        }

        for fid, fname in enumerate(frame_names):
            fimg = Image.open(osp.join(video_dir, fname)).convert("RGB")
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.imshow(fimg)
            for mask in masks[:, fid, :, :]:
                show_mask(mask, ax)
            if fid == 0:
                for box in box_coords:
                    show_box(box, ax)
            fig.tight_layout()
            fig.savefig(
                osp.join(out_dir, f"det_segment_{strip_str(object_name)}_{fname[:-4]}.png"),
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                dpi=128,
            )
            plt.close(fig)

        return detseg

    def detect_and_segment_object_parts(self, video_path, object_masks, object_name, object_prompt, part_name, part_prompt):
        video_dir, video_path, out_dir = self.get_paths(video_path)
        frame_names = list_files(video_dir, "*.jpg", alphanum_sort=True)

        assert object_masks.shape[0] == len(frame_names)

        for frame_name in frame_names:
            frame_id = int(frame_name[:-4])
            obj_mask = object_masks[frame_id]

            frame_path = osp.join(video_dir, frame_name)
            box_coords = self.open_detect(image_path=frame_path, text_prompt=f"the {part_prompt} part of {object_prompt}")
            if len(box_coords) == 0:
                continue

            box_masks = np.zeros((len(box_coords), *obj_mask.shape))
            for bid, bc in enumerate(box_coords):
                xmin, ymin, xmax, ymax = bc
                box_masks[bid, ymin:ymax, xmin:xmax] = 1
            box_masks = np.logical_and(box_masks > 0, np.expand_dims(obj_mask, axis=0))
            box_overlaps = np.any(box_masks, axis=(1, 2))
            box_coords = [bc for bid, bc in enumerate(box_coords) if box_overlaps[bid].item()]
            if len(box_coords) == 0:
                continue
            break

        if len(box_coords) == 0:
            masks = np.zeros_like(object_masks)
        else:

            masks = self.segment_objects(video_path=video_path, bboxes=box_coords, frame_id=frame_id)

            masks = masks.any(axis=0)
        masks = np.logical_and(masks, object_masks)

        detseg = {
            "object_name": object_name,
            "object_prompt": object_prompt,
            "part_name": part_name,
            "part_prompt": part_prompt,
            "frame_id": frame_id,
            "bboxes": box_coords,
            "masks": masks,
            "video_path": video_path,
            "video_dir": video_dir,
            "frame_names": frame_names,
        }

        for fid, fname in enumerate(frame_names):
            fimg = Image.open(osp.join(video_dir, fname)).convert("RGB")
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.imshow(fimg)
            show_mask(masks[fid], ax)
            if fid == 0:
                for box in box_coords:
                    show_box(box, ax)
            fig.tight_layout()
            fig.savefig(
                osp.join(out_dir, f"det_segment_{strip_str(object_name)}_{strip_str(part_name)}_{fname[:-4]}.png"),
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                dpi=128,
            )
            plt.close(fig)

        return detseg

    def infer_pmap(self, video_path, interval=1):
        video_dir, video_path, out_dir = self.get_paths(video_path)

        LIB_DIRS = [f"{THIRD_PARTY_DIR}/MoGe"]
        for LIB_DIR in LIB_DIRS:
            if LIB_DIR not in sys.path:
                sys.path.append(LIB_DIR)

        from moge.model import MoGeModel
        import utils3d

        remove_edge = True

        model = MoGeModel.from_pretrained(self.cfg.pmap_model_path).eval()
        model = model.to(self.device)

        frame_names = list_files(video_dir, "*.jpg", alphanum_sort=True)
        res = {
            "points": list(),

            "mask": list(),
            "intrinsics": list(),
        }
        for fidx, fname in tqdm(enumerate(frame_names), "Infer pmap"):

            image = Image.open(osp.join(video_dir, fname)).convert("RGB")
            image = np.array(image) / 255.0
            image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)
            image_tensor = image_tensor.permute(2, 0, 1)

            output = model.infer(image_tensor, resolution_level=9, apply_mask=True)

            points = to_numpy(output["points"])
            depth = to_numpy(output["depth"])
            mask = to_numpy(output["mask"])
            intrinsics = to_numpy(output["intrinsics"])
            if remove_edge:
                mask = mask & ~utils3d.numpy.depth_edge(depth, mask=mask, rtol=0.02)
            mask = mask & (depth > 0)

            if fidx % interval == 0 or fidx == len(frame_names) - 1:
                save_o3d_pcd(osp.join(out_dir, f"pmap_{fname[:-4]}.ply"), V=points[mask, :], VC=image[mask, :])

            res["points"].append(points)

            res["mask"].append(mask)
            res["intrinsics"].append(intrinsics)

        for k, v in res.items():
            res[k] = np.stack(v, axis=0)

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return res

    def fit_human_poses_gvhmr(self, video_path, human_masks):
        video_dir, video_path, out_dir = self.get_paths(video_path)

        num_humans, n_frames, frame_height, frame_width = human_masks.shape

        human_boxes = list()
        for hid in range(num_humans):
            hboxes = list()
            for fid in range(n_frames):
                hmask = human_masks[hid, fid]
                bmask = np.zeros_like(hmask)
                if np.sum(hmask).item() != 0:
                    hcoords = np.argwhere(hmask)
                    h_ymin, h_xmin = hcoords.min(axis=0)
                    h_ymax, h_xmax = hcoords.max(axis=0) + 1
                    bmask[h_ymin:h_ymax, h_xmin:h_xmax] = True
                hboxes.append(bmask)
            human_boxes.append(hboxes)
        human_boxes = np.array(human_boxes)

        def assign_by_overlap(vitpose, assigned_ids):
            assigned_ids = set(assigned_ids)
            ores = list()
            for hid in range(num_humans):
                if hid in assigned_ids:
                    continue
                ocount = 0
                for fid in range(vitpose.shape[0]):
                    kpts = vitpose[fid, vitpose[fid, :, 2] > 0, :2].astype(np.int32)
                    kpts_flag = (kpts[:, 0] >= 0) & (kpts[:, 0] < frame_width) & (kpts[:, 1] >= 0) & (kpts[:, 1] < frame_height)
                    kpts = kpts[kpts_flag, :]
                    if kpts.shape[0] == 0:
                        continue
                    ocount += np.sum(human_boxes[hid, fid][kpts[:, 1], kpts[:, 0]]).item()
                ores.append((hid, ocount))
            ores = sorted(ores, key=lambda x: x[1], reverse=True)
            return ores[0][0]

        smplx_model = smplx.create(
            model_path=self.cfg.smplx_path,
            model_type=self.cfg.smplx_type,
            batch_size=n_frames,
            gender=self.cfg.gender,
        )
        smplx_model.to(self.device)
        smplx_model.requires_grad_(False)

        res = list()
        res_indices = list()
        for hid in range(num_humans):
            rpath = osp.join(out_dir, f"gvhmr_output_{hid}.pkl")
            vpath = osp.join(out_dir, f"gvhmr_vitpose_{hid}.pt")
            if osp.exists(rpath):
                with open(rpath, "rb") as fh:
                    hdict = pickle.load(fh)
            else:
                os.system(f"python {THIRD_PARTY_DIR}/GVHMR/gvhmr_api.py --video {video_path} --human_id {hid} --output_pth {out_dir}")
                with open(rpath, "rb") as fh:
                    hdict = pickle.load(fh)

            frame_ids = hdict["frame_ids"]
            hdict["2d_poses"] = to_numpy(torch.load(vpath))[frame_ids]

            hdict_filled = dict()
            for tpk, tpv in hdict.items():
                if tpk == "frame_ids":
                    continue
                hdict_filled[tpk] = np.zeros((n_frames, *tpv.shape[1:]), dtype=tpv.dtype)
                hdict_filled[tpk][frame_ids] = tpv
            for fid in range(n_frames):
                if fid not in frame_ids:

                    nearest_idx = frame_ids[np.argmin(np.abs(frame_ids - fid))]
                    for tpk, tpv in hdict_filled.items():
                        hdict_filled[tpk][fid] = tpv[nearest_idx]
            hdict = hdict_filled
            res.append(hdict)

            aid = assign_by_overlap(hdict["2d_poses"], res_indices)
            res_indices.append(aid)

            smplxdict = smplx_model(
                transl=to_torch(hdict["transl"]).to(self.device),
                global_orient=to_torch(hdict["global_orient"]).to(self.device),
                betas=to_torch(hdict["betas"]).to(self.device),
                body_pose=to_torch(hdict["body_pose"]).to(self.device),
            )

            vertices = to_numpy(smplxdict.vertices)
            faces = to_numpy(smplx_model.faces_tensor.int())
            for fid in range(n_frames):
                mesh = to_o3d_mesh(vertices[fid], faces)
                o3d.io.write_triangle_mesh(osp.join(out_dir, f"gvhmr_{hid}_{fid:04d}.ply"), mesh)

        if len(res) == num_humans:
            with open(osp.join(out_dir, "gvhmr_human_indices_map.json"), "w") as fh:
                json.dump({"deg_segment_indices": res_indices}, fh, indent=4)
            res = [res[idx] for idx in res_indices]
            return res
        else:
            return list()

    def get_masks(self, object_names, object_descs, num_frames, frame_height, frame_width, video_path):

        object_names_nonum = list()
        for object_name in object_names:
            osuffix = object_name.split(" ")[-1]
            if osuffix.isdigit():
                object_names_nonum.append(object_name[: -len(osuffix)].strip())
            else:
                object_names_nonum.append(object_name)

        visited = [False] * len(object_names)

        def get_duplicates(oname_nonum, odesc):
            duplicates = list()
            for oid in range(len(object_names)):
                if not visited[oid]:
                    if object_names_nonum[oid] == oname_nonum and object_descs[oid] == odesc:
                        duplicates.append(oid)
            return duplicates

        object_masks = np.zeros((len(object_names), num_frames, frame_height, frame_width)) > 0

        for oid in range(len(object_names)):
            if visited[oid]:
                continue
            visited[oid] = True
            odindices = get_duplicates(object_names_nonum[oid], object_descs[oid])
            if len(odindices) > 0:

                omask = self.detect_and_segment_objects(
                    video_path=video_path,
                    object_name=object_names[oid],
                    object_prompt=object_descs[oid],
                    seg_all=True,
                )["masks"]
                odindices = [oid] + odindices
                mincnt = min(len(odindices), omask.shape[0])
                object_masks[odindices[:mincnt]] = omask[:mincnt]
                for odid in odindices[1:mincnt]:
                    visited[odid] = True
            else:

                omask = self.detect_and_segment_objects(
                    video_path=video_path,
                    object_name=object_names[oid],
                    object_prompt=object_descs[oid],
                    seg_all=False,
                )["masks"]
                object_masks[oid] = np.squeeze(omask, axis=0)

        return object_masks

    def process(self, video_path, igraph, estimate_3d=True, **kwargs):
        video_dir, video_path, out_dir = self.get_paths(video_path)

        res_path = osp.join(out_dir, "video_processed.pkl")
        if osp.exists(res_path):
            with open(res_path, "rb") as fh:
                return pickle.load(fh)

        frame_names = list_files(video_dir, "*.jpg", alphanum_sort=True, full_path=True)
        n_frames = len(frame_names)
        frame_width, frame_height = Image.open(frame_names[0]).convert("RGB").size

        human_names = igraph.get_humans()
        human_descs = [igraph.get_human(hname).description for hname in human_names]
        human_masks = self.get_masks(human_names, human_descs, n_frames, frame_height, frame_width, video_path)

        object_names = igraph.get_objects()
        object_descs = [igraph.get_object(oname).description for oname in object_names]
        object_masks = self.get_masks(object_names, object_descs, n_frames, frame_height, frame_width, video_path)

        has_humans = np.any(human_masks, axis=(0, 2, 3))
        has_objects = np.any(object_masks, axis=(0, 2, 3))

        if not has_humans.any().item() or not has_objects.any().item():
            return None

        object_part_masks = list()
        for oid in range(len(object_names)):
            oname = object_names[oid]
            oname_suffix = oname.split(" ")[-1]
            if oname_suffix.isdigit():
                oname_nonum = oname[: -len(oname_suffix)].strip()
            else:
                oname_nonum = oname
            omask = object_masks[oid]

            opmasks = dict()
            for pname in igraph.get_object_parts(oname):
                pdetseg = self.detect_and_segment_object_parts(
                    video_path=video_path,
                    object_masks=omask,
                    object_name=oname,
                    object_prompt=oname_nonum,
                    part_name=pname,
                    part_prompt=pname,
                )
                opmasks[pname] = pdetseg["masks"]
            object_part_masks.append(opmasks)

        res = {
            "human_masks": human_masks,
            "object_masks": object_masks,
            "object_names": object_names,
            "object_part_masks": object_part_masks,
        }

        if estimate_3d:

            pmap_res = self.infer_pmap(video_path, interval=10)
            pmap_points = pmap_res["points"]
            pmap_masks = pmap_res["mask"]
            pmap_intrinsics = pmap_res["intrinsics"]

            for i in range(n_frames):
                height, width = pmap_points[i].shape[:2]
                pmap_intrinsics[i, 0, 0] *= width
                pmap_intrinsics[i, 0, 2] *= width
                pmap_intrinsics[i, 1, 1] *= height
                pmap_intrinsics[i, 1, 2] *= height

            erode_kernel = np.ones((self.cfg.erode_kernel, self.cfg.erode_kernel), np.uint8)
            erode_iters = self.cfg.erode_iters

            human_points = list()
            for hid in range(len(human_masks)):
                hmasks = np.logical_and(human_masks[hid], pmap_masks)
                hpoints = list()
                for i in range(n_frames):
                    hmask = cv2.erode(hmasks[i].astype(np.uint8) * 255, erode_kernel, iterations=erode_iters)
                    hmask = hmask > 127
                    hpoints.append(pmap_points[i][hmask, :])
                human_points.append(hpoints)

            human_centers = list()
            human_bboxes = list()
            human_findices = list()
            for i in range(n_frames):
                hpoints = [human_points[hid][i] for hid in range(len(human_points))]
                hpoints = np.concatenate(hpoints, axis=0)
                if hpoints.shape[0] > 0:
                    hcenter = np.mean(hpoints, axis=0)

                    hbbox = np.concatenate([np.amin(hpoints, axis=0), np.amax(hpoints, axis=0)])
                    human_findices.append(i)
                else:
                    hcenter = None
                    hbbox = None
                human_centers.append(hcenter)
                human_bboxes.append(hbbox)
            human_findices = np.array(human_findices)

            for i in range(n_frames):
                if human_centers[i] is None:

                    nearest_idx = human_findices[np.argmin(np.abs(human_findices - i))]
                    human_centers[i] = human_centers[nearest_idx]
                    human_bboxes[i] = human_bboxes[nearest_idx]
            human_centers = np.array(human_centers)
            human_bboxes = np.array(human_bboxes)

            zranges = list()
            zoffset_ratio = 0.8
            for i in range(n_frames):
                hbbox_size = human_bboxes[i][3:] - human_bboxes[i][:3]
                maxside = np.amax(hbbox_size[:2])
                zrange = human_centers[i][2] + np.array([-maxside / 2, maxside / 2]) * zoffset_ratio
                zranges.append(zrange)
            zranges = np.array(zranges)

            object_points = list()
            for oid in range(len(object_masks)):
                omasks = np.logical_and(object_masks[oid], pmap_masks)
                opoints = list()
                for i in range(n_frames):
                    omask = cv2.erode(omasks[i].astype(np.uint8) * 255, erode_kernel, iterations=erode_iters)
                    omask = omask > 127
                    opcd = pmap_points[i][omask, :]
                    zmask = np.logical_and(opcd[:, 2] >= zranges[i, 0], opcd[:, 2] <= zranges[i, 1])
                    opcd = opcd[zmask, :]
                    opoints.append(opcd)
                object_points.append(opoints)

            object_part_points = list()
            for oid in range(len(object_part_masks)):
                oppoints = dict()
                for pname, pmasks in object_part_masks[oid].items():
                    pmasks = np.logical_and(pmasks, pmap_masks)
                    ppoints = list()
                    for i in range(n_frames):
                        pmask = cv2.erode(pmasks[i].astype(np.uint8) * 255, erode_kernel, iterations=erode_iters)
                        pmask = pmask > 127
                        if np.sum(pmask) < 30:
                            pmask = pmasks[i]
                        ppcd = pmap_points[i][pmask, :]
                        zmask = np.logical_and(ppcd[:, 2] >= zranges[i, 0], ppcd[:, 2] <= zranges[i, 1])
                        ppcd = ppcd[zmask, :]
                        ppoints.append(ppcd)
                    oppoints[pname] = ppoints
                object_part_points.append(oppoints)

            human_poses = self.fit_human_poses_gvhmr(video_path, human_masks)
            if len(human_poses) == 0:
                return None

            res["intrinsics"] = pmap_intrinsics
            res["human_points"] = human_points
            res["object_points"] = object_points
            res["object_part_points"] = object_part_points
            res["human_poses"] = human_poses

        with open(res_path, "wb") as fh:
            pickle.dump(res, fh)

        return res
