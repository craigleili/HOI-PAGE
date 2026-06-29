import os.path as osp
import sys
import numpy as np
import pickle
import json
import ast
import gc
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.scene import MeshScenes
from hms.view_data import ViewData
from hms.render import NVRenderer
from hms.common import get_device, to_numpy, to_cuda, write_json, list_files, strip_str, may_create_folder, valid_str, save_o3d_pcd, load_o3d_pcd, KNNSearch
from hms.config import AffordanceConfig, parse_config

def parse_json(json_str):
    left_pos = json_str.find("{")
    right_pos = json_str.rfind("}")
    if left_pos < 0 or right_pos < 0:
        return dict()
    json_str = json_str[left_pos : right_pos + 1]
    json_output = json.loads(json_str)
    return json_output

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

@dataclass
class iGraphObject:
    dtype: str
    name: str
    translational: bool
    rotational: bool
    description: str

@dataclass
class iGraphNode:
    obj: iGraphObject
    name: str

@dataclass
class iGraphEdge:
    nodes: tuple
    rel_static: bool
    continuous: bool

class iGraph:
    def __init__(self, afford_dict):
        self.objs = list()
        for otype in ["object", "human"]:
            for obj_state in afford_dict[f"{otype} states"]:
                self.objs.append(
                    iGraphObject(
                        dtype=otype,
                        name=obj_state["name"],
                        translational=obj_state.get("is_translational", True),
                        rotational=obj_state.get("is_rotational", True),
                        description=obj_state["description"],
                    )
                )

        self.nodes = list()
        for ntype in ["object part nodes", "body part nodes"]:
            for ntuple in afford_dict[ntype]:
                obj_name, part_name = ntuple.split(",")
                obj_name = obj_name.strip()
                part_name = part_name.strip()
                obj = self._find_obj(obj_name)
                if obj is not None:
                    self.nodes.append(iGraphNode(obj=obj, name=part_name))

        self.edges = list()
        for edict in afford_dict["interaction edges"]:
            obj_name1_part_name1 = edict["nodes"][0]
            obj_name2_part_name2 = edict["nodes"][1]

            obj_name1, part_name1 = obj_name1_part_name1.split(",")
            obj_name2, part_name2 = obj_name2_part_name2.split(",")
            obj_name1 = obj_name1.strip()
            part_name1 = part_name1.strip()
            obj_name2 = obj_name2.strip()
            part_name2 = part_name2.strip()

            n1 = self._find_node(obj_name1, part_name1)
            n2 = self._find_node(obj_name2, part_name2)

            self.edges.append(
                iGraphEdge(
                    nodes=(n1, n2),
                    rel_static=edict["is_rel_static"],
                    continuous=edict["is_continuous"],
                )
            )

    def _find_obj(self, obj_name):
        for o in self.objs:
            if o.name == obj_name:
                return o
        return None

    def _find_node(self, obj_name, part_name):
        for n in self.nodes:
            if n.obj.name == obj_name and n.name == part_name:
                return n
        return None

    def get_object(self, obj_name):
        for o in self.objs:
            if o.dtype == "object" and o.name == obj_name:
                return o
        return None

    def get_objects(self):

        return [o.name for o in self.objs if o.dtype == "object"]

    def get_object_parts(self, obj_name):
        return sorted([n.name for n in self.nodes if n.obj.dtype == "object" and n.obj.name == obj_name])

    def get_human(self, human_name):
        for o in self.objs:
            if o.dtype == "human" and o.name == human_name:
                return o
        return None

    def get_humans(self):
        return sorted([o.name for o in self.objs if o.dtype == "human"])

    def get_human_parts(self):
        return sorted(list(set([n.name for n in self.nodes if n.obj.dtype == "human" and n.obj.dtype == "human"])))

    def has_interaction(self, obj_name1, part_name1, obj_name2, part_name2):
        for e in self.edges:
            n1, n2 = e.nodes
            if n1.obj.name == obj_name1 and n1.name == part_name1 and n2.obj.name == obj_name2 and n2.name == part_name2:
                return e
            elif n1.obj.name == obj_name2 and n1.name == part_name2 and n2.obj.name == obj_name1 and n2.name == part_name1:
                return e
        return None

class Affordance(object):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(AffordanceConfig, cfg)
        self.device = get_device()

        self.configure(*args, **kwargs)

    def configure(self, mllm, glctx, renderer_cfg, scene_cfgs, interaction_prompt, *args, **kwargs):
        self.mllm = mllm
        self.glctx = glctx
        self.renderer_cfg = renderer_cfg
        if not isinstance(scene_cfgs, (list, tuple)):
            scene_cfgs = [scene_cfgs]
        self.scene_cfgs = scene_cfgs
        self.interaction_prompt = interaction_prompt

    def render_object(self, scene_cfg, out_dir):
        may_create_folder(out_dir)

        scene_cfg = deepcopy(scene_cfg)
        scene_cfg["n_frames"] = self.cfg.n_frames

        scenes = MeshScenes(scene_cfg)
        scene_pcd = scenes()["pcds"][0]
        assert scene_pcd.ndim == 2
        scene_center = torch.mean(scene_pcd, dim=0)
        scene_radius = torch.linalg.norm(scene_pcd - scene_center.unsqueeze(0), dim=1).max().item()
        camera_distance = scene_radius * self.cfg.camera_distance_factor

        view_data_cfg = {
            "n_elevations": self.cfg.n_elevations,
            "n_azimuths": self.cfg.n_azimuths,
            "elevation_range": self.cfg.elevation_range,
            "azimuth_range": self.cfg.azimuth_range,
            "camera_distance": camera_distance,
        }
        view_data = ViewData(view_data_cfg)

        renderer_cfg = deepcopy(self.renderer_cfg)
        renderer = NVRenderer(renderer_cfg, glctx=self.glctx, objects_to_render=[scenes])

        view_data_loader = DataLoader(
            view_data,
            num_workers=0,
            batch_size=view_data.cfg.batch_size,
            collate_fn=view_data.collate,
            shuffle=False,
        )

        images = dict()
        pmaps_world = dict()
        pmaps_cam = dict()
        for vid, vdata in enumerate(view_data_loader):
            vname = f"view_{vid:03d}"
            vdata = to_cuda(vdata, self.device)
            rdict = renderer(**vdata, center=scene_center)
            rgb = to_numpy(rdict["comp_rgb_nobg"].squeeze(0))
            mask = to_numpy(rdict["comp_mask"].squeeze(0))
            wpmap = to_numpy(rdict["comp_world_coords"].squeeze(0))
            cpmap = to_numpy(rdict["comp_cam_coords"].squeeze(0))
            assert rgb.ndim == mask.ndim == cpmap.ndim == wpmap.ndim == 3
            rgba = np.concatenate((rgb, mask), axis=-1)
            rgba = (rgba * 255).astype(np.uint8)

            images[vname] = rgba
            pmaps_world[vname] = wpmap
            pmaps_cam[vname] = cpmap

            Image.fromarray(rgba).save(osp.join(out_dir, f"{vname}.png"))

            mask = mask.flatten() > 0
            save_o3d_pcd(osp.join(out_dir, f"{vname}_cam.ply"), V=np.reshape(cpmap, (-1, 3))[mask, :])
            save_o3d_pcd(osp.join(out_dir, f"{vname}_world.ply"), V=np.reshape(wpmap, (-1, 3))[mask, :])

        return out_dir, images, pmaps_world, pmaps_cam

    def open_detect(self, image_path, object_name, object_part, human_part):
        cfg = self.cfg
        if valid_str(human_part):
            det_user_prompt = cfg.det_user_prompt.format(f"the {object_part} part of {object_name} that can be contacted by human {human_part}")
        else:
            det_user_prompt = cfg.det_user_prompt.format(f"the {object_part} part of {object_name}")

        det_res = self.mllm.chat_images(
            provider=cfg.det_provider,
            model_name=cfg.det_model,
            system_prompt=cfg.det_system_prompt,
            user_prompt=det_user_prompt,
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
    def segment_part(self, object_name, igraph, image_paths, out_dir):
        res_path = osp.join(out_dir, "part_segment2d.pkl")
        if osp.exists(res_path):
            with open(res_path, "rb") as fh:
                return pickle.load(fh)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(self.cfg.segmenter_model_cfg, self.cfg.segmenter_model_path, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        may_create_folder(out_dir)

        res = dict()
        for object_part in igraph.get_object_parts(object_name):
            masks = dict()
            for image_path in image_paths:
                image_name = Path(image_path).stem

                det_bboxes = self.open_detect(image_path=image_path, object_name=object_name, object_part=object_part, human_part=None)

                if len(det_bboxes) == 0:
                    continue

                img = Image.open(image_path).convert("RGBA")
                img_mask = np.array(img)[:, :, 3] > 0
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img).convert("RGB")

                sam2_predictor.set_image(img)
                seg_masks, _, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(det_bboxes),
                    multimask_output=False,
                )
                if seg_masks.ndim == 3:
                    seg_masks = np.expand_dims(seg_masks, axis=0)

                seg_mask = np.any(seg_masks > 0, axis=(0, 1))
                if np.sum(img_mask & seg_mask) / np.sum(img_mask) > 0.9:
                    continue
                masks[image_name] = seg_mask

                fig, ax = plt.subplots()
                ax.axis("off")
                ax.imshow(img)
                for seg_mask in seg_masks:
                    show_mask(seg_mask.squeeze(0), ax)
                for det_box in det_bboxes:
                    show_box(det_box, ax)
                fig.tight_layout()
                fig.savefig(
                    osp.join(out_dir, f"part_detseg_{strip_str(object_part)}_{image_name}.png"),
                    bbox_inches="tight",
                    pad_inches=0,
                    transparent=True,
                    dpi=128,
                )
                plt.close(fig)

            if object_part in res:
                for image_name, seg_mask in masks.items():
                    if image_name in res[object_part]:
                        res[object_part][image_name] = np.logical_or(res[object_part][image_name], seg_mask)
                    else:
                        res[object_part][image_name] = seg_mask
            else:
                res[object_part] = masks

        with open(res_path, "wb") as fh:
            pickle.dump(res, fh)

        del sam2_predictor
        del sam2_model
        torch.cuda.empty_cache()
        gc.collect()

        return res

    def segment_object(self, scene_cfg, igraph, out_dir):
        res_path = osp.join(out_dir, "part_segment3d.pkl")
        if osp.exists(res_path):
            with open(res_path, "rb") as fh:
                return pickle.load(fh)

        may_create_folder(out_dir)
        object_name = scene_cfg["obj_name"].strip()
        object_parts = igraph.get_object_parts(object_name)

        render_dir, render_images, render_pmaps_world, _ = self.render_object(scene_cfg, out_dir)
        view_image_paths = list_files(render_dir, name_filter="view_*.png", alphanum_sort=True, full_path=True)

        segres = self.segment_part(object_name=object_name, igraph=igraph, image_paths=view_image_paths, out_dir=out_dir)

        object_pcd_path = scene_cfg["obj_path"][:-4] + "_pcd.ply"
        object_pcd = load_o3d_pcd(object_pcd_path)["V"]

        knn_search = KNNSearch(object_pcd)

        _, nndists = knn_search.query(object_pcd, k=2, return_dists=True)
        nn_thresh = min(np.amax(nndists[:, 1]).item() * 1.5, 0.07)

        n_parts = len(object_parts)
        label_counts = np.zeros((object_pcd.shape[0], n_parts), dtype=np.int32)
        for pid, object_part in enumerate(object_parts):
            for image_name, seg_mask in segres[object_part].items():
                pmap = render_pmaps_world[image_name]
                pmask = render_images[image_name][..., -1] > 0
                seg_mask = seg_mask & pmask
                seg_pcd = pmap[seg_mask, :]
                nnindices, nndists = knn_search.query(seg_pcd, k=1, return_dists=True)
                nnindices = nnindices[nndists <= nn_thresh]
                label_counts[nnindices, pid] += 1

        nnindices, nndists = knn_search.query(object_pcd, k=16 + 1, return_dists=True)
        nnindices = nnindices[:, 1:]
        nndists = nndists[:, 1:]
        nnweights = 1.0 / (nndists + 1e-6)
        nnweights /= np.sum(nnweights, axis=1, keepdims=True)
        label_votes = label_counts[nnindices, :].astype(np.float32) * np.expand_dims(nnweights, axis=-1)
        label_votes = np.sum(label_votes, axis=1)

        no_labels = np.sum(label_votes, axis=1) == 0
        no_labels_indices = np.where(no_labels)[0]
        has_labels_indices = np.where(~no_labels)[0]

        part_labels = np.argmax(label_votes, axis=1)
        part_labels[no_labels_indices] = n_parts
        if no_labels_indices.shape[0] != 0 and has_labels_indices.shape[0] != 0:
            knn_search = KNNSearch(object_pcd[has_labels_indices])
            nnindices, nndists = knn_search.query(object_pcd[no_labels_indices], k=1, return_dists=True)
            part_labels[no_labels_indices] = part_labels[has_labels_indices[nnindices]]
            part_labels[no_labels_indices[nndists > nn_thresh]] = n_parts

        label_colors = np.random.default_rng(seed=3).integers(64, 255, (n_parts, 3))
        part_colors = np.zeros((object_pcd.shape[0], 3), dtype=np.uint8)
        for pid in range(n_parts):
            part_colors[part_labels == pid] = np.expand_dims(label_colors[pid], 0)
        save_o3d_pcd(osp.join(out_dir, "part_labels.ply"), V=object_pcd, VC=part_colors)

        write_json(
            osp.join(out_dir, "part_colors.json"),
            OrderedDict([(object_part, label_colors[pid].tolist()) for pid, object_part in enumerate(object_parts)]),
        )

        res = {
            "object_part_labels": object_parts,
            "object_part_colors": label_colors,
            "object_labeling": part_labels,
        }

        with open(res_path, "wb") as fh:
            pickle.dump(res, fh)

        return res

    def query_mllm(self, user_prompt):
        response = self.mllm.chat_text(
            provider=self.cfg.afford_provider,
            model_name=self.cfg.afford_model,
            system_prompt=self.cfg.afford_system_prompt,
            user_prompt=self.cfg.afford_user_prompt.format(user_prompt),
            task_name=self.cfg.afford_task,
            temperature=self.cfg.afford_temperature,
        )["response"]
        if not valid_str(response):
            raise ValueError("Invalid response from LLM")
        response_parsed = parse_json(response)
        if "interaction" not in response_parsed:
            raise ValueError("Invalid response from LLM")
        igraph = iGraph(response_parsed)
        video_prompt = response_parsed["interaction"]
        return igraph, video_prompt, response_parsed, response

    def process(self, out_dir):
        res_path = osp.join(out_dir, "affordance.pkl")
        if osp.exists(res_path):
            with open(res_path, "rb") as fh:
                return pickle.load(fh)

        obj_names = [scfg["obj_name"].strip() for scfg in self.scene_cfgs]

        llm_input = {"objects": obj_names, "interaction": self.interaction_prompt}

        with open(osp.join(out_dir, "affordance_input.json"), "w") as fh:
            json.dump(llm_input, fh, indent=4)

        max_tries = 3
        for attempt in range(max_tries):
            try:
                igraph, video_prompt, afford_res, afford_raw = self.query_mllm(json.dumps(llm_input))
                break
            except Exception as e:
                if attempt == max_tries - 1:
                    raise RuntimeError(f"LLM query failed after {max_tries} attempts")

        with open(osp.join(out_dir, "affordance_output.txt"), "w") as fh:
            fh.write(afford_raw)

        with open(osp.join(out_dir, "affordance_output.json"), "w") as fh:
            json.dump(afford_res, fh, indent=4)

        seg_res = dict()
        for scfg in self.scene_cfgs:
            object_name = scfg["obj_name"].strip()
            sres = self.segment_object(
                scene_cfg=scfg,
                igraph=igraph,
                out_dir=osp.join(out_dir, f"segment_{strip_str(object_name)}"),
            )
            seg_res[object_name] = sres

        res = {
            "interaction_graph": igraph,
            "video_prompt": video_prompt,
            "object_segmentations": seg_res,
        }

        with open(res_path, "wb") as fh:
            pickle.dump(res, fh)

        return res
