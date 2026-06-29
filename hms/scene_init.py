import os.path as osp
import sys
import numpy as np
import json
import pickle
import torch
from collections import OrderedDict

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.common import (
    get_device,
    to_numpy,
    valid_str,
    get_translate_matrix,
    get_rotation_matrix,
    apply_transform3d,
    apply_transforms3d,
    save_o3d_pcd,
    save_o3d_mesh,
)
from hms.config import SceneInitConfig, parse_config

def parse_json(json_str):
    left_pos = json_str.find("{")
    right_pos = json_str.rfind("}")
    if left_pos < 0 or right_pos < 0:
        return dict()
    json_str = json_str[left_pos : right_pos + 1]
    json_output = json.loads(json_str)
    return json_output

class SceneInit(object):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(SceneInitConfig, cfg)
        self.device = get_device()

        self.configure(*args, **kwargs)

    def configure(self, mllm, interaction_prompt, *args, **kwargs):
        self.mllm = mllm
        self.interaction_prompt = interaction_prompt

        joint_indices = OrderedDict()
        joint_indices["nose"] = 15
        joint_indices["neck"] = 12
        joint_indices["left shoulder"] = 16
        joint_indices["right shoulder"] = 17
        joint_indices["left arm"] = 18
        joint_indices["right arm"] = 19
        joint_indices["left hand"] = 20
        joint_indices["right hand"] = 21
        joint_indices["hips"] = 0
        joint_indices["left leg"] = 4
        joint_indices["right leg"] = 5
        joint_indices["left foot"] = 10
        joint_indices["right foot"] = 11
        self.joint_indices = joint_indices

    def query_mllm(self, user_prompt):
        response = self.mllm.chat_text(
            provider=self.cfg.boxpred_provider,
            model_name=self.cfg.boxpred_model,
            system_prompt=self.cfg.boxpred_system_prompt,
            user_prompt=self.cfg.boxpred_user_prompt.format(user_prompt),
            task_name=self.cfg.boxpred_task,

        )["response"]
        if not valid_str(response):
            raise ValueError("Invalid response from LLM")
        response_parsed = parse_json(response)
        if "rotations" not in response_parsed:
            raise ValueError("Invalid response from LLM")
        return response_parsed, response

    def process(self, humans, scenes, igraph, obj_segments, out_dir):
        res_path = osp.join(out_dir, "xforms.pkl")
        if osp.exists(res_path):
            with open(res_path, "rb") as fh:
                return pickle.load(fh)

        with torch.no_grad():
            human_data = list()
            for hid in range(len(humans)):
                hdata = humans[hid]()
                human_data.append(hdata)

            scene_data = list()
            for sid in range(len(scenes)):

                sdata = scenes[sid]()
                scene_data.append(sdata)

        def get_scene_id(name):
            for sid in range(len(scenes)):
                if scene_data[sid]["name"] == name:
                    return sid
            return None

        def get_human_id(name):
            return int(name.split(" ")[-1].strip()) - 1

        def get_object_part_labeling(object_name, part_name):
            object_parts = obj_segments[object_name]["object_part_labels"]
            object_labeling = obj_segments[object_name]["object_labeling"]
            opart_labeling = object_labeling == object_parts.index(part_name)
            return opart_labeling

        out_xforms = [list() for sid in range(len(scenes))]
        object_names = igraph.get_objects()
        for oid, oname in enumerate(object_names):
            sid = get_scene_id(oname)

            llm_input = OrderedDict()

            llm_input["interaction"] = self.interaction_prompt

            sbbox_min = torch.min(scene_data[sid]["pcds"][0], dim=0)[0]
            sbbox_max = torch.max(scene_data[sid]["pcds"][0], dim=0)[0]
            scenter = to_numpy((sbbox_min + sbbox_max) / 2.0)
            xform_t = get_translate_matrix(-scenter)
            xform_r = get_rotation_matrix("x", 180)
            xform = xform_r @ xform_t

            joint_indices = self.joint_indices
            hjoints = [to_numpy(human_data[hid]["joints"][0]) for hid in range(len(humans))]
            hjoints = np.stack(hjoints, axis=0)
            hjoints = apply_transforms3d(hjoints, np.expand_dims(xform, axis=0))

            llm_input["joints"] = list()
            for hid in range(len(humans)):

                hvertices = to_numpy(human_data[hid]["vertices"][0])
                hvertices = apply_transform3d(hvertices, xform)
                hfaces = to_numpy(human_data[hid]["faces"])
                save_o3d_mesh(osp.join(out_dir, f"scene_{sid}_human_{hid}_cano.ply"), V=hvertices, F=hfaces)

                for jname, jidx in joint_indices.items():
                    llm_input["joints"].append(
                        {
                            "name": f"person {hid + 1}, {jname}",
                            "position": [
                                round(hjoints[hid, jidx, 0].item(), 1),
                                round(hjoints[hid, jidx, 1].item(), 1),
                                round(hjoints[hid, jidx, 2].item(), 1),
                            ],
                        }
                    )

            llm_input["object"] = dict()

            llm_input["object"]["name"] = oname

            opcd = to_numpy(scene_data[sid]["pcds"][0])
            opcd = apply_transform3d(opcd, xform)
            obmin = np.amin(opcd, axis=0)
            obmax = np.amax(opcd, axis=0)
            llm_input["object"]["bounding_box"] = [
                round(obmin[0].item(), 1),
                round(obmin[1].item(), 1),
                round(obmin[2].item(), 1),
                round(obmax[0].item(), 1),
                round(obmax[1].item(), 1),
                round(obmax[2].item(), 1),
            ]

            save_o3d_pcd(osp.join(out_dir, f"scene_{sid}_cano.ply"), V=opcd)

            llm_input["object"]["parts"] = list()
            for pname in igraph.get_object_parts(oname):
                pdict = {"name": pname}
                plabels = get_object_part_labeling(oname, pname)
                plabels = to_numpy(plabels)
                if plabels.any().item():
                    ppcd = opcd[plabels, :]
                    pbmin = np.amin(ppcd, axis=0)
                    pbmax = np.amax(ppcd, axis=0)
                    pdict["bounding_box"] = [
                        round(pbmin[0].item(), 1),
                        round(pbmin[1].item(), 1),
                        round(pbmin[2].item(), 1),
                        round(pbmax[0].item(), 1),
                        round(pbmax[1].item(), 1),
                        round(pbmax[2].item(), 1),
                    ]
                llm_input["object"]["parts"].append(pdict)

            llm_input["contact"] = list()
            for iedge in igraph.edges:
                inodes = iedge.nodes
                if inodes[0].obj.dtype == "object" and inodes[1].obj.dtype == "object":
                    continue
                if inodes[0].obj.name != oname and inodes[1].obj.name != oname:
                    continue
                llm_input["contact"].append(f"{inodes[0].obj.name}, {inodes[0].name}, {inodes[1].obj.name}, {inodes[1].name}")

            with open(osp.join(out_dir, f"scene_{sid}_llm_input.json"), "w") as fh:
                json.dump(llm_input, fh, indent=4)

            max_tries = 3
            for attempt in range(max_tries):
                try:
                    boxpred_res, boxpred_raw = self.query_mllm(json.dumps(llm_input))
                    break
                except Exception as e:
                    if attempt == max_tries - 1:
                        raise RuntimeError(f"LLM query failed after {max_tries} attempts")

            with open(osp.join(out_dir, f"scene_{sid}_llm_output.txt"), "w") as fh:
                fh.write(boxpred_raw)

            with open(osp.join(out_dir, f"scene_{sid}_llm_output.json"), "w") as fh:
                json.dump(boxpred_res, fh, indent=4)

            xform_r = get_rotation_matrix("x", 180)
            xform_r_inv = np.linalg.inv(xform_r)
            for tid, tdict in enumerate(boxpred_res["rotations"]):
                xform_pred = get_rotation_matrix("y", tdict["y_angle"])

                save_o3d_pcd(
                    osp.join(out_dir, f"scene_{sid}_trans_{tid}.ply"),
                    V=apply_transform3d(opcd, xform_pred),
                )

                out_xforms[sid].append(xform_r_inv @ xform_pred @ xform_r)

        out_xforms = [[out_xforms[sid][tid] for sid in range(len(scenes))] for tid in range(len(out_xforms[0]))]

        with open(res_path, "wb") as fh:
            pickle.dump(out_xforms, fh)

        return out_xforms
