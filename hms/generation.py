import os
import os.path as osp
import sys
import numpy as np
import nvdiffrast.torch as dr

GLCTX = dr.RasterizeCudaContext("cuda")

import math
import shutil
import torch
import torch.nn as nn
import torchvision
import pickle
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from PIL import Image

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.view_data import ViewData
from hms.human import SmplxHumans
from hms.scene import MeshScenes
from hms.scene_init import SceneInit
from hms.render import NVRenderer
from hms.mllm import Mllm
from hms.affordance import Affordance
from hms.video_diffusion import VideoDiffusion
from hms.video_processor import VideoProcessor
from hms.losses import LiftingLoss, Video3DData
from hms.common import may_create_folder, parent_folder, write_yaml, to_numpy, to_torch, to_cuda, get_device, seeding, valid_str, read_json, write_json, list_files, save_o3d_pcd, load_o3d_pcd, apply_transform3d
from hms.config import GenerationConfig, parse_config

class Generation(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(GenerationConfig, cfg)
        self.device = get_device()

        self.save_code(self.get_save_path("code", is_file=False))
        OmegaConf.save(self.cfg, self.get_save_path("config.yaml"))

        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs):

        view_data = ViewData(self.cfg.view_data)
        view_data_loader = DataLoader(
            view_data,
            num_workers=0,
            batch_size=view_data.cfg.batch_size,
            collate_fn=view_data.collate,
            shuffle=False,
        )
        self.view_data = [item for item in view_data_loader]

        self.human_cfgs = list()
        hid = 0
        while True:
            human_cfg = getattr(self.cfg, f"humans_{hid}", None)
            if human_cfg is None or len(human_cfg) == 0:
                break
            self.human_cfgs.append(human_cfg)
            hid += 1
        self.humans = [SmplxHumans(hcfg) for hcfg in self.human_cfgs]

        self.scene_cfgs = list()
        sid = 0
        while True:
            scene_cfg = getattr(self.cfg, f"scenes_{sid}", None)
            if scene_cfg is None or len(scene_cfg) == 0:
                break
            self.scene_cfgs.append(scene_cfg)
            sid += 1
        self.scenes = [MeshScenes(scfg) for scfg in self.scene_cfgs]

        self.renderer = NVRenderer(self.cfg.renderer, glctx=GLCTX, objects_to_render=self.humans + self.scenes)

        self.mllm = Mllm(cache_dir=self.cfg.cache_dir)

        self.video_diffusion = VideoDiffusion(self.cfg.video_diffusion, mllm=self.mllm)

        self.video_processor = VideoProcessor(self.cfg.video_processor, mllm=self.mllm)

        self.affordance = Affordance(
            self.cfg.affordance,
            mllm=self.mllm,
            glctx=GLCTX,
            renderer_cfg=self.cfg.renderer,
            scene_cfgs=self.scene_cfgs,
            interaction_prompt=self.cfg.interaction,
        )

        self.scene_init = SceneInit(
            self.cfg.scene_init,
            mllm=self.mllm,
            interaction_prompt=self.cfg.interaction,
        )

        if valid_str(self.cfg.wandb_api_key):
            os.environ["WANDB_API_KEY"] = self.cfg.wandb_api_key
        else:
            os.environ["WANDB_MODE"] = "offline"

        wandb_dir = self.cfg.exp_dir
        run_name = "@".join(self.cfg.exp_name.split("@")[-2:])
        notes = self.cfg.exp_name
        if valid_str(self.cfg.description):
            notes += f"; {self.cfg.description}"

        wandb.init(
            project=self.cfg.project,
            dir=wandb_dir,
            group=self.cfg.exp_group,
            tags=self.cfg.exp_tags if len(self.cfg.exp_tags) > 0 else None,
            name=run_name,
            notes=notes,
            settings=wandb.Settings(start_method="fork"),
            config=OmegaConf.to_container(OmegaConf.load(self.get_save_path("config.yaml"))),
        )
        wandb_cfg = {
            "id": wandb.run.id,
            "name": wandb.run.name,
            "group": wandb.run.group,
            "project": wandb.run.project,
            "url": wandb.run.url,
        }
        write_yaml(self.get_save_path("wandb.yaml"), wandb_cfg, flow_style=False)

    def run(self):
        self.global_step = 0
        n_frames = self.video_diffusion.cfg.n_frames

        afford_results = self.affordance.process(self.get_save_path("affordance", is_file=False))

        for oname, oseg in afford_results["object_segmentations"].items():
            oseg["object_labeling"] = to_torch(oseg["object_labeling"]).int().to(self.device)

        self.interaction_prompt = self.cfg.interaction
        self.interaction_prompt_enhanced = afford_results["video_prompt"]
        self.igraph = afford_results["interaction_graph"]
        self.obj_segments = afford_results["object_segmentations"]

        video_path, viewdata_path = self._generate_video(
            view_data=to_cuda(self.view_data[0], self.device),
            save_dir=self.get_save_path("gen_video", is_file=False),
        )

        video_data = self.video_processor.process(video_path=video_path, igraph=self.igraph, estimate_3d=True)
        if video_data is None:
            return
        with open(viewdata_path, "rb") as fh:
            view_data = pickle.load(fh)

        video3d_data = Video3DData(
            intrinsics=video_data["intrinsics"],
            object_points=video_data["object_points"],
            object_masks=video_data["object_masks"],
            object_part_points=video_data["object_part_points"],
            object_part_masks=video_data["object_part_masks"],
            human_points=video_data["human_points"],
            human_masks=video_data["human_masks"],
            human_poses=video_data["human_poses"],
            c2w=view_data["c2w"],
            mv_mat=view_data["mv_mat"],
            proj_mat=view_data["proj_mat"],
            n_joints=self.humans[0].smplx_model.NUM_BODY_JOINTS,
        )

        lifting_loss = LiftingLoss(
            self.cfg.lifting_loss,
            humans=self.humans,
            scenes=self.scenes,
            igraph=self.igraph,
            obj_segments=self.obj_segments,
            glctx=GLCTX,
        )

        save_dir = self.get_save_path("fit_humans", is_file=False)
        self._fit_humans(
            lifting_loss=lifting_loss,
            video3d_data=video3d_data,
            video_path=video_path,
            save_dir=save_dir,
        )

        object_names = self.igraph.get_objects()
        object_centers = list()
        with torch.no_grad():
            video_data = video3d_data()
            for oid, oname in enumerate(object_names):
                fcenters = list()
                findices = list()
                for j in range(n_frames):
                    opoints = video_data["object_points"][oid][j]
                    if opoints.shape[0] == 0:
                        fcenters.append(None)
                        continue
                    obcenter = (torch.min(opoints, dim=0)[0] + torch.max(opoints, dim=0)[0]) / 2
                    fcenters.append(obcenter)
                    findices.append(j)

                findices = np.asarray(findices)
                for j in range(n_frames):
                    if j not in findices:

                        nnfidx = findices[np.argmin(np.abs(findices - j))].item()
                        fcenters[j] = fcenters[nnfidx]
                fcenters = torch.stack(fcenters, dim=0).float().to(self.device)
                object_centers.append(fcenters)

        scene_centers = [None] * len(self.scenes)
        with torch.no_grad():
            scene_data = list()
            for sid in range(len(self.scenes)):
                sdata = self.scenes[sid]()
                scene_data.append(sdata)

            def get_scene_id(name):
                for sid in range(len(self.scenes)):
                    if scene_data[sid]["name"] == name:
                        return sid
                return None

            for oid, oname in enumerate(object_names):
                sid = get_scene_id(oname)
                sbbox_min = torch.min(scene_data[sid]["pcds"], dim=1)[0].float().to(self.device)
                sbbox_max = torch.max(scene_data[sid]["pcds"], dim=1)[0].float().to(self.device)
                scene_centers[sid] = (sbbox_min + sbbox_max) / 2.0
                self.scenes[sid].reset_translations(object_centers[oid] - scene_centers[sid])

        save_dir = self.get_save_path("init_scenes", is_file=False)
        scene_rotations = self.scene_init.process(
            humans=self.humans,
            scenes=self.scenes,
            igraph=self.igraph,
            obj_segments=self.obj_segments,
            out_dir=save_dir,
        )

        for epoch_id in range(len(scene_rotations)):

            for oid, oname in enumerate(object_names):
                sid = get_scene_id(oname)
                srots = to_torch(scene_rotations[epoch_id][sid]).float().to(self.device)
                srots = srots.unsqueeze(0)
                strans = torch.eye(4).float().to(self.device)
                strans = strans.unsqueeze(0).repeat(scene_centers[sid].shape[0], 1, 1)
                strans[:, :3, 3] = -scene_centers[sid]
                otrans = torch.eye(4).float().to(self.device)
                otrans = otrans.unsqueeze(0).repeat(object_centers[oid].shape[0], 1, 1)
                otrans[:, :3, 3] = object_centers[oid]
                init_xforms = otrans @ srots @ strans
                self.scenes[sid].reset_rotations(init_xforms[..., :3, :3])
                self.scenes[sid].reset_translations(init_xforms[..., :3, 3])

            save_dir = self.get_save_path(f"fit_objects_e{epoch_id}", is_file=False)
            self._fit_objects(
                lifting_loss=lifting_loss,
                video3d_data=video3d_data,
                video_path=video_path,
                save_dir=save_dir,
                epoch_id=epoch_id,
            )

            for view_data in self.view_data:
                self.visualize(view_data=view_data, save_dir=save_dir)

    def _generate_video(self, view_data, save_dir):
        n_frames = self.video_diffusion.cfg.n_frames

        view_id = view_data["index"][0]
        inter_dir = osp.join(save_dir, f"diffusion_intermediate-view_{view_id}")
        prompts_path = osp.join(save_dir, inter_dir, "prompts.json")
        vin_path = osp.join(save_dir, inter_dir, "diffusion_input.mp4")
        vout_path = osp.join(save_dir, f"diffusion_output-view_{view_id}.mp4")
        vdata_path = osp.join(save_dir, f"viewdata_{view_id}.pkl")

        may_create_folder(inter_dir)
        if osp.exists(vout_path) and osp.exists(vdata_path):
            return vout_path, vdata_path

        azimuth = int(round(view_data["azimuth"].item() * 180 / math.pi))
        elevation = int(round(view_data["elevation"].item() * 180 / math.pi))

        with torch.no_grad():
            centers = [self.humans[hid]()["joints"][0, 0].detach() for hid in range(len(self.humans))]
            centers = torch.stack(centers, dim=0).mean(0)
            rout = self.renderer(**view_data, center=centers)

        video = torch.where(rout["comp_mask"] > 0, rout["comp_rgb_nobg"], 1.0)
        if video.shape[0] == 1:
            video = video.repeat(n_frames, 1, 1, 1)
        self.save_video(vin_path, video)

        seeding(self.cfg.seed)
        vdict = self.video_diffusion(prompt=self.interaction_prompt, prompt_enhanced=self.interaction_prompt_enhanced, out_dir=inter_dir)

        video = vdict["video"]
        self.save_video(vout_path, video)
        self.save_video(vout_path, video, as_images=True)
        for vk, vv in vdict.items():
            if vk.startswith("video_"):
                self.save_video(osp.join(inter_dir, f"diffusion_{vk[6:]}.mp4"), vv)

        write_json(prompts_path, {vk: vv for vk, vv in vdict.items() if vk.startswith("prompt")})

        with open(vdata_path, "wb") as fh:
            pickle.dump(
                {
                    "azimuth": to_numpy(view_data["azimuth"]),
                    "elevation": to_numpy(view_data["elevation"]),
                    "c2w": to_numpy(view_data["c2w"]),

                    "mv_mat": to_numpy(rout["mv_mat"]),
                    "proj_mat": to_numpy(rout["proj_mat"]),

                },
                fh,
            )

        return vout_path, vdata_path

    def _fit_humans(self, lifting_loss, video3d_data, video_path, save_dir):
        n_frames = self.video_diffusion.cfg.n_frames

        ckpt_filenames = list_files(save_dir, "ckpt_humans_*.pth")
        reuse_ckpt = len(ckpt_filenames) == len(self.humans) and osp.exists(osp.join(save_dir, "ckpt_video3d.pth"))
        if reuse_ckpt:
            for hid in range(len(self.humans)):
                ckpt_path = osp.join(save_dir, f"ckpt_humans_{hid}.pth")
                self.humans[hid].load_state_dict(torch.load(ckpt_path, map_location=self.device))
            video3d_data.load_state_dict(torch.load(osp.join(save_dir, "ckpt_video3d.pth"), map_location=self.device))
            return

        with torch.no_grad():
            vdata = video3d_data()
            for hid in range(len(self.humans)):
                self.humans[hid].update(
                    transl=vdata["humans_transl"][hid],
                    global_orient=vdata["humans_global_orient"][hid],
                    betas=vdata["humans_betas"][hid],
                    body_pose=vdata["humans_body_pose"][hid],
                )

        for hid in range(len(self.humans)):
            self.humans[hid].export(output_dir=osp.join(save_dir, f"stage1_meshes_humans_{hid}"), interval=10)
            torch.save(self.humans[hid].state_dict(), osp.join(save_dir, f"ckpt_humans_{hid}.pth"))

        optimizer = torch.optim.Adam(video3d_data.parameters(), lr=self.cfg.lr)

        n_steps = lifting_loss.cfg.fit_human_steps
        pbar = tqdm(range(n_steps), disable=not self.cfg.verbose)
        for step_id in pbar:
            self.global_step += 1
            with torch.enable_grad():
                optimizer.zero_grad()

                loss_terms = lifting_loss.fit_humans(
                    step_id=step_id,
                    n_steps=n_steps,
                    video_data=video3d_data(),
                    use_human_cd3d=True,
                    use_human_cd2d=True,
                )

                loss = torch.tensor(0).float().to(self.device)
                for lw, lt, ll in loss_terms:
                    loss = loss + lw * lt

                loss.backward()
                optimizer.step()

                msg = f"Fit Humans: Loss - {loss.item():.4f}"
                pbar.set_description(msg)

                log_dict = {"Loss": loss.item(), "Global Step": self.global_step}
                for lw, lt, ll in loss_terms:
                    log_dict[f"Loss_{ll}"] = lt.item()
                wandb.log(log_dict)

        torch.save(video3d_data.state_dict(), osp.join(save_dir, "ckpt_video3d.pth"))

        with torch.no_grad():
            vdata = video3d_data()

        video_dir = f"{video_path[:-4]}_processed"
        pmap_xforms = list()
        for j in range(n_frames):
            pxform = vdata["pmap_xform"][j]
            pmap_xforms.append({"frame": j, "transform": pxform.flatten().tolist()})
            pmap_path = osp.join(video_dir, f"pmap_{j:05d}.ply")
            if osp.exists(pmap_path):
                o3d_pcd = load_o3d_pcd(pmap_path)
                o3d_pcd["V"] = apply_transform3d(o3d_pcd["V"], to_numpy(pxform))
                save_o3d_pcd(osp.join(video_dir, f"pmap_xforms_{j:05d}.ply"), **o3d_pcd)
        write_json(osp.join(video_dir, "pmap_xforms.json"), {"transforms": pmap_xforms})

    def _fit_objects(self, lifting_loss, video3d_data, video_path, save_dir, epoch_id):
        ckpt_filenames = list_files(save_dir, "ckpt_scenes_*.pth")
        reuse_ckpt = len(ckpt_filenames) == len(self.scenes)
        if reuse_ckpt:
            for sid in range(len(self.scenes)):
                ckpt_path = osp.join(save_dir, f"ckpt_scenes_{sid}.pth")
                self.scenes[sid].load_state_dict(torch.load(ckpt_path, map_location=self.device))
            log_dict = read_json(osp.join(save_dir, "losses.json"))
            return log_dict

        for sid in range(len(self.scenes)):
            self.scenes[sid].export(output_dir=osp.join(save_dir, f"stage2_meshes_scenes_{sid}_before"), interval=10)

        parameters = list()
        for sid in range(len(self.scenes)):
            parameters += list(self.scenes[sid].parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.cfg.lr)

        n_steps = lifting_loss.cfg.fit_object_steps
        pbar = tqdm(range(n_steps), disable=not self.cfg.verbose)
        for step_id in pbar:
            self.global_step += 1
            with torch.enable_grad():
                optimizer.zero_grad()

                loss_terms = lifting_loss.fit_objects(
                    step_id=step_id,
                    n_steps=n_steps,
                    video_data=video3d_data(),
                    use_object_cd3d=True,
                    use_object_cd2d=True,
                    use_object_part_cd3d=True,
                    use_object_part_cd2d=True,
                    use_object_smooth_trans=True,
                    use_object_smooth_rot=True,
                    use_object_scale=False,
                    use_intersect=True,
                    use_nocontact=True,
                    use_contact_drift=True,
                )

                loss = torch.tensor(0).float().to(self.device)
                for lw, lt, ll in loss_terms:
                    loss = loss + lw * lt

                loss.backward()
                optimizer.step()

                msg = f"Fit Objects: Loss - {loss.item():.4f}"
                pbar.set_description(msg)

                log_dict = {"Loss": loss.item(), "Global Step": self.global_step}
                for lw, lt, ll in loss_terms:
                    log_dict[f"Loss_{ll}"] = lt.item()
                wandb.log(log_dict)

        with torch.no_grad():
            loss_terms = lifting_loss.fit_objects(
                step_id=n_steps - 1,
                n_steps=n_steps,
                video_data=video3d_data(),
                use_object_cd3d=True,
                use_object_cd2d=True,
                use_object_part_cd3d=True,
                use_object_part_cd2d=True,
                use_object_smooth_trans=True,
                use_object_smooth_rot=True,
                use_object_scale=False,
                use_intersect=True,
                use_nocontact=True,
                use_contact_drift=True,
            )
            loss = torch.tensor(0).float().to(self.device)
            for lw, lt, ll in loss_terms:
                loss = loss + lw * lt
        log_dict = {"Loss": loss.item(), "Global Step": self.global_step}
        for lw, lt, ll in loss_terms:
            log_dict[f"Loss_{ll}"] = lt.item()

        write_json(osp.join(save_dir, "losses.json"), log_dict)

        for sid in range(len(self.scenes)):
            self.scenes[sid].export(output_dir=osp.join(save_dir, f"stage2_meshes_scenes_{sid}_after"), interval=10)
            torch.save(self.scenes[sid].state_dict(), osp.join(save_dir, f"ckpt_scenes_{sid}.pth"))

        return log_dict

    def visualize(self, view_data, save_dir, *args, **kwargs):
        view_id = view_data["index"][0]

        view_data = to_cuda(view_data, self.device)
        with torch.no_grad():
            centers = [self.humans[hid]()["joints"][0, 0].detach() for hid in range(len(self.humans))]
            centers = torch.stack(centers, dim=0).mean(0)
            rout = self.renderer(**view_data, center=centers)

        video = torch.where(rout["comp_mask"] > 0, rout["comp_rgb_nobg"], 1.0)
        vout_path = osp.join(save_dir, f"val-view_{view_id}.mp4")
        self.save_video(vout_path, video)
        self.save_video(vout_path, video, as_images=True)

    def save_video(self, save_path, imgs, fps=8, as_images=False):
        if isinstance(imgs, list) and isinstance(imgs[0], Image.Image):
            imgs = [np.array(img) for img in imgs]
            imgs = np.stack(imgs, axis=0).astype(np.uint8)

        imgs = to_numpy(imgs)
        if imgs.shape[-1] == 1:
            imgs = np.tile(imgs, (1, 1, 1, 3))
        if imgs.shape[-1] != 3:
            assert imgs.shape[1] == 3
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        if imgs.dtype != np.uint8:
            if np.amax(imgs) < 1 + 1e-1:
                imgs = (np.clip(imgs, 0, 1) * 255).astype(np.uint8)
            else:
                imgs = np.clip(imgs, 0, 255).astype(np.uint8)

        if as_images:
            if save_path.endswith(".mp4"):
                save_path = save_path[:-4]
            if not osp.exists(save_path):
                may_create_folder(save_path)
            for i in range(imgs.shape[0]):
                Image.fromarray(imgs[i]).save(osp.join(save_path, f"{i:05d}.jpg"))
        else:
            torchvision.io.write_video(save_path, imgs, fps=fps)
        return save_path

    def save_code(self, save_path):
        fileexts = [".py", ".ipynb", ".slang", ".yaml", ".sh", ".job", ".json", ".txt", ".bin"]
        for fileext in fileexts:
            for filepath in Path(ROOT_DIR).glob(f"**/*{fileext}"):
                filepath = filepath.relative_to(ROOT_DIR)
                if filepath.parts[0] in []:
                    continue
                may_create_folder(osp.join(save_path, filepath.parent))
                filepath = str(filepath)
                shutil.copy(osp.join(ROOT_DIR, filepath), osp.join(save_path, filepath))

    def get_save_path(self, filename, is_file=True):
        save_path = osp.join(self.cfg.exp_dir, filename)
        if is_file:
            may_create_folder(parent_folder(save_path))
        else:
            may_create_folder(save_path)
        return save_path

if __name__ == "__main__":

    cfg = OmegaConf.from_cli()
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    app = Generation(cfg)
    app.run()
