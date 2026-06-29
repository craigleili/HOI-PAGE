import os
import os.path as osp
import randomname
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
DATA_DIR = os.environ.get("HMS_DATA_DIR", osp.join(ROOT_DIR, "data"))
THIRD_PARTY_DIR = os.environ.get("HMS_THIRD_PARTY_DIR", osp.join(ROOT_DIR, "third_party"))

NUM_FRAMES = 49
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 480

MLLM_CFG = {
    "openrouter": {
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "base_url": "https://openrouter.ai/api/v1",
    },
    "vllm-vlm": {
        "api_key": os.environ.get("HMS_VLM_API_KEY", "EMPTY"),
        "base_url": os.environ.get("HMS_VLM_BASE_URL", "http://localhost:7698/v1"),
    },
}

@dataclass
class GenerationConfig:
    project: str = "hms"
    description: Optional[str] = ""
    exp_group: Optional[str] = "cogvideox_iterative"
    exp_name: Optional[str] = None
    exp_tags: List[str] = field(default_factory=lambda: ["sketchfab"])
    exp_root_dir: str = f"{DATA_DIR}/log_scratch"
    exp_dir: str = ""
    cache_dir: str = f"{DATA_DIR}/log_scratch/cache"

    reuse_exp_dir: Optional[str] = ""
    reuse_subdirs: List[str] = field(
        default_factory=lambda: [
            "affordance",
            "gen_video",
            "fit_humans",
            "init_scenes",

        ]
    )
    exclude_files: List[str] = field(default_factory=lambda: [""])

    seed: int = 0

    view_data: dict = field(default_factory=dict)

    humans_0: dict = field(default_factory=dict)
    humans_1: dict = field(default_factory=dict)

    scenes_0: dict = field(default_factory=dict)
    scenes_1: dict = field(default_factory=dict)

    interaction: str = ""

    renderer: dict = field(default_factory=dict)

    mllm: dict = field(default_factory=dict)

    affordance: dict = field(default_factory=dict)

    video_diffusion: dict = field(default_factory=dict)

    video_processor: dict = field(default_factory=dict)

    scene_init: dict = field(default_factory=dict)

    lifting_loss: dict = field(default_factory=dict)
    lr: float = 1e-2

    verbose: bool = True

    wandb_api_key: str = os.environ.get("WANDB_API_KEY", "")

    def __post_init__(self):
        self.exp_name = get_run_name(self.exp_name)
        self.exp_dir = os.path.join(self.exp_root_dir, f"{self.project}-{self.exp_group}", self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        if self.reuse_exp_dir is not None and osp.exists(self.reuse_exp_dir):

            for subdir in self.reuse_subdirs:
                src_dir = osp.join(self.reuse_exp_dir, subdir)
                if not osp.exists(src_dir):
                    continue
                dst_dir = osp.join(self.exp_dir, subdir)
                os.makedirs(dst_dir, exist_ok=True)
                os.system(f"cp -r {src_dir}/* {dst_dir}/")

            for filename in self.exclude_files:
                filepath = osp.join(self.exp_dir, filename)
                if Path(filepath).is_file():
                    os.system(f"rm {filepath}")

@dataclass
class ViewDataConfig:
    batch_size: int = 1
    n_elevations: int = 1
    n_azimuths: int = 12
    elevation_range: Tuple[float, float] = (0, 0)
    azimuth_range: Tuple[float, float] = (0, 360)
    camera_distance: float = 4
    fovy_deg: float = 60.0

@dataclass
class SmplxHumansConfig:
    human_name: str = ""

    n_frames: int = NUM_FRAMES
    use_latent_pose: bool = False
    use_continous_rot_repr: bool = True

    vposer_path: str = f"{DATA_DIR}/SMPL-X/vposer_V02_05"

    smplx_path: str = f"{DATA_DIR}/SMPL-X/models_smplx_v1_1"
    smplx_type: str = "smplx"
    gender: str = "neutral"
    uv_path: str = f"{DATA_DIR}/SMPL-X/smplx_uv/smplx_uv_template.txt"
    tex_path: str = f"{DATA_DIR}/SMPL-X/smplx_uv/smplx_texture_f_alb_1024.png"
    segmentation_path: str = f"{DATA_DIR}/SMPL-X/SMPL_body_segmentation/smplx/smplx_vert_segmentation_merged_reduced.json"

    rotation_axes: str = "x"
    rotation_angles: List[float] = field(default_factory=lambda: [180])
    translation: Tuple[float, float, float] = (0.0, 0.2, 3.0)

@dataclass
class MeshScenesConfig:
    obj_name: str = ""
    obj_path: str = ""

    n_frames: int = NUM_FRAMES
    use_continous_rot_repr: bool = True

@dataclass
class NVRendererConfig:
    viewport_width: int = IMAGE_WIDTH
    viewport_height: int = IMAGE_HEIGHT

    envmap_path: str = f"{DATA_DIR}/env_textures/green_point_park/green_point_park_1k.hdr"
    envmap_bg_scale: float = 1.0
    envmap_sh_scale: float = 1.5
    boost: float = 1.0

    texture_size: Tuple[int, int] = (256, 512)

@dataclass
class AffordanceConfig:

    n_frames: int = 1

    n_elevations: int = 2
    n_azimuths: int = 4
    elevation_range: Tuple[float, float] = (-15, 15)
    azimuth_range: Tuple[float, float] = (45, 315)
    camera_distance_factor: float = 4

    afford_provider: str = "openrouter"
    afford_model: str = "deepseek/deepseek-r1"
    afford_task: str = "afford_predict"
    afford_temperature: float = 0.6
    afford_user_prompt: str = "{}"
    afford_system_prompt: str = """"You are a helpful assistant in analyzing human-object interactions.

- Task: You will be given a list of objects and a short text description of human interactions with these objects. Your task is to analyze all the interaction relations among human body parts and object parts and output the results as a graph in the JSON format.

- Input format: The input is provided in the JSON format as follows
{
	"objects": [
		"object 1",
		"object 2"
	],
	"interaction": "a short interaction description"
}

- Output format: Provide the output strictly in JSON format, without any additional explanation or commentary, structured as follows:
{
	"object part nodes": [
		"object 1, object part 1",
		"object 1, object part 2"
	],
	"body part nodes": [
		"person 1, human body part 1",
		"person 1, human body part 2"
	],
	"interaction edges": [
		{
			"nodes": [
				"object a, object part b",
				"person c, human body part d"
			],
			"is_rel_static": <true or false indicating if the two nodes' movements remain relatively stationary during interaction>,
			"is_continuous": <true or false indicating if the two nodes remain in continuous physical contact during interaction>
		},
		{
			"nodes": [
				"object x, object part y",
				"person z, human body part w"
			],
			"is_rel_static": <true or false>,
			"is_continuous": <true or false>
		}
	],
	"interaction": "a long description in 150 words summarizing the output interaction graph to guide a realistic video generation",
	"object states": [
		{
			"name": "object 1",
			"is_translational": <true or false indicating if object 1 has translational motions during interaction>,
			"is_rotational": <true or false indicating if object 1 has rotational motions during interaction>,
			"description": "a short description in 20 words identifying object 1 during interaction"
		},
		{
			"name": "object 2",
			"is_translational": <true or false>,
			"is_rotational": <true or false>,
			"description": "a short description in 20 words identifying object 2 during interaction"
		}
	],
	"human states": [
		{
			"name": "person 1",
			"description": "a short description in 20 words identifying person 1 during interaction"
		}
	]
}

- Rules for analysis:
  (1) There are two types of nodes in the output interaction graph: "object part nodes" representing object parts and "body part nodes" representing human body parts.
  (2) The "object part nodes" field represent a part-level segmentation of each input object. Segmentations should roughly cover the entire object without becoming excessively detailed. Use descriptive, specific part names rather than generic terms, for example, avoid "surface", "edge", "body", "base", "area", "cover", "support", "connector", "frame", and the like. Do not differentiate between left and right parts. Avoid numbering object parts. Example: For a "bike", use the following parts: "handlebar", "pedal", "seat", "frame tubes", "wheels". For a "skateboard", use the following parts: "longboard deck", "wheels". For a "cordless vacuum cleaner", use the following parts: "ergonomic hand grip", "wand", "floor roller". For a "ladder", use the following parts: "side rail tubes", "rungs". For a "boxing bag", use the following parts: "punching bag".
  (3) The "body part nodes" field must be the following: "left hand", "right hand", "left arm", "right arm", "left shoulder", "right shoulder", "left leg", "right leg", "left foot", "right foot", "head", "hips". Distinguish between left/right human body parts.
  (4) The "interaction edges" represent direct physical contact relationships between two end nodes. An edge connects an object part node to either a human body part node or another object part node. Do not connect part nodes within the same object. Example: when ironing on an ironing board, the soleplate part of an iron should be connected to the top flat panel part of the ironing board. Each edge has two attributes: "is_continuous" and "is_rel_static". The "is_continuous" attribute is true if the two end nodes are in continuous physical contact during the interaction process, otherwise false. Example: when holding a dumbbell, the hand is in continuous contact with the handle without any separation; when punching a boxing bag, the hands are not in continuous contact with the bag; when a person stepping up a ladder, the feet and hands are both not in continuous contact with the rungs. The "is_rel_static" attribute is true if the two end nodes' movements are relatively stationary to each other while being in continuous physical contact during the interaction process, otherwise false. Example: when riding a bike, hands are relatively stationary to the handlebar; when playing a guitar, the hand strumming strings is not relatively stationary to the main compartment of the guitar.
  (5) Explicitly mentioned body parts in the input "interaction" field must be included. Example: For a description "a person is lifting a single dumbbell with one hand", include either "left hand" or "right hand" in the analysis. If no specific body part is mentioned, use the most common ergonomic interactions in the physical contact analysis.
  (6) Focus on primary actions influencing object use or movement in the physical contact analysis. Example: For "a person walking and carrying a briefcase in one hand", the primary action for analysis is "carrying".
  (7) Ensure the identified object parts belong to their respective objects in the node and edge outputs of the interaction graph.
  (8) Ensure plausible distribution and avoid conflicts or duplication of human body parts during the interaction analysis.
  (9) Exclude environmental elements, like floor, ground, or wall, from the physical contact analysis.
  (10) The "interaction" field in the output JSON must concisely summarize the "interaction edges" of the graph to guide realistic video generation. Follow this structure:
	(a) Begin with the interaction(s) as described in the input short "interaction" description. Clearly specify each participant's role if multiple people or objects are involved. All motions must occur at an extremely slow pace.
	(b) Then describe the interaction motion details, focusing on physical contact between human body parts and object parts. If a human is specified to be non-static, make sure their body parts without physical contact show expressive movement. For example, when "skateboarding", the person's arms can swing to maintain balance, and the legs can bend slightly; when "cleaning with a cordless vacuum cleaner", the arm that is not holding the vacuum can swing naturally while walking; when "riding a scooter", one foot can remain static on the deck while the other swings to push off the ground and gain speed. Importantly, the human body parts without physical contact must also move in slow motion.
	(c) Next, describe the appearance of people, objects, and environments. For people, you must strictly include the following four aspects: their hair styles, facial expressions, clothes, and shoes. For example, "short black hair", "neutral facial expression", "wearing a gray shirt, blue jeans, and white sneakers". For objects, describe general type and appearance without overly specific details. The environment is always a clean, spacious indoor area with white walls and a wooden floor. Ensure the environment supports the action without adding unnecessary complexity.
	(d) The "interaction" summarization must not exceed 150 words.
  (11) The "object states" in the output JSON have four attributes, "name", "is_translational", "is_rotational", and "description", for each object. The "is_translational" attribute is true if the corresponding object has global translational motions during interaction, otherwise false. The "is_rotational" attribute is true if the corresponding object has global rotational motions during interaction, otherwise false. Both "is_translational" and "is_rotational" attributes must consider only the object's overall motion, not motions of individual parts, for example, a bike being ridden should be considered as moving translationally as a whole, while ignoring the rotation of its pedals. The object "description" attribute should clearly identify the object by briefly stating its type, appearance, and its interactions with human bodies, using no more than 20 words. The object "description" should be based on relevant "interaction edges" and the long "interaction" fields in the output. In the object "description", avoid using numerical or ordinal references.
  (12) The "human states" in the output JSON have two attributes, "name" and "description", for each person. The human "description" attribute should clearly identify the person by briefly stating their appearance and interactions with object parts in 20 words. The human "description" should be based on relevant "interaction edges" and the long "interaction" fields in the output. Avoid using numerical or ordinal references in the "description" attribute.

- Examples:
  (1) If the input is
{
	"objects": [
		"umbrella",
		"suitcase"
	],
	"interaction": "a person is dragging a suitcase with one hand and holding an open umbrella with the other hand while walking"
}
  then the output is
{
	"object part nodes": [
		"umbrella, canopy",
		"umbrella, shaft",
		"suitcase, main compartment",
		"suitcase, handle",
		"suitcase, wheels"
	],
	"body part nodes": [
		"person 1, left hand",
		"person 1, right hand",
		"person 1, left arm",
		"person 1, right arm",
		"person 1, left shoulder",
		"person 1, right shoulder",
		"person 1, left leg",
		"person 1, right leg",
		"person 1, left foot",
		"person 1, right foot",
		"person 1, head",
		"person 1, hips"
	],
	"interaction edges": [
		{
			"nodes": [
				"umbrella, shaft",
				"person 1, left hand"
			],
			"is_rel_static": true,
			"is_continuous": true
		},
		{
			"nodes": [
				"suitcase, handle",
				"person 1, right hand"
			],
			"is_rel_static": true,
			"is_continuous": true
		}
	],
	"interaction": "A person is dragging a suitcase's handle with the right hand and holding a open umbrella's shaft with the left hand while walking at a slow pace. The suitcase rolls smoothly behind them as they move, and the open umbrella is held steadily above. The person has black short hair and a neutral facial expression. They wear a gray shirt, blue jeans, and white sneakers. The scene takes place in a clean, spacious indoor area with white walls and a wooden floor.",
	"object states": [
		{
			"name": "umbrella",
			"is_translational": true,
			"is_rotational": false,
			"description": "the open umbrella being held"
		},
		{
			"name": "suitcase",
			"is_translational": true,
			"is_rotational": false,
			"description": "the suitcase being dragged"
		}
	],
	"human states": [
		{
			"name": "person 1",
			"description": "the person with black short hair who is wearing gray shirt and blue jeans and holding/dragging the objects"
		}
	]
}

  (2) If the input is
{
	"objects": [
		"bike"
	],
	"interaction": "a person is riding a bike"
}
  then the output is
{
	"object part nodes": [
		"bike, handlebar",
		"bike, pedal",
		"bike, seat",
		"bike, frame tubes",
		"bike, wheels"
	],
	"body part nodes": [
		"person 1, left hand",
		"person 1, right hand",
		"person 1, left arm",
		"person 1, right arm",
		"person 1, left shoulder",
		"person 1, right shoulder",
		"person 1, left leg",
		"person 1, right leg",
		"person 1, left foot",
		"person 1, right foot",
		"person 1, head",
		"person 1, hips"
	],
	"interaction edges": [
		{
			"nodes": [
				"bike, handlebar",
				"person 1, left hand"
			],
			"is_rel_static": true,
			"is_continuous": true
		},
		{
			"nodes": [
				"bike, handlebar",
				"person 1, right hand"
			],
			"is_rel_static": true,
			"is_continuous": true
		},
		{
			"nodes": [
				"bike, pedal",
				"person 1, left foot"
			],
			"is_rel_static": true,
			"is_continuous": true
		},
		{
			"nodes": [
				"bike, pedal",
				"person 1, right foot"
			],
			"is_rel_static": true,
			"is_continuous": true
		},
		{
			"nodes": [
				"bike, seat",
				"person 1, hips"
			],
			"is_rel_static": true,
			"is_continuous": true
		}
	],
	"interaction": "A person is riding a bike at a slow, steady pace in a clean, spacious indoor area with white walls and a wooden floor. Their hands grip the handlebars firmly and feet remain securely on the pedals. The bike has a simple, modern design with a black frame and straight handlebars. The rider has short brown hair and a neutral facial expression. They wear a blue shirt, black shorts, and white sneakers.",
	"object states": [
		{
			"name": "bike",
			"is_translational": true,
			"is_rotational": false,
			"description": "the bike having a black frame and being ridden"
		}
	],
	"human states": [
		{
			"name": "person 1",
			"description": "the person who is wearing blue shirt and black shorts and riding"
		}
	]
}

  (3) If the input is
{
	"objects": [
		"guitar"
	],
	"interaction": "a person is playing a guitar while standing"
}
  then the output is
{
	"object part nodes": [
		"guitar, neck",
		"guitar, main compartment"
	],
	"body part nodes": [
		"person 1, left hand",
		"person 1, right hand",
		"person 1, left arm",
		"person 1, right arm",
		"person 1, left shoulder",
		"person 1, right shoulder",
		"person 1, left leg",
		"person 1, right leg",
		"person 1, left foot",
		"person 1, right foot",
		"person 1, head",
		"person 1, hips"
	],
	"interaction edges": [
		{
			"nodes": [
				"guitar, neck",
				"person 1, left hand"
			],
			"is_rel_static": false,
			"is_continuous": true
		},
		{
			"nodes": [
				"guitar, main compartment",
				"person 1, right hand"
			],
			"is_rel_static": false,
			"is_continuous": true
		}
	],
	"interaction": "A person is playing a guitar while standing in a clean, spacious indoor area with white walls and a wooden floor. Their left hand is holding the guitar's fretboard, and their right hand is strumming the strings slowly. The guitar is a classic acoustic model with a polished wood finish. The person has short brown hair and a happy faical expression. They wear a black shirt, blue jeans, and black boots, gently swaying their body to the rhythm.",
	"object states": [
		{
			"name": "guitar",
			"is_translational": true,
			"is_rotational": false,
			"description": "the wooden guitar being played"
		}
	],
	"human states": [
		{
			"name": "person 1",
			"description": "the person with short brown hair who is wearing blue jeans and playing the guitar"
		}
	]
}
"""

    det_provider: str = "vllm-vlm"
    det_model: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    det_task: str = "part_detect"
    det_temperature: float = 0.1
    det_user_prompt: str = "Detect {} in the image and output all the bbox coordinates in JSON format."
    det_system_prompt: str = 'You are a helpful assistant in image understanding and object/part detection. When performing object detection, ensure that the detected objects precisely match the input text descriptions. When performing object part detection, the text input will be in the format: "the X part of object Y", and you should focus specifically on detecting part X within object Y, not object Y as a whole.'

    segmenter_model_path: str = f"{THIRD_PARTY_DIR}/sam2/checkpoints/sam2.1_hiera_large.pt"
    segmenter_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

@dataclass
class VideoDiffusionConfig:

    t2i_model_path: str = "black-forest-labs/FLUX.1-dev"
    i2v_model_path: str = "THUDM/CogVideoX-5b-I2V"

    t2i_trials: int = 20
    frame_width: int = IMAGE_WIDTH
    frame_height: int = IMAGE_HEIGHT
    n_frames: int = NUM_FRAMES

    prompt_suffix_image: str = (
        "The camera is fixed at shoulder height, capturing the entire human body and objects within the frame from a slight three-quarter side view."
    )
    prompt_suffix_video: str = (
        "The camera is fixed at shoulder height, capturing the entire human body and objects within the frame from a slight three-quarter side view."
    )
    negative_prompt: str = "close-up shot, body out of frame, cropped, cut off, incomplete body, missing limbs, cartoon, anime, illustration, digital paint, drawing, 3d render, CGI"

    det_provider: str = "vllm-vlm"
    det_model: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    det_task: str = "object_detect"
    det_temperature: float = 0.1
    det_user_prompt: str = "Detect {} in the image and output all the bbox coordinates in JSON format."
    det_system_prompt: str = 'You are a helpful assistant in image understanding and object/part detection. When performing object detection, ensure that the detected objects precisely match the input text descriptions. When performing object part detection, the text input will be in the format: "the X part of object Y", and you should focus specifically on detecting part X within object Y, not object Y as a whole.'

    segmenter_model_path: str = f"{THIRD_PARTY_DIR}/sam2/checkpoints/sam2.1_hiera_large.pt"
    segmenter_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    pose2d_model_path: str = "usyd-community/vitpose-plus-huge"

    icomp_provider: str = "openrouter"
    icomp_model: str = "openai/gpt-4.1-mini"
    icomp_task: str = "image_compare"
    icomp_temperature: float = 0.6
    icomp_user_prompt: str = "{}"
    icomp_system_prompt: str = """You are a helpful assistant in image understanding and comparison.
- Task: You will receive one image file that actually contains two separate images shown side-by-side (left and right), along with a short text describing human-object interactions. Look closely at both images and read the text description. Use the "Analysis Rules" below to decide which single image ("left" or "right") is a better match for both the rules and the text description.
- Input format:
	(1) One image file that includes two images placed next to each other horizontally, like this: [left image | right image].
	(2) One short text that describes the human-object interactions that should be happening in the images.
- Output format: You must output only one word: either "left" or "right". Do not add any other words, explanations, or comments.
- Analysis Rules:
	(1) Full Human Figures: Prefer the image where people are shown completely, from their heads down to their feet, inside the image area, and where the front faces of the main people involved in the interaction are clearly visible.
	(2) Correct Anatomy: Prefer the image where humans have normal-looking body parts and proportions. Avoid images showing people with distorted, disfigured, or anatomically incorrect limbs or bodies.
	(3) Matching Text Description: Prefer the image where the human-object interactions match the provided short text description.
	(4) Plausible Interactions: Prefer the image where interactions between people and objects look natural, physically plausible. Avoid interactions that involve problematic body parts, like strangely bent or extra limbs. Avoid images with unrealistic physics, like people or objects floating in the air.
	(5) Camera View: Prefer wide-shot images taken from a shoulder-height, three-quarter side view that clearly shows both the pose and the interaction. If that's not available, prefer side views over straight-on front views. Avoid images taken from high-up, low-down, or close-up views that crop or obscure full human figures. Also avoid images where people or objects are too close to walls or background objects.
	(6) Sharp Details: Prefer images with clear, sharp details, and avoid images with motion blur around human body parts.
	(7) Realistic Style: Prefer photographic or realistic images over cartoons, drawings, illustrations, or images with very artistic styles.
	(8) Do not consider the mood, feeling, or atmosphere of the image in your comparison.
"""

@dataclass
class VideoProcessorConfig:
    det_provider: str = "vllm-vlm"
    det_model: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    det_task: str = "object_detect"
    det_temperature: float = 0.1
    det_user_prompt: str = "Detect {} in the image and output all the bbox coordinates in JSON format."
    det_system_prompt: str = 'You are a helpful assistant in image understanding and object/part detection. When performing object detection, ensure that the detected objects precisely match the input text descriptions. When performing object part detection, the text input will be in the format: "the X part of object Y", and you should focus specifically on detecting part X within object Y, not object Y as a whole.'

    segmenter_model_path: str = f"{THIRD_PARTY_DIR}/sam2/checkpoints/sam2.1_hiera_large.pt"
    segmenter_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    pmap_model_path: str = f"{THIRD_PARTY_DIR}/moge-vitl/model.pt"

    erode_kernel: int = 3
    erode_iters: int = 1

    dilate_kernel: int = 3
    dilate_iters: int = 1

    smplx_path: str = f"{DATA_DIR}/SMPL-X/models_smplx_v1_1"
    smplx_type: str = "smplx"
    gender: str = "neutral"

@dataclass
class SceneInitConfig:
    boxpred_provider: str = "openrouter"
    boxpred_model: str = "deepseek/deepseek-r1"
    boxpred_task: str = "box_predict"
    boxpred_temperature: float = 0.6
    boxpred_user_prompt: str = "{}"
    boxpred_system_prompt: str = """You are a helpful assistant in analyzing 3D human-object interactions.

- Task: You will be given the following information:
  (1) a short human-object interaction description.
  (2) a list of 3D joint positions for one or more humans.
  (3) a specified 3D object involved in the interaction. The object has a 3D bounding box, a list of segmented parts, and optionally, bounding boxes for each part.
  (4) a list of physical contact pairs between an object part and a human body part.
  Your task is to generate four rotation transformation candidates for the specified 3D object. Each transformation should make the object's placement physically plausible, based on the human poses and the described interaction.

- Input format: The input is provided in the JSON format as follows
{
	"interaction": "a short interaction description",
	"joints": [
		{
			"name": "person 1, human body part 1",
			"position": [x1, y1, z1]
		},
		{
			"name": "person 1, human body part 2",
			"position": [x2, y2, z2]
		}
	],
	"object": {
		"name": "object 1",
		"bounding_box": [x_min, y_min, z_min, x_max, y_max, z_max],
		"parts": [
			{
				"name": "object part a",
				"bounding_box": [x_min, y_min, z_min, x_max, y_max, z_max]
			},
			{
				"name": "object part b"
			}
		]
	},
	"contact": [
		"object 1, object part a, person 1, human body part b"
	]
}

- Output format: Provide the output strictly in JSON format, without any additional explanation or commentary:
{
	"rotations": [
		{"y_angle": theta_1},
		{"y_angle": theta_2},
		{"y_angle": theta_3},
		{"y_angle": theta_4}
	]
}

- Rules for analysis:
  (1) The up direction is along the positive y-axis, and the gravity direction is along the negative y-axis.
  (2) Multiple people can be involved in the interaction. People are labeled as "person 1", "person 2", etc.
  (3) Human "joints" are specified by combining a person ID with their body part names, along with precise 3D coordinates. These positions are fixed and should not be transformed during the analysis.
  (4) Estimate person orientation using key joints such as the neck and shoulders.
  (5) The specified 3D object is placed near the origin of the world coordinate system, with its up direction along the positive y-axis and its front direction along the positive z-axis.
  (6) The object's bounding box is axis-aligned and defined by its minimum and maximum x, y, and z coordinates.
  (7) The object has a list of segmented "parts", roughly covering the entire object. Example: a "bike" has the following part names: "handlebar", "pedal", "seat", "frame tubes", "wheels". An object part can have a bounding box defined by its minimum and maximum x, y, and z coordinates. This part bounding box information is optional in the input, can be imprecise, and should only be used to get rough part locations or object orientations.
  (8) Each "contact" entry includes four parts: object name, object part name, person name, and human body part name. For example, "bicycle, handlebar, person 1, left hand" means the left hand of person 1 is in contact with the bicycle's handlebar.
  (9) You must return four rotation transformation candidates for the specified 3D object. The transformations must be different and cover diverse object placements relative to the humans. Each transformation is defined by a rotation angle in degrees around the y-axis (the "y_angle" field in the output json). Concretely, suppose a transformation candidate is defined as follows: "y_angle" is theta degrees. The coordinates of a 3D point [x, y, z] are transformed as follows: [x * cos(theta * pi / 180) + z * sin(theta * pi / 180), y, -x * sin(theta * pi / 180) + z * cos(theta * pi / 180)].
  (10) For each transformation, you must apply the transformation to the object and its part bounding boxes, then validate whether the transformed object and its parts respect the contact constraints with the specified human joint positions.
  (11) To generate valid object transformations, consider all available input information: the interaction description, joint positions, human orientations, object bounding box, object part bounding boxes, and the part-level contact information.
"""

@dataclass
class LiftingLossConfig:
    fit_human_steps: int = 300
    fit_object_steps: int = 600

    human_cd3d_weight: Tuple[float, float] = (1e3, 1e3)
    human_cd2d_weight: Tuple[float, float] = (1e-3, 1e-3)

    object_cd3d_weight: Tuple[float, float] = (1e3, 1e3)
    object_cd2d_weight: Tuple[float, float] = (1e-4, 1e-4)

    object_part_cd3d_weight: Tuple[float, float] = (1e2, 1e2)
    object_part_cd2d_weight: Tuple[float, float] = (1e-4, 1e-4)

    object_smooth_trans_weight: Tuple[float, float] = (1e3, 1e3)
    object_smooth_rot_weight: Tuple[float, float] = (1e3, 1e3)

    object_scale_weight: Tuple[float, float] = (1.0, 1.0)

    intersect_weight: Tuple[float, float] = (0.0, 10.0)
    nocontact_weight: Tuple[float, float] = (1e3, 1e3)
    contact_drift_weight: Tuple[float, float] = (10, 1e3)

def get_run_name(name):
    if name is None:
        name = "run"
    if name.count("@") < 2:
        timestamp = datetime.now().strftime("@%y%m%d_%H%M%S")
        name += timestamp + f"@{randomname.get_name()}"
    return name

def parse_config(fields, cfg):
    from omegaconf import OmegaConf

    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
