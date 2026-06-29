# Installation

## 1. Python environment

Python 3.10 with PyTorch 2.3.1 / CUDA 11.8.

```bash
conda create -n hms python=3.10
conda activate hms

conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install \
    diffusers==0.30.3 transformers==4.45.1 accelerate==0.34.2 \
    omegaconf==2.3.0 roma==1.5.0 smplx==0.1.28 \
    "git+https://github.com/nghorbani/human_body_prior.git" \
    "git+https://github.com/NVlabs/nvdiffrast.git" \
    "git+https://github.com/openai/CLIP.git" \
    open-clip-torch==2.26.1 open3d==0.18.0 trimesh imageio "imageio[ffmpeg]" \
    opencv-python-headless mediapipe pysdf xatlas==0.0.9 \
    numpy==1.23.5 scipy==1.14.1 scikit-image scikit-learn \
    tqdm wandb randomname openai pillow plyfile slangpy==1.1.22
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
```

## 2. Third-party libraries (into `third_party/`)

Install each into `third_party/` following their instructions:

| Library | Source |
|---|---|
| SAM2 | https://github.com/facebookresearch/sam2 |
| MoGe | https://github.com/microsoft/MoGe |
| GVHMR | https://github.com/craigleili/GVHMR/tree/myupdate |

## 3. Models and data (into `data/`)

```
data/
  SMPL-X/
  env_textures/
  sketchfab_objects/
  log_scratch/
```

- [ ] SMPL-X / VPoser: download from https://smpl-x.is.tue.mpg.de
- [ ] Data to be released

## 4. Multimodal LLM backends

- Reasoning: set `OPENROUTER_API_KEY` to use [OpenRouter](https://openrouter.ai/).
- Visual detection: serve `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` with [vLLM](https://github.com/vllm-project/vllm) (OpenAI-compatible) and set `HMS_VLM_BASE_URL` to its endpoint.

## 5. Environment variables

| Variable | Default |
|---|---|
| `HMS_DATA_DIR` | `<repo>/data` |
| `HMS_THIRD_PARTY_DIR` | `<repo>/third_party` |
| `OPENROUTER_API_KEY` | empty |
| `HMS_VLM_BASE_URL` | `http://localhost:7698/v1` |
| `WANDB_API_KEY` | empty |
