# Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2309.06135-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2309.06135)
[![LLaMA](https://img.shields.io/badge/Project_Page-P4D-FFB000.svg?style=flat-square)](https://joycenerd.github.io/prompting4debugging/)
[![huggingface](https://img.shields.io/badge/Dataset-%F0%9F%A4%97-78ac62.svg?style=flat-square)](https://huggingface.co/datasets/joycenerd/p4d)



Official Implementation of the ICML 2024 paper Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts

Paper: https://arxiv.org/abs/2309.06135 \
Authors: [Zhi-Yi Chin](https://joycenerd.github.io/) $^{\dagger*}$, Chieh-Ming Jiang $^{\dagger*}$, [Ching-Chun Huang](http://acm.cs.nctu.edu.tw/) $^\dagger$, [Pin-Yu Chen](https://sites.google.com/site/pinyuchenpage) $^\ddagger$, [Wei-Chen Chiu](https://walonchiu.github.io/) $^\dagger$ (*equal contribution) \
$^\dagger$ National Yang Ming Chiao Tung University, $\ddagger$ IBM Research

![](figure/teaser.png)

Text-to-image diffusion models, e.g. Stable Diffusion (SD), lately have shown remarkable ability in high-quality content generation, and become one of the representatives for the recent wave of transformative AI. Nevertheless, such advance comes with an intensifying concern about the misuse of this generative technology, especially for producing copyrighted or NSFW (i.e. not safe for work) images. Although efforts have been made to filter inappropriate images/prompts or remove undesirable concepts/styles via model fine-tuning, the reliability of these safety mechanisms against diversified problematic prompts remains largely unexplored. In this work, we propose **Prompting4Debugging (P4D)** as a debugging and red-teaming tool that automatically finds problematic prompts for diffusion models to test the reliability of a deployed safety mechanism. We demonstrate the efficacy of our P4D tool in uncovering new vulnerabilities of SD models with safety mechanisms. Particularly, our result shows that around half of prompts in existing safe prompting benchmarks which were originally considered ``safe'' can actually be manipulated to bypass many deployed safety mechanisms, including concept removal, negative prompt, and safety guidance. Our findings suggest that, without comprehensive testing, the evaluations on limited safe prompting benchmarks can lead to a false sense of safety for text-to-image models.

## Installation

```bash
conda create -n diffusion python=3.10
conda activate diffusion
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

Change the `torch` version according to your CUDA version. See [here](https://pytorch.org/get-started/previous-versions/) for more details.

## Dataset Setup
Data use in our experiments can be find in **data/**.

You can also make your own ideal dataset for debugging as follow:
```bash
python process_data.py \
    --save-prompts custom_ideal_data.csv \
    --prompts-csv [ORIGINAL_PROMPT].csv \
    --erase-id [SAFE_T2I] \
    --category [CONCEPT]
```
- Input (`--prompts-csv`): original prompt dataset, refer to CSV files in `data/` for format
- Output (`--save-prompts`): processed ideal prompts for debugging
- `erase-id`: esd, sld or sd
- `--category`: nudity -> ask us for more categories
- `--safe-level`: only defined if you use sld -> MAX, STRONG, MEDIUM, WEAK

## Prompt Optimization

### P4D-N
```bash
python run_p4dn.py \
    --config ./configs/esd_nudity_optmz_config.json \
    --data ./data/unsafe-prompts-nudity_esd.csv \
    --save-prompts ./esd_nudity_optmz.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --category nudity \
    --erase-id esd \
    --device cuda:x \
    --device-2 cuda:y
```

### P4D-K
```bash
python run_p4dk.py \
    --config ./configs/esd_nudity_optmz_config.json \
    --data ./data/unsafe-prompts-nudity_esd.csv \
    --save-prompts ./esd_nudity_optmz.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --category nudity \
    --erase-id esd \
    --device cuda:x \
    --device-2 cuda:y
```

- Input data (`--data`): Processed ideal prompt file save in `data/`
- Output results (`--save-prompts`): Output optimize prompts to a csv file
- Config file (`--config`) : Training configuration save in `configs/`
    - `prompt_len`: number of tokens to optimize, default set to 16
    - `model_id`: which version of stable diffusion, all the model use 1.4 except SD with negative prompt use 2.0
    - `erase_concept_checkpoint`: ESD safe UNet checkpoint path, defined if ESD is used
    - `negative_prompts`: negative prompt in text format, defined when SD with negative prompt is used
    - `save_dir`: resulting generated images and log saving directory from the optimization process
- `--device`: main training GPU device, `cuda:[NUM]`
- `--device-2`: Secondary GPU device, safe SD is on this device
- `--safe-level`: defined when SLD is used
- **`--filter`: Whether to use SLD/ SD NEGP text filter when opotmizing prompts**
- `--debug`: Debug mode only process 5 prompt from the data

ESD UNet checkpoints can be download from:
1. [ESD Project Website](https://erasing.baulab.info/weights/esd_models/)
2. [ESD Hugging Face Space](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/tree/main/models)

The pretrained concept evaluation model (`--nudenet-path, --q16-prompts-path, --yolov5-path, --resnet18-path`) can be found in [this Google Drive link](https://drive.google.com/drive/folders/1u8rb7ABCwA0Ns59hlM3VxGwE8S0iuGI5?usp=drive_link)

## Quantitative Results

### Main Results (concept and obejct)
![](figure/quantitative.png)

### Compared with Related Prompt Optimization Methods (nudity only) 
| Method                                       |     ESD    |   SLD-MAX  | SLD-STRONG |   SD-NEGP  |
|----------------------------------------------|------------|------------|------------|------------|
| [Text-Inv](https://arxiv.org/abs/2208.01618) |   11.91%   |   13.73%   |   35.71%   |    8.13%   |
| [PEZ-Orig](https://arxiv.org/abs/2302.03668) |   12.47%   |   24.51%   |   28.57%   |   20.57%   |
| [PEZ-PInv](https://arxiv.org/abs/2302.03668) |   26.59%   |   22.06%   |   22.32%   |   12.44%   |
| OURS (P4D-N)                                 |   50.65%   |   25.67%   |   34.03%   |   25.44%   |
| OURS (P4D-K)                                 |   47.19%   |   38.69%   |   37.84%   |   20.36%   |
| OURS (P4D-UNION)                             | **66.58%** | **52.66%** | **55.29%** | **40.98%** |

## Qualitative

Please refer to our [project page](https://joycenerd.github.io/prompting4debugging/) for qualitative results.

## Acknowledgement
- PEZ: https://github.com/YuxinWenRick/hard-prompts-made-easy
- Nudenet: https://github.com/notAI-tech/NudeNet
- ESD: https://erasing.baulab.info/
- SLD: https://github.com/ml-research/safe-latent-diffusion

## TODO
- [x] P4D-N code
- [x] P4D-K code
- [ ] Update Project Page

## Citation
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2309.06135-EE4C2C.svg?style=flat-square)](https://doi.org/10.48550/arXiv.2309.06135)

Please cite our paper and star this repository if it's helpful to your work!

```
@inproceedings{chin2023prompting4debugging,
  title={Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts},
  author={Zhi-Yi Chin and Chieh-Ming Jiang and Ching-Chun Huang and Pin-Yu Chen and Wei-Chen Chiu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024},
  url={https://arxiv.org/abs/2309.06135}
}
```

## Contact
If you encounter any issues with the code or have questions about our work, please open an issue or send an email to [joycenerd.cs09@nycu.edu.tw](mailto:joycenerd.cs09@nycu.edu.tw).

