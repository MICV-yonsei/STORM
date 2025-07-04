# STORM
[CVPR 2025] Official Pytorch Code for Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis


# Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis (CVPR 2025)

> Woojung Han, Yeonkyung Lee, Chanyoung Kim, Kwanghyun Park, Seong Jae Hwang
> 
> Yonsei University
>
> Diffusion-based text-to-image (T2I) models have recently excelled in high-quality image generation, particularly in a training-free manner, enabling cost-effective adaptability and generalization across diverse tasks. However, while the existing methods have been continuously focusing on several challenges, such as "missing objects" and "mismatched attributes," another critical issue of "mislocated objects" remains where generated spatial positions fail to align with text prompts. Surprisingly, ensuring such seemingly basic functionality remains challenging in popular T2I models due to the inherent difficulty of imposing explicit spatial guidance via text forms. To address this, we propose STORM (Spatial Transport Optimization by Repositioning Attention Map), a novel training-free approach for spatially coherent T2I synthesis. STORM employs Spatial Transport Optimization (STO), rooted in optimal transport theory, to dynamically adjust object attention maps for precise spatial adherence, supported by a Spatial Transport (ST) Cost function that enhances spatial understanding. Our analysis shows that integrating spatial awareness is most effective in the early denoising stages, while later phases refine details. Extensive experiments demonstrate that STORM surpasses existing methods, effectively mitigating mislocated objects while improving missing and mismatched attributes, setting a new benchmark for spatial alignment in T2I synthesis.

<a href="https://arxiv.org/abs/2503.22168"><img src="https://img.shields.io/badge/arXiv-2301.13826-b31b1b.svg" height=22.5></a>
<a href="https://micv-yonsei.github.io/storm2025/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://www.youtube.com/watch?v=h-KjlzQT6UY&embeds_referring_euri=https%3A%2F%2Fmicv-yonsei.github.io%2F&source_ve_path=Mjg2NjY"><img src="https://img.shields.io/static/v1?label=5-Minute&message=Video&color=darkgreen" height=20.5></a> 
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/Attend-and-Excite)
[![Replicate](https://replicate.com/daanelson/attend-and-excite/badge)](https://replicate.com/daanelson/attend-and-excite) -->

<p align="center">
<img src="docs/storm_main.png" width="800px"/>  


## Environment
Our code builds on the requirement of the official [Attend&Excite repository](https://github.com/yuval-alaluf/Attend-and-Excite). To set up their environment, please run:

```
conda env create -f environment/environment.yaml
conda activate storm
python -m spacy download en_core_web_sm
```

On top of these requirements, we add several requirements which can be found in `environment/requirements.txt`. These requirements will be installed in the above command.


## Usage

<p align="center">
<img src="docs/method.png" width="800px"/>  
<br>
Example generations outputted by Stable Diffusion with Attend-and-Excite. 
</p>

To generate an image, you can simply run the `run.py` script. For example,
```
python run.py --prompt "a cat to the left of a blue dog" --seeds [0] --token_indices [[2,9],[None,8]]
```
Notes:

- To apply STORM on Stable Diffusion 2.1, specify: `--sd_2_1 True`
- You may run multiple seeds by passing a list of seeds. For example, `--seeds [0,1,2,3]`.
- If you do not provide a list of which token indices to alter using `--token_indices`, we will split the text according to the Stable Diffusion's tokenizer and display the index of each token. You will then be able to input which indices you wish to alter.
- If you wish to run the standard Stable Diffusion model without STORM, you can do so by passing `--run_standard_sd True`.
- All parameters are defined in `config.py` and are set to their defaults according to the official paper.

All generated images will be saved to the path `"{config.output_path}/{prompt}/{seed}"`. We will also save a grid of all images (in the case of multiple seeds) under `config.output_path`.

### Float16 Precision
When loading the Stable Diffusion model, you can use `torch.float16` in order to use less memory and attain faster inference:
```python
stable = StormPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
```
Note that this may result in a slight degradation of results in some cases. 

## Notebooks 
We provide Jupyter notebooks to reproduce the results from the paper for image generation

### Generation
`notebooks/demo.ipynb` enables image generation using a free-form text prompt with and without STORM.

## Acknowledgements 
This code is builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library as well as the [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/) codebase.

## Citation
If you found this code useful, please cite the following paper:  
```
@InProceedings{2025storm,
    author    = {Han, Woojung and Lee, Yeonkyung and Kim, Chanyoung and Park, Kwanghyun and Hwang, Seong Jae},
    title     = {Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025}
}
```
