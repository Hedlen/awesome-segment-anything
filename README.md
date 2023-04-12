
![img](./imgs/teaser.png)
# Awesome Segment Anything [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
"Segment Anything" has led to a new breakthrough in the field of computer vision (CV), and this repository will continue to track and summarize the research progress of "Segment Anything" in various fields, including Papers/Projects/Demo, etc. 

If you find this repository helpful, please consider Starts ⭐ or Sharing ⬆️. Thanks.

## News
```
- 2023.4.12: An initial version of add recent papers or projects.
```

## Contents

- [Basemodel Papers](#basemodel) 
- [Derivative Papers](#derivative-papers)
- [Derivative Projects](#derivative-projects) 
  - [Segmetion task](#segmetion-task)
  - [Medical image Segmentation task](#medical-image-segmentation-task)
  - [Inpainting task](#inpainting-task)
  - [Image Generation task](#image-generation-task)
  - [Video Segmentation task](#video-segmantation-task)

## Papers/Projects
### Basemodel
| Title |Paper page | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:|
| CLIP | [arXiv](https://arxiv.org/abs/2103.00020) | - | [Code](https://github.com/openai/CLIP) | OPENAI | CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs.| 
| OWL-ViT | [ECCV2022](https://arxiv.org/abs/2205.06230) | - | [Code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) | Google | A open-vocabulary object detector. Given an image and a free-text query, it finds objects matching that query in the image. It can also do one-shot object detection.| 
| Painter | [CVPR2023](https://arxiv.org/abs/2212.02499) | - | [Code](https://github.com/baaivision/Painter) | BAAI | A Generalist Painter for In-Context Visual Learning.| 
| Grounding DINO | [arXiv](https://arxiv.org/abs/2303.05499) | [Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb) &[Huggingface](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo) | [Code](https://github.com/IDEA-Research/GroundingDINO) | IDEA | A stronger open-set object detector|
| Segment Anything | [arXiv](https://arxiv.org/abs/2304.02643) | [Project page](https://segment-anything.com/) | [Code](https://github.com/facebookresearch/segment-anything) | Meta | A stronger Large model which can be used to generate masks for all objects in an image.| 
| SegGPT | [arXiv](https://arxiv.org/abs/2304.03284) | [Project page](https://huggingface.co/spaces/BAAI/SegGPT) | [Code](https://github.com/baaivision/Painter) | BAAI | Segmenting Everything In Context based on Painter.|

### Derivative Papers
| Title |Paper page | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:|
| Segment Anything Model (SAM) for Digital Pathology | [arXiv](https://arxiv.org/abs/2304.04155) | - | - | - | SAM model on representative segmentation tasks, including (1) tumor segmentation, (2) tissue segmentation, and (3) cell nuclei segmentation. |
| SAMCOD | [arXiv](https://arxiv.org/abs/2304.04709) | - | [Code](https://github.com/luckybird1994/SAMCOD) | - | This paper try to ask if SAM can address the Camouflaged object detection (COD) task.|

### Derivative Projects
#### Segmetion task
| Title | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| 
| Grounded Segment Anything |[Unofficial demo](https://github.com/camenduru/grounded-segment-anything-colab) & [Huggingface](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything) | [Code](https://github.com/IDEA-Research/Grounded-Segment-Anything) | - | Combining Grounding DINO and Segment Anything| - | 
| GroundedSAM Anomaly Detection | - | [Code](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection)| - | Combining Grounding DINO and Segment Anything to segment any anomaly without any training. |
| Semantic Segment Anything | - | [Code](https://github.com/fudan-zvg/Semantic-Segment-Anything) | Fudan | Semantic Segment Anything (SSA) project enhances the Segment Anything dataset (SA-1B) with a dense category annotation engine. |
| Magic Copy | - |[Code](https://github.com/kevmo314/magic-copy) | - | Magic Copy is a Chrome extension that uses SAM. |
| Segment Anything with Clip | - |[Code](https://github.com/Curt-Park/segment-anything-with-clip) | -  | SAM + CLIP| 
| SAM-Clip | - |[Code](https://github.com/Curt-Park/segment-anything-with-clip) | - | Classify the output masks of segment-anything with the off-the-shelf CLIP models.| 
| Prompt Segment Anything | - | [Code](https://github.com/RockeyCoss/Prompt-Segment-Anything)| - | An implementation of zero-shot instance segmentation using Segment Anything.|
| RefSAM | - |[Code](https://github.com/helblazer811/RefSAM) | - | Evaluating the basic performance of SAM on the Referring Image Segmementation task.| 
| SAM-RBox | - |[Code](https://github.com/Li-Qingyun/sam-mmrotate) | - | This is an implementation of SAM for generating rotated bounding boxes with MMRotate.|
| Open Vocabulary Segment Anything | - |[Code](https://github.com/ngthanhtin/owlvit_segment_anything) | - | An interesting demo by combining OWL-ViT of Google and Segment Anything of Meta.|
| SegDrawer | - |[Code](https://github.com/lujiazho/SegDrawer) | - | Simple static web-based mask drawer, supporting semantic drawing with SAM. |
| Annotation Anything Pipeline | - |[Code](https://github.com/Yuqifan1117/Annotation-anything-pipeline) | - | Combining GPT and SAM. Annotation anything just all in one-pipeline.|

#### Medical image Segmentation task
| Title | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| 
| SAM in Napari | ![video](https://www.youtube.com/watch?v=OPE1Xnw487E)|- |[Code](https://github.com/MIC-DKFZ/napari-sam) | - | Segment anything with our Napari integration of SAM.|

#### Inpainting task
| Title | Presentation|  Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| SegAnythingPro | ![img](https://camo.githubusercontent.com/7d5fb67ffcd6c209cf22ffe302d95b3b46d36b92116fe216022bf2a359c4b588/68747470733a2f2f6a6968756c61622e636f6d2f676f646c792f666765722f2d2f7261772f6d61696e2f696d616765732f323032332f30342f31315f31325f345f34325f32303233303431313132303433392e706e67)|- |[Code](https://github.com/jinfagang/Disappear) | - | SAM + Inpainting/Replacing.|
| Inpaint Anything | ![img1](https://github.com/geekyutao/Inpaint-Anything/blob/main/example/framework.png)|- |[Code](https://github.com/geekyutao/Inpaint-Anything) | - | SAM Meets Image Inpainting.|

#### Image Generation task
| Title | Presentation| Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| Edit Anything | ![img](https://github.com/sail-sg/EditAnything/blob/main/images/edit_sample1.jpg) | - |[Code](https://github.com/sail-sg/EditAnything) | - | Edit and Generate Anything in an image. |
| Image Edit Anything |![img](https://user-images.githubusercontent.com/37614046/230707537-206c0714-de32-41cd-a277-203fd57cd300.png)| - |[Code](https://github.com/feizc/IEA) | - | Using stable diffusion and segmentation anything models for image editing.|

#### Video Segmention task
| Title | Presentation| Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| MetaSeg | ![img](https://github.com/kadirnar/segment-anything-pip/releases/download/v0.2.2/metaseg_demo.gif) |[HuggingFace](https://huggingface.co/spaces/ArtGAN/Segment-Anything-Video) |[Code](https://github.com/kadirnar/segment-anything-video) | - | SAM + Video. |

## References

- [awesome-segment-anything-extensions](https://github.com/JerryX1110/awesome-segment-anything-extensions). 

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
