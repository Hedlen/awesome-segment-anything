
<div align="center">
<br>
<image src="./imgs/teaser.png", width="600px", height="287px">
<br>
</div>
<!-- ![img](./imgs/teaser.png) -->

# Awesome Segment Anything [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Segment Anything has led to a new breakthrough in the field of Computer Vision (CV), and this repository will continue to track and summarize the research progress of Segment Anything in various fields, including Papers/Projects, etc. 

If you find this repository helpful, please consider Stars ⭐ or Sharing ⬆️. Thanks.

## News
```
- 2023.4.18: Add two nice job Inpainting Anything and SAM-Track.
- 2023.4.12: Add some presentations.
- 2023.4.12: An initial version of recent papers or projects.
```

## Contents

- [Basemodel Papers](#basemodel-papers) 
- [Derivative Papers](#derivative-papers)
- [Derivative Projects](#derivative-projects) 
  - [Image Segmentation task](#image-segmentation-task)
  - [Video Segmentation task](#video-segmentation-task)
  - [Medical image Segmentation task](#medical-image-segmentation-task)
  - [Inpainting task](#inpainting-task)
  - [3D task](#3d-task)
  - [Image Generation task](#image-generation-task)
  - [Remote Sensing task](#remote-sensing-task)
  - [Moving Object Detection task](#moving-object-detection-task)
  - [OCR task](#ocr-task)

## Papers/Projects
### Basemodel Papers
| Title |Presentation| Paper page | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
| CLIP | ![img](./imgs/clip.png) | [arXiv](https://arxiv.org/abs/2103.00020) | [Colab](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb) | [Code](https://github.com/openai/CLIP) | OpenAI | Contrastive Language-Image Pre-Training.| 
| OWL-ViT | ![img](https://github.com/google-research/scenic/raw/main/scenic/projects/owl_vit/data/text_cond_wiki_stillife_1.gif)| [ECCV2022](https://arxiv.org/abs/2205.06230) | - | [Code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) | Google | A open-vocabulary object detector. | 
| OvSeg | ![img](https://github.com/facebookresearch/ov-seg/raw/main/resources/ovseg.gif) | [CVPR2023](https://arxiv.org/abs/2210.04150) | [Project](https://jeff-liangf.github.io/projects/ovseg/) | [Code](https://github.com/facebookresearch/ov-seg) | META | Segment an image into semantic regions according to text descriptions.| 
| Painter | ![img](https://github.com/baaivision/Painter/raw/main/Painter/docs/teaser.jpg) | [CVPR2023](https://arxiv.org/abs/2212.02499) | - | [Code](https://github.com/baaivision/Painter) | BAAI | A Generalist Painter for In-Context Visual Learning.| 
| Grounding DINO | ![img](https://github.com/IDEA-Research/GroundingDINO/raw/main/.asset/hero_figure.png)| [arXiv](https://arxiv.org/abs/2303.05499) | [Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb) &[Huggingface](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo) | [Code](https://github.com/IDEA-Research/GroundingDINO) | IDEA | A stronger open-set object detector|
| Segment Anything | ![img](https://github.com/facebookresearch/segment-anything/raw/main/assets/model_diagram.png?raw=true)![img](https://github.com/facebookresearch/segment-anything/raw/main/assets/masks2.jpg?raw=true)| [arXiv](https://arxiv.org/abs/2304.02643) | [Project page](https://segment-anything.com/) | [Code](https://github.com/facebookresearch/segment-anything) | Meta | A stronger Large model which can be used to generate masks for all objects in an image.| 
| SegGPT | ![img](https://github.com/baaivision/Painter/raw/main/SegGPT/seggpt_teaser.png)| [arXiv](https://arxiv.org/abs/2304.03284) | [Project page](https://huggingface.co/spaces/BAAI/SegGPT) | [Code](https://github.com/baaivision/Painter) | BAAI | Segmenting Everything In Context based on Painter.|

### Derivative Papers
| Title | Presentation| Paper page | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
| CLIP_Surgery | ![img](https://github.com/xmed-lab/CLIP_Surgery/raw/master/figs/fig4.jpg)| [arXiv](https://arxiv.org/pdf/2304.05653.pdf) |[Demo](https://github.com/xmed-lab/CLIP_Surgery/blob/master/demo.ipynb)| [Code](https://github.com/xmed-lab/CLIP_Surgery) | HKUST | This work about SAM based on CLIP's explainability to achieve text to mask without manual points.|
| Segment Anything Model (SAM) for Digital Pathology | ![img](./imgs/samdp.png) | [arXiv](https://arxiv.org/abs/2304.04155) | - | - | - | SAM + Tumor segmentation/Tissue segmentation/Cell nuclei segmentation. |
| SAMCOD | - | [arXiv](https://arxiv.org/abs/2304.04709) | - | [Code](https://github.com/luckybird1994/SAMCOD) | - | SAM + Camouflaged object detection (COD) task.|
| Segment Anything Is Not Always Perfect | ![img](./imgs/sainap.png) | [arXiv](https://arxiv.org/pdf/2304.05750.pdf) | - | - | Samsung | This paper analyze and discuss the benefits and limitations of SAM.|
| Inpaint Anything | ![img1](./imgs/ia.png)|[arXiv](https://arxiv.org/abs/2304.06790)| - |[Code](https://github.com/geekyutao/Inpaint-Anything) | USTC & EIT | SAM + Inpainting, which is able to remove the object smoothly.|

### Derivative Projects
#### Image Segmentation task
| Title | Presentation|  Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:|
| Grounded Segment Anything | ![img](https://github.com/IDEA-Research/Grounded-Segment-Anything/raw/main/assets/acoustics/gsam_whisper_inpainting_demo.png)|[Colab](https://github.com/camenduru/grounded-segment-anything-colab) & [Huggingface](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything) | [Code](https://github.com/IDEA-Research/Grounded-Segment-Anything) | - | Combining Grounding DINO and Segment Anything| - | 
| GroundedSAM Anomaly Detection | ![img](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection/raw/master/assets/framework.png) | - | [Code](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection)| - | Grounding DINO + SAM to segment any anomaly. |
| Semantic Segment Anything | ![img](https://github.com/fudan-zvg/Semantic-Segment-Anything/raw/main/figures/sa_225091_class_name.png) |- | [Code](https://github.com/fudan-zvg/Semantic-Segment-Anything) | Fudan | A dense category annotation engine. |
| Magic Copy | ![img](./imgs/magic.png) | - |[Code](https://github.com/kevmo314/magic-copy) | - | Magic Copy is a Chrome extension that uses SAM. |
| Segment Anything with Clip | ![img](https://user-images.githubusercontent.com/14961526/230437084-79ef6e02-a254-421e-bd4c-32e87415c623.png) |  - |[Code](https://github.com/Curt-Park/segment-anything-with-clip) | -  | SAM + CLIP| 
| SAM-Clip | ![img](https://github.com/maxi-w/CLIP-SAM/blob/main/assets/example-segmented.png) | - |[Code](https://github.com/maxi-w/CLIP-SAM) | - | SAM + CLIP.| 
| Prompt Segment Anything | ![img](https://github.com/RockeyCoss/Prompt-Segment-Anything/blob/master/assets/example4.jpg)| - | [Code](https://github.com/RockeyCoss/Prompt-Segment-Anything)| - | SAM + Zero-shot Instance Segmentation.|
| RefSAM | - | - |[Code](https://github.com/helblazer811/RefSAM) | - | Evaluating the basic performance of SAM on the Referring Image Segmementation task.| 
| SAM-RBox | ![img](https://user-images.githubusercontent.com/79644233/230732578-649086b4-7720-4450-9e87-25873bec07cb.png) | - |[Code](https://github.com/Li-Qingyun/sam-mmrotate) | - | An implementation of SAM for generating rotated bounding boxes with MMRotate.|
| Open Vocabulary Segment Anything | ![img1](./imgs/ovsa.png)| - |[Code](https://github.com/ngthanhtin/owlvit_segment_anything) | - | An interesting demo by combining OWL-ViT of Google and SAM.|
| SegDrawer |![img1](https://github.com/lujiazho/SegDrawer/blob/main/example/demo.gif)![img](https://github.com/lujiazho/SegDrawer/blob/main/example/demo1.gif) | - |[Code](https://github.com/lujiazho/SegDrawer) | - | Simple static web-based mask drawer, supporting semantic drawing with SAM.|
| Annotation Anything Pipeline |![img](https://user-images.githubusercontent.com/48062034/231222391-5423f45c-6133-45f0-81b1-be0cdaeda545.png) | [YoutubeDemo](https://www.youtube.com/watch?v=5iQSGL7ebXE) |[Code](https://github.com/vietanhdev/anylabeling) | - | SAM + Labelme + LabelImg + Auto-labeling.|
| AnyLabel |![img](https://user-images.githubusercontent.com/18329471/231320488-2f8133bc-6b51-48f8-82a6-dd3b267f5156.png) | - |[Code](https://github.com/Yuqifan1117/Annotation-anything-pipeline) | - | GPT + SAM.|
| Roboflow Annotate |![roboflow-sam-optimized-faster](https://user-images.githubusercontent.com/870796/231834341-b0674467-ddc9-4996-b5ae-2d40dcc22409.gif) | - |[Blog](https://blog.roboflow.com/label-data-segment-anything-model-sam/) & [App](https://app.roboflow.com) | Roboflow | SAM-assisted labeling for training computer vision models.|
| SALT |![img](https://github.com/anuragxel/salt/raw/main/assets/how-it-works.gif) | - |[Code](https://github.com/anuragxel/salt) | - | A tool that adds a basic interface for image labeling and saves the generated masks in COCO format.]
| SAM U Specify |![img](./imgs/sus.png) | - |[Code](https://github.com/MaybeShewill-CV/segment-anything-u-specify) | - | Use SAM and CLIP model to segment unique instances you want.]


#### Video Segmentation task
| Title | Presentation| Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| MetaSeg | ![img](https://github.com/kadirnar/segment-anything-pip/releases/download/v0.2.2/metaseg_demo.gif) |[HuggingFace](https://huggingface.co/spaces/ArtGAN/Segment-Anything-Video) |[Code](https://github.com/kadirnar/segment-anything-video) | - | SAM + Video. |
| SAM-Track | [Video](https://www.youtube.com/watch?v=UPhtpf1k6HA&feature=youtu.be&themeRefresh=1) |[YoutubeDemo](https://www.youtube.com/watch?v=Xyd54AngvV8) |[Code](https://github.com/z-x-yang/Segment-and-Track-Anything) | - | This project, which is based on SAM and DeAOT, focuses on segmenting and tracking objects in videos. |

#### Medical image Segmentation task
| Title | Presentation|  Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| SAM in Napari |[Video](https://www.youtube.com/watch?v=OPE1Xnw487E)|- |[Code](https://github.com/MIC-DKFZ/napari-sam) | - | Segment anything with Napari integration of SAM.|
| SAM Medical Imaging |![img](https://miro.medium.com/v2/resize:fit:720/format:webp/1*_0vvAXd8LnMu2t0rEt5a0Q.png)|- |[Code](https://github.com/amine0110/SAM-Medical-Imaging) | - | SAM for Medical Imaging.|

#### Inpainting task
| Title | Presentation|  Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| SegAnythingPro | ![img](https://camo.githubusercontent.com/7d5fb67ffcd6c209cf22ffe302d95b3b46d36b92116fe216022bf2a359c4b588/68747470733a2f2f6a6968756c61622e636f6d2f676f646c792f666765722f2d2f7261772f6d61696e2f696d616765732f323032332f30342f31315f31325f345f34325f32303233303431313132303433392e706e67)|- |[Code](https://github.com/jinfagang/Disappear) | - | SAM + Inpainting/Replacing.|


#### 3D task
| Title | Presentation|  Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| 3D-Box | ![img](https://github.com/dvlab-research/3D-Box-Segment-Anything/raw/main/images/sam-voxelnext.png)|- |[Code](https://github.com/dvlab-research/3D-Box-Segment-Anything) | - | SAM is extended to 3D perception by combining it with VoxelNeXt.|
| Anything 3DNovel View | ![img](https://github.com/Anything-of-anything/Anything-3D/raw/main/novel-view/assets/1.jpeg)|- |[Code](https://github.com/Anything-of-anything/Anything-3D) | - | SAM + [Zero 1-to-3](https://github.com/cvlab-columbia/zero123).|
| Any 3DFace | ![img](https://github.com/Anything-of-anything/Anything-3D/raw/main/AnyFace3D/assets/celebrity_selfie/mask_2.jpg)![img](https://github.com/Anything-of-anything/Anything-3D/raw/main/AnyFace3D/assets/celebrity_selfie/2.gif)|- |[Code](https://github.com/Anything-of-anything/Anything-3D) | - | SAM + [HRN](https://younglbw.github.io/HRN-homepage/).|

#### Image Generation task
| Title | Presentation| Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| Edit Anything | ![img](https://github.com/sail-sg/EditAnything/blob/main/images/edit_sample1.jpg) | - |[Code](https://github.com/sail-sg/EditAnything) | - | Edit and Generate Anything in an image.|
| Image Edit Anything |![img](https://user-images.githubusercontent.com/37614046/230707537-206c0714-de32-41cd-a277-203fd57cd300.png)| - |[Code](https://github.com/feizc/IEA) | - | Stable Diffusion + SAM.|
| SAM for Stable Diffusion Webui |![img](./imgs/samsdb.png)| - |[Code](https://github.com/continue-revolution/sd-webui-segment-anything) | - | Stable Diffusion + SAM.|

#### Remote Sensing task
| Title | Presentation| Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| Earth Observation Tools | ![img](https://github.com/aliaksandr960/segment-anything-eo/raw/main/title_sameo.png?raw=true) |[Colab](https://colab.research.google.com/drive/1RC1V68tD1O-YissBq9nOvS2PHEjAsFkA?usp=share_link) |[Code](https://github.com/aliaksandr960/segment-anything-eo) | - | SAM + Remote Sensing. |

#### Moving Object Detection task
| Title | Presentation| Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| Moving Object Detection | ![img](https://camo.githubusercontent.com/cd073471951017a15cd445062d196242a446eb20acd90b2afa1728f239465fc7/687474703a2f2f7777772e616368616c646176652e636f6d2f70726f6a656374732f616e797468696e672d746861742d6d6f7665732f766964656f732f5a584e36412d747261636b65642d776974682d6f626a6563746e6573732d7472696d6d65642e676966) | - |[Code](https://github.com/achalddave/segment-any-moving) | - | SAM + Moving Object Detection. |


#### OCR task
| Title | Presentation| Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:| 
| OCR-SAM | ![img](https://github.com/yeungchenwa/OCR-SAM/raw/main/imgs/sam_vis.png) | [Blog](https://www.zhihu.com/question/593914819/answer/2976012032)|[Code](https://github.com/yeungchenwa/OCR-SAM) | - | Optical Character Recognition with SAM. |

## Acknowledgement
Some of the presentations in this repository are borrowed from the original author, and we are very thankful for their contribution.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
