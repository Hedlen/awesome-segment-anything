
# Awesome Segment Anything [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
"Segment Anything" has led to a new breakthrough in the field of computer vision (CV), and this repository will continue to track and summarize the research progress of "Segment Anything" in various fields, including papers/projects/demo, etc. 

If you find this repository helpful, please consider Starts ⭐ or Sharing ⬆️. Thanks.

```News
- 2023.4.12: An initial version of add recent papers or projects.
```

## Contents

- [Basemodel](#basemodel) 
- [Derivative Papers](#derivativepapers) 
- [Derivative Projects](#derivativeprojects) 

## Papers/Projects
### Basemodel
| Title |Paper page | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:|
| CLIP | [arXiv](https://arxiv.org/abs/2103.00020) | - | [Code](https://github.com/openai/CLIP) | OPENAI | CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet. | 
| OWL-ViT | [ECCV2022](https://arxiv.org/abs/2205.06230) | - | [Code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) | Google | A open-vocabulary object detector. Given an image and a free-text query, it finds objects matching that query in the image. It can also do one-shot object detection | 
| Painter | [CVPR2023](https://arxiv.org/abs/2212.02499) | - | [Code](https://github.com/baaivision/Painter) | BAAI | A Generalist Painter for In-Context Visual Learning | 
| Grounding DINO | [arXiv](https://arxiv.org/abs/2303.05499) | [Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb) &[Huggingface](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo) | [Code](https://github.com/IDEA-Research/GroundingDINO) | IDEA | A stronger open-set object detector|
| Segment Anything | [arXiv](https://arxiv.org/abs/2304.02643) | [Project page](https://segment-anything.com/) | [Code](https://github.com/facebookresearch/segment-anything) | Meta | A stronger Large model which can be used to generate masks for all objects in an image | 
| SegGPT | [arXiv](https://arxiv.org/abs/2304.03284) | [Project page](https://huggingface.co/spaces/BAAI/SegGPT) | [Code](https://github.com/baaivision/Painter) | BAAI | Segmenting Everything In Context based on Painter |

### Derivative Papers
| Title |Paper page | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| :---:|
| Segment Anything Model (SAM) for Digital Pathology | [arXiv](https://arxiv.org/abs/2304.04155) | - | - | - | SAM model on representative segmentation tasks, including (1) tumor segmentation, (2) tissue segmentation, and (3) cell nuclei segmentation. |
| SAMCOD | [arXiv](https://arxiv.org/abs/2304.04709) | - | [Code](https://github.com/luckybird1994/SAMCOD) | - | This paper try to ask if SAM can address the Camouflaged object detection (COD) task. |

### Derivative Projects
#### Segmetion task
| Title | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| 
| Grounded Segment Anything |[Unofficial demo](https://github.com/camenduru/grounded-segment-anything-colab) & [Huggingface](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything) | [Code](https://github.com/IDEA-Research/Grounded-Segment-Anything) | - | Combining Grounding DINO and Segment Anything| - | 
| GroundedSAM Anomaly Detection |[Unofficial demo](https://github.com/camenduru/grounded-segment-anything-colab) & [Huggingface](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything) | [Code](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection)| - | Combining Grounding DINO and Segment Anything to segment any anomaly without any training. 
| Semantic Segment Anything | - | [Code](https://github.com/fudan-zvg/Semantic-Segment-Anything) | Fudan | Semantic Segment Anything (SSA) project enhances the Segment Anything dataset (SA-1B) with a dense category annotation engine. 
| Magic Copy | - |[Code](https://github.com/kevmo314/magic-copy) | - | Magic Copy is a Chrome extension that uses SAM. |
| Segment Anything with Clip | - |[Code](https://github.com/Curt-Park/segment-anything-with-clip) |  | | 
| SAM-Clip | - |[Code](https://github.com/Curt-Park/segment-anything-with-clip) | - | Classify the output masks of segment-anything with the off-the-shelf CLIP models.| 
| Prompt Segment Anything | - | [Code](https://github.com/RockeyCoss/Prompt-Segment-Anything)| - | An implementation of zero-shot instance segmentation using Segment Anything. |
| RefSAM | - |[Code](https://github.com/helblazer811/RefSAM) | - | Evaluating the basic performance of SAM on the Referring Image Segmementation task. | 
| SAM-RBox | - |[Code](https://github.com/Li-Qingyun/sam-mmrotate) | - | This is an implementation of SAM (Segment Anything Model) for generating rotated bounding boxes with MMRotate. |
| Open Vocabulary Segment Anything | - |[Code](https://github.com/ngthanhtin/owlvit_segment_anything) | - | An interesting demo by combining OWL-ViT of Google and Segment Anything of Meta. |
| SegDrawer | - |[Code](https://github.com/lujiazho/SegDrawer) | - | Simple static web-based mask drawer, supporting semantic drawing with SAM. |
| Annotation Anything Pipeline | - |[Code](https://github.com/Yuqifan1117/Annotation-anything-pipeline) | - | Combining GPT and SAM. Annotation anything just all in one-pipeline. |

#### Medical image Segmentation task
| Title | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| 
| SAM in Napari | - |[Code](https://github.com/MIC-DKFZ/napari-sam) | - | Segment anything with our Napari integration of SAM . |

#### Inpainting task
| Title | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| 
| SegAnythingPro | - |[Code](https://github.com/jinfagang/Disappear) | - | Apply SOTA techniques to make amazing visual effect such as inpainting, replacing etc. |
| Inpaint Anything | - |[Code](https://github.com/geekyutao/Inpaint-Anything) | - | Apply SOTA techniques to make amazing visual effect such as inpainting, replacing etc. |

#### Image Generation task
| Title | Project page | Code base | Affiliation| Description|
|:---:|:---:|:---:|:---:| :---:| 
| Edit Anything | - |[Code](https://github.com/sail-sg/EditAnything) | - | Edit and Generate Anything in an image. |
| Image Edit Anything | - |[Code](https://github.com/feizc/IEA) | - | Using stable diffusion and segmentation anything models for image editing. |

## References

- [Ref Repo](https://github.com/JerryX1110/awesome-segment-anything-extensions)

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
