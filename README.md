# Nanodet + Deep Sort with PyTorch





<div align="center">
<p>
<img src="nanodet_file/demo.gif" width="640"/>
</p>
<br>
<a href="https://colab.research.google.com/drive/1ByKhrmgyhW4feRcgNolqr-legHtV7TbE?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>


## Introduction

This repository contains a two-stage-tracker. The detections generated by [Nanodet](https://github.com/RangiLyu/nanodet), a super fast and lightweight anchor-free object detection model, are passed to a [Deep Sort algorithm](https://github.com/ZQPei/deep_sort_pytorch) which tracks the objects. It can track any object that your Nanodet model was trained to detect.

What's more, we also retain the access to [YOLOv5](https://github.com/ultralytics/yolov5), a family of object detection architectures and models pretrained on the COCO dataset.

## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/ZhengJun-AI/Nanodet_DeepSort_Pytorch.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.7 or later with all [requirements.txt](https://github.com/ZhengJun-AI/Nanodet_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Tracking sources

Tracking can be run on most video formats (We offer an APP for utilizing mobile phone camera, see [release](https://github.com/ZhengJun-AI/Nanodet_DeepSort_Pytorch/releases/tag/v1.0))

```bash
python nanodet_track.py/enhanced_yolo_track.py --source 0  # webcam
                                                        img.jpg  # image
                                                        vid.mp4  # video
                                                        path/  # directory
                                                        path/*.jpg  # glob
                                                        'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                        'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

# examples
# nanodet detection with mobile phone camera
python nanodet_track.py --source http://192.168.137.162:4747/mjpegfeed?1920x1080 --img 416

# nanodet detection with video file
python nanodet_track.py --source pedestrian.mp4 --img 416 --class 0 2 --save-vid

# yolo detection with webcam
python enhanced_yolo_track.py --source 0 --yolo_model yolov5n.pt --img 640 --class 0
```

## Text results

Corresponding logs can be saved to your experiment folder `track/expN` by 

```bash
python nanodet_track.py --source ... --save-txt
```

![log.txt](nanodet_file/log.png)

## Filter tracked classes

By default the tracker tracks all MS COCO classes.

If you only want to track persons I recommend you to get [these weights](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) for increased performance

```bash
python enhanced_yolo_track.py --source 0 --yolo_model yolov5/weights/crowdhuman_yolov5m.pt --classes 0  # tracks persons, only
```

If you want to track a subset of the MS COCO classes, add their corresponding index after the classes flag

```bash
python nanodet_track.py --source 0 --classes 16 17  # tracks cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero.


## Knowledge Distillation & Pruning

What's more, we also perform knowledge distillation (KD) and model pruning on YOLO models. You can easily reproduce our results in folder `yolo_train` by following corresponding tips in the folder.

