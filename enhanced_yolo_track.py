# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)  # 检测识别网络
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    average_bbox_length = [0, 0, 0, 0]  # 四个区域的平均bbox大小
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        raw_img = img
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)  # yolo检测结果

        ###1(测试改进1)-------------------------水平翻转测试后与原测试结果结合------------------------#
        if not webcam:
            t1 = time_sync()
            w_max = raw_img.shape[2]
            flip_img = raw_img.transpose((1, 2, 0))
            flip_img = cv2.flip(flip_img, 1)
            flip_img = flip_img.transpose((2, 0, 1))

            flip_img = torch.from_numpy(flip_img).to(device)
            flip_img = flip_img.half() if half else flip_img.float()  # uint8 to fp16/32
            flip_img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if flip_img.ndimension() == 3:
                flip_img = flip_img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
            flip_pred = model(flip_img, augment=opt.augment, visualize=visualize)  # yolo检测结果
            flip_pred[:, :, 0] = w_max - flip_pred[:, :, 0] + 1
            pred = torch.cat([pred, flip_pred], dim=1)
            # if pre_pred!=None:
            #     pred = torch.cat([pred,flip_pred,pre_pred],dim = 1)

            # else:
            #     pred = torch.cat([pred,flip_pred],dim = 1)
            # pre_pred = pred_temp
        ###2(测试改进2)----------------------------自适应放大小目标区域------------------------------#
        if not webcam:
            patchs = []

            h_max, w_max = raw_img.shape[1], raw_img.shape[2]
            h_offset = [0, 0, h_max // 2, h_max // 2]
            w_offset = [0, w_max // 2, 0, w_max // 2]
            patchs.append(raw_img[:, :h_max // 2, :w_max // 2])
            patchs.append(raw_img[:, :h_max // 2, w_max // 2:])
            patchs.append(raw_img[:, h_max // 2:, :w_max // 2])
            patchs.append(raw_img[:, h_max // 2:, w_max // 2:])
            for i in range(4):
                if average_bbox_length[i] > 0.8 * np.max(average_bbox_length):
                    continue
                patch_img = patchs[i]

                patch_img = patch_img.transpose((1, 2, 0))
                patch_img = cv2.resize(patch_img, (0, 0), fx=2.0, fy=2.0)
                patch_img = patch_img.transpose((2, 0, 1))

                patch_img = torch.from_numpy(patch_img).to(device)
                patch_img = patch_img.half() if half else patch_img.float()  # uint8 to fp16/32
                patch_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if patch_img.ndimension() == 3:
                    patch_img = patch_img.unsqueeze(0)
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
                patch_pred = model(patch_img, augment=opt.augment, visualize=visualize)  # yolo检测结果
                patch_pred[:, :, 0] = patch_pred[:, :, 0] / 2 + w_offset[i]
                patch_pred[:, :, 1] = patch_pred[:, :, 1] / 2 + h_offset[i]
                patch_pred[:, :, 2] = patch_pred[:, :, 2] / 2
                patch_pred[:, :, 3] = patch_pred[:, :, 3] / 2
                pred = torch.cat([pred, patch_pred], dim=1)
        # ----------------------------

        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)  # NMS筛选框
        dt[2] += time_sync() - t3

        # Process detections
        t4 = time_sync()
        t5 = time_sync()

        ###3(测试改进2)--------存放框大小---------#
        bbox_dis_list = [[], [], [], []]
        # --------------------------#

        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])  # 坐标格式转换
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                # print(im0.shape)
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)  # 采用deepsort进行跟踪,标记个性标识
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        ###4(测试改进2)--------统计框大小-------------
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        if bbox_top < im0.shape[1] // 2:
                            if bbox_left < im0.shape[2] // 2:
                                bbox_dis_list[0].append((bbox_w ** 2 + bbox_h ** 2) ** 0.5)
                            else:
                                bbox_dis_list[1].append((bbox_w ** 2 + bbox_h ** 2) ** 0.5)
                        else:
                            if bbox_left < im0.shape[2] // 2:
                                bbox_dis_list[2].append((bbox_w ** 2 + bbox_h ** 2) ** 0.5)
                            else:
                                bbox_dis_list[3].append((bbox_w ** 2 + bbox_h ** 2) ** 0.5)
                        for i in range(4):
                            if len(bbox_dis_list[i]) != 0:
                                average_bbox_length[i] = (average_bbox_length[i] + np.mean(
                                    bbox_dis_list[i])) / 2  # 更新区域的框大小期望
                        # ---------------------------------------

                        if save_txt:
                            # to MOT format

                            # bbox_left = output[0]
                            # bbox_top = output[1]
                            # bbox_w = output[2] - output[0]
                            # bbox_h = output[3] - output[1]

                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

            else:
                deepsort.increment_ages()

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='./crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='./demo_video', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
