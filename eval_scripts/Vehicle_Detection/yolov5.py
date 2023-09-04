import torch
from PIL import Image
from pathlib import Path

import sys
FILE = Path(__file__).absolute()
sys.path.insert(0, FILE.parents[0].as_posix())  # add yolov5/ to path
from .utils.torch_utils import select_device, load_classifier
from models.experimental import attempt_load
from utils.general import check_img_size, is_ascii, non_max_suppression
from utils.datasets import LoadPILImages


class YOLOv5():
    def __init__(self, device, half=False, weights='yolov5s.pt', imgsz=[640, 640]):
        # Initialize
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = weights[0] if isinstance(weights, list) else weights
        classify, suffix = False, Path(w).suffix.lower()
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
        self.pt = pt
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if self.pt:
            model = attempt_load(weights, map_location=self.device)  # load FP32 model
            self.stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16
            if classify:  # second-stage classifier
                modelc = load_classifier(name='resnet50', n=2)  # initialize
                modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
        else:
            raise NotImplementedError("Only .pt pretrained weights are supported for now.")
        
        # check image size
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

        # check model
        # if self.pt and self.device.type != 'cpu':
        #     model(torch.zeros(1, 3, self.imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        self.model = model
        

    @torch.no_grad()
    def detect(self, images: Image, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
        # Dataloader
        dataset = LoadPILImages(images, img_size=self.imgsz, stride=self.stride)
        bs = 1
        res = []
        
        # Run inference
        for _, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            pred = self.model(img, augment=False, visualize=False)[0]
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            res.append(1 if len(pred[0]) else 0)
        
        return res
            

if __name__ == "__main__":
    m = YOLOv5("cpu", weights="/eva_data0/evil-prompt/pretrained/vehicle_yolov5_best.pt")
    path = "./cars"
    imgs = []
    for img_path in Path(path).glob("*.jpg"):
        img = Image.open(img_path)
        imgs.append(img)
    res_car = m.detect(imgs)
    print(res_car)
    # paths = [
    #     "./b10_p139_19.jpg",
    #     "./b10_p146_13.jpg",
    #     "./b11_p176_6.jpg",
    #     "./b4_p78_18.jpg",
    #     "./b5_p82_6.jpg",
    #     "./b8_p115_17.jpg"
    # ]
    # for p in paths:
    #     img = Image.open(p)
    #     print(p, m.detect(img))
