import ctypes
from ctypes import *
import math
import random
import numpy as np
import sophon.sail as sail


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL(b"/home/bitmain/git_repo/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

#load_image = lib.load_image_color
#load_image.argtypes = [c_char_p, c_int, c_int]
#load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

#predict_image = lib.network_predict_image
#predict_image.argtypes = [c_void_p, IMAGE]
#predict_image.restype = POINTER(c_float)

bm_load_image_and_resize_to_arr = lib.bm_load_image_and_resize_to_arr
bm_load_image_and_resize_to_arr.argtypes = [c_char_p, c_int, c_int, c_int, c_int, POINTER(c_float), POINTER(c_int)]

bm_get_network_boxes = lib.bm_get_network_boxes
bm_get_network_boxes.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
bm_get_network_boxes.restype = POINTER(DETECTION)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(bm_engine, bm_graph_name, bm_input_tensor_name, meta, image, anchors, thresh=.5, hier_thresh=.5, nms=.45, net_w=416, net_h=416, max_stride=32, classes=80, channels=255):
    number_obj_per_point = int(channels / (5 + classes)) # 255 / 85 = 3

    input_len = 1 * 3 * net_w * net_h
    input_data = (c_float * input_len)()
    oriWH = (c_int * 2)()
    bm_load_image_and_resize_to_arr(image, 0, 0, net_w, net_h, input_data, oriWH)
    data = np.array(input_data).astype(np.float32).reshape(1, 3, net_h, net_w) # default, data shape is (1, 3, 416, 416) 

    input_dict = {bm_input_tensor_name: data}
    output = bm_engine.process(bm_graph_name, input_dict) # default, output is a dict with three outputs, Yolo0 Yolo1 Yolo2

    pnum = (c_int * 1)()
    yolo0_ptr = output['Yolo0'].ctypes.data_as(POINTER(ctypes.c_float))
    yolo1_ptr = output['Yolo1'].ctypes.data_as(POINTER(ctypes.c_float))
    yolo2_ptr = output['Yolo2'].ctypes.data_as(POINTER(ctypes.c_float))
    dets = bm_get_network_boxes(yolo0_ptr, yolo1_ptr, yolo2_ptr, anchors, net_w, net_h, max_stride, number_obj_per_point, classes, oriWH[0], oriWH[1], thresh, hier_thresh, None, 0, pnum)

    if (nms): do_nms_obj(dets, pnum[0], meta.classes, nms);

    res = []
    for j in range(pnum[0]):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, pnum[0])
    return res
    
if __name__ == "__main__":
    #net = load_net(b"cfg/tiny-yolo.cfg", b"tiny-yolo.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    anchors_ = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
    net_w = 416 # defaul use 416
    net_h = 416
    anchors = (c_float * 18)()
    for i in range(0,18,2):
        anchors[i] = anchors_[i] * 416 / net_w
        anchors[i+1] = anchors_[i+1] * 416 / net_h
    engine = sail.Engine("/home/bitmain/yolov3_bmodel/compilation.bmodel", "0", sail.IOMode.SYSIO)
    graph_name = engine.get_graph_names()[0]
    input_tensor_name = engine.get_input_names(graph_name)[0]
    r = detect(engine, graph_name, input_tensor_name, meta, ("data/dog.jpg").encode(encoding='utf-8'), anchors, net_w=net_w, net_h=net_h)
    print(r)
    

