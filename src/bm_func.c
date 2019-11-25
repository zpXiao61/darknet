#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

#include "darknet.h"
#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"


box bm_get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}


void bm_correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
/*
void get_yolo_layer_ids_and_output_len(network *net, int *ids, int *output_len){
  int i;
  int j=0;
  for (i=0; i<net->n;i++){
    layer l = net->layers[i];
    if (l.type == YOLO){
      ids[j] = i;
      output_len[j] = l.outputs;
      j++;
    }
  }
}

void get_yolo_layer_output_data(network *net, int* ids, float* yolo0, float* yolo1, float* yolo2){
  int i, j;
  layer layer_0 = net->layers[ids[0]];
  for (j=0; j<layer_0.outputs; j++){
    yolo0[j] = layer_0.output[j];
  }
  layer layer_1 = net->layers[ids[1]];
  for (j=0; j<layer_1.outputs; j++){
    yolo1[j] = layer_1.output[j];
  }
  layer layer_2 = net->layers[ids[2]];
  for (j=0; j<layer_2.outputs; j++){
    yolo2[j] = layer_2.output[j];
  }
}

*/
int bm_num_detections_one_yolo_layer(float* yolo, int size, int classes, int obj_num, float thresh){
    int i, n;
    int count = 0;
    for (i = 0; i < size*size; ++i){
        for(n = 0; n < obj_num; ++n){
            int obj_index  = n * (5 + classes) * size * size + 4 * size * size + i;
            if(yolo[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}


int bm_get_yolo_detections(float *predictions, float* anchors, int* mask, int classes, int obj_num, int lw, int lh, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    //float anchors[18] = {10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326};
    int i,j,n;
    int count = 0;
    for (i = 0; i < lw*lh; ++i){
        int row = i / lw;
        int col = i % lw;
        for(n = 0; n < obj_num; ++n){
            int obj_index  = n*(5+classes)*lw*lh + 4*lw*lh + i;
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = n*(5+classes)*lw*lh + i;
            dets[count].bbox = bm_get_yolo_box(predictions, anchors, mask[n], box_index, col, row, lw, lh, netw, neth, lw*lh);
            dets[count].objectness = objectness;
            dets[count].classes = classes;
            for(j = 0; j < classes; ++j){
                int class_index = n*(5+classes)*lw*lh + (5+j)*lw*lh + i;
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    bm_correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}


detection *bm_get_network_boxes(float* yolo0, float* yolo1, float* yolo2, float* anchors, int size_base, int n, int classes, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
  int i;
  int count;
  int nboxes = 0;
  nboxes += bm_num_detections_one_yolo_layer(yolo0, size_base, classes, n, thresh);
  nboxes += bm_num_detections_one_yolo_layer(yolo1, size_base * 2, classes, n, thresh);
  nboxes += bm_num_detections_one_yolo_layer(yolo2, size_base * 4, classes, n, thresh);
  if(num) *num = nboxes;
  detection *dets = calloc(nboxes, sizeof(detection));
  for (i=0; i<nboxes; i++){
    dets[i].prob = calloc(classes, sizeof(float));
  }
  
  detection *dets_ = dets;
  int mask_0[] = {6,7,8};
  int mask_1[] = {3,4,5};
  int mask_2[] = {0,1,2};
  count = bm_get_yolo_detections(yolo0, anchors, mask_0, classes, n, size_base, size_base, w, h, 416, 416, thresh, map, relative, dets_);
  dets_ += count;
  count = bm_get_yolo_detections(yolo1, anchors, mask_1, classes, n, size_base*2, size_base*2, w, h, 416, 416, thresh, map, relative, dets_);
  dets_ += count;
  count = bm_get_yolo_detections(yolo2, anchors, mask_2, classes, n, size_base*4, size_base*4, w, h, 416, 416, thresh, map, relative, dets_);

  return dets;
}


void get_preprocessed_image_data(int size, image im, float* out)
{
    int data_len = 1;
    data_len = 1 * 3 * size * size;
    image imr = letterbox_image(im, size, size);
    int i = 0;
    for (i=0; i< data_len; i++){
      out[i] = imr.data[i];
    }
    free_image(imr);
    return;
}

void bm_load_image_and_resize_to_arr(char *filename, int w, int h, int size, float* out, int* oriWH)
{
    image im = load_image(filename, w, h, 3);
    oriWH[0] = im.w;
    oriWH[1] = im.h;
    get_preprocessed_image_data(size, im, out);
    free_image(im);
    return;
}
