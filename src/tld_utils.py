#coding=utf-8
import cv2
import numpy as np
import math
import random

# 传进来的points和box都是列表的形式
def drawBox(image, box, color, thick):
  # Mat& image, CvRect box, Scalar color, int thick
  # cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
  cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, thick)

def drawPoints(image, points, color):
  # Mat& image, vector<Point2f> points,Scalar color
  for i in range(len(points)):
    center = (points[i][0], points[i][1])
    # 绘制圆形也很简单，只需要确定圆心与半径
    # cv2.circle(img,(200,200),50,(55,255,155),1)#修改最后一个参数
    cv2.circle(image, center, 2, color, 1)

def median(v):
  # vector<float> v
  n = int(math.floor(len(v) / 2))
  return v[n]

def createMask(image, box):
  # const Mat& image, CvRect box
  mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
  drawBox(mask, box, (255, 255, 255), cv2.FILLED)
  return mask

def index_shuffle(begin, end):
  indexes = []
  for i in range(end-begin):
    indexes.append(i)
  random.shuffle(indexes)
  return indexes

def swap(matrix1, matrix2):
  return matrix2, matrix1

def bbOverlap(box1, box2):
  # box1 - scaled bbox, box2 - initial bbox
  # box.shape = x, y, width ,height
  if box1[0] > box2[0] + box2[2]:
    return 0.0
  if box1[1] > box2[1] + box2[3]:
    return 0.0
  if box1[0] + box1[2] < box2[0]:
    return 0.0
  if box1[1] + box1[3] < box2[1]:
    return 0.0

  colInt = min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0])
  rowInt = min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1])

  intersection = colInt * rowInt
  area1 = box1[2] * box1[3]
  area2 = box2[2] * box2[3]
  return intersection / (area1 + area2 - intersection)

def mat_operator(mat, bbox):
  """
  :param mat: opencv Mat - image
  :param bbox: bounding box
  :return: cropped image
  """
  x = bbox[0]
  y = bbox[1]
  width = bbox[2]
  height = bbox[3]
  return mat[y:y+height, x:x+width]

def getVar(box, sum, sqsum):
  """
  :param box: const BoundingBox&
  :param sum: const Mat&
  :param sqsum: const Mat&
  :return: double
  """
  brs = sum[box[1]+box[3], box[0]+box[2]]
  bls = sum[box[1]+box[3], box[0]]
  trs = sum[box[1], box[0]+box[2]]
  tls = sum[box[1], box[0]]
  brsq = sqsum[box[1]+box[3], box[0]+box[2]]
  blsq = sqsum[box[1]+box[3], box[0]]
  trsq = sqsum[box[1],box[0]+box[2]]
  tlsq = sqsum[box[1], box[0]]
  mean = (brs+tls-trs-bls) / float(box[2] * box[3])
  smean =  (brsq+tlsq-trsq-blsq) / float(box[2] * box[3])
  return smean - mean * mean

def make_pair(item1, item2):
  pair = (item1, item2)
  return pair

