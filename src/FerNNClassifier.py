#coding=utf-8
import random
from time import time

import cv2
import numpy as np
import math
from Feature import Feature


class FerNNClassifier:
  valid = 0.5
  ncc_thesame = 0.95
  nstructs = 10
  structSize = 13
  thr_fern = 0.6
  thr_nn = 0.65
  thr__nn_valid = 0.7
  features = []
  posteriors = []
  pCounter = []
  nCounter = []
  pEx = []
  nEx = []
  numIndices = 0
  numScales = 0
  thrP = 0.0
  thrN = 0.0
  acum = 0

  def __init__(self):
    pass

  def _read(self, file):
    # Classifier Parameters
    self.valid = file["valid"]
    self.ncc_thesame = file["ncc_thesame"]
    self.nstructs = file["num_trees"]
    self.structSize = file["num_features"]
    self.thr_fern = file["thr_fern"]
    self.thr_nn = file["thr_nn"]
    self.thr_nn_valid = file["thr_nn_valid"]

  def getNumStructs(self):
    return self.nstructs

  def getFernTh(self):
    return self.thr_fern

  def getNNTh(self):
    return self.thr_nn

  def prepare(self, scales):
    # const vector<Size>& scales
    # Initialize test locations for features
    totalFeatures = self.nstructs * self.structSize
    for i in range(len(scales)):
      tmp = []
      self.features.append(tmp)

    for i in range(totalFeatures):
      x1f = random.random()
      x2f = random.random()
      y1f = random.random()
      y2f = random.random()
      for j in range(len(scales)):
        # scales[j][0] = width, scales[j][1] = height
        x1 = x1f * scales[j][0]
        y1 = y1f * scales[j][1]
        x2 = x2f * scales[j][0]
        y2 = y2f * scales[j][1]
        self.features[j].append(Feature(x1, y1, x2, y2))

    # Thresholds
    self.thrN = 0.5 * self.nstructs

    # Initialize Posteriors
    # positives = Pcounter, negatives = Ncounter
    for i in range(self.nstructs):
      self.posteriors.append([0]*pow(2, self.structSize))
      self.pCounter.append([0]*pow(2, self.structSize))
      self.nCounter.append([0]*pow(2, self.structSize))

  def getFeatures(self, image, scale_idx, fern):
    # 得到该Patch对应的特征值
    """
    :param image:
    :param scale_idx:
    :param fern:
    :return: None
    """
    for t in range(self.nstructs):
      leaf = 0
      for f in range(self.structSize):
        leaf = (leaf << 1) +  self.features[scale_idx][t*self.nstructs+f](image)
      fern[t] = leaf
    return fern

  def measure_forest(self, fern):
    # 计算该特征值对应的后验概率累加值
    """
    :param fern: vector<int> fern
    :return:
    """
    votes = 0.0
    for i in range(self.nstructs):
      votes += self.posteriors[i][fern[i]]
    return votes

  def trainF(self, ferns, resample):
    # const vector<std::pair<vector<int>,int> >& ferns,int resample
    # Conf = function(2,X,Y,Margin,Bootstrap,Idx)
    #                0 1 2 3      4         5
    # double *X     = mxGetPr(prhs[1]); -> ferns[i].first
    # int numX      = mxGetN(prhs[1]);  -> ferns.size()
    # double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
    # double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
    # int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
    self.thrP = self.thr_fern * self.nstructs
    for j in range(resample):
      for i in range(len(ferns)):
        if ferns[i][1] == 1:
          if self.measure_forest(ferns[i][0]) <= self.thrP:
            self.update(ferns[i][0], 1, 1)
          elif self.measure_forest(ferns[i][0]) >= self.thrN:
            self.update(ferns[i][0], 0, 1)

  def update(self, fern, C, N):
    # const vector<int>& fern, int C, int N
    for i in range(self.nstructs):
      idx = fern[i]
      if C == 1:
        self.pCounter[i][idx] += N
      else:
        self.nCounter[i][idx] += N

      if self.pCounter[i][idx] == 0:
        self.posteriors[i][idx] = 0
      else:
        self.posteriors[i][idx] = (float(self.pCounter[i][idx]) / self.pCounter[i][idx] + self.nCounter[i][idx])

  def trainNN(self, nn_examples):
    # const vector<cv::Mat>& nn_examples
    y = [0] * len(nn_examples)
    y[0] = 1
    for i in range(len(nn_examples)):
      isin, conf, dummy = self.NNConf(nn_examples[i])
      if y[i] == 1 and conf <= self.thr_nn:
        if isin[1] < 0:
          self.pEx = [nn_examples[i]]
          continue
        # pEx.insert(pEx.begin() + isin[1], nn_examples[i]);
        self.pEx.append(nn_examples[i])

      if y[i] == 0 and conf > 0.5:
        self.nEx.append(nn_examples[i])

    self.acum += 1
    print("%d. Trained NN examples: %d positive %d negative" % (self.acum, int(len(self.pEx)), int(len(self.nEx))))

  def NNConf(self, example):
    # 计算图像片Pattern到在线模型M(最近邻分类器)之间的相关相似度和保守相似度
    # isin[0]表示是否存在与样本完全匹配的正样本模型,isin[1]存放与样本最匹配的正样本索引,isin[2]表示是否存在与样本完全匹配的负样本模型
    isin = [-1, -1, -1]
    if len(self.pEx) == 0: # if is empty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0 # conf1 = zeros(1,size(x,2));
      csconf = 0
      return isin, rsconf, csconf
    if len(self.nEx) == 0:
      rsconf = 1
      csconf = 1
      return isin, rsconf, csconf
    ncc = np.zeros((1, 1), np.float32)
    csmaxP = 0.0
    maxP = 0.0
    maxN = 0.0
    maxPidx = 0
    validatedPart = math.ceil(len(self.pEx) * self.valid)
    anyN = False
    anyP = False
    example = example.astype(np.uint8)
    # measure NCC to positive examples
    for i in range(len(self.pEx)):
      self.pEx[i] = self.pEx[i].astype(np.uint8)
      cv2.matchTemplate(self.pEx[i], example, result=ncc, method=cv2.TM_CCORR_NORMED)
      nccP = (float(ncc[0][0] + 1) * 0.5)
      if nccP > self.ncc_thesame:
        anyP = True
      if nccP > maxP:
        maxP = nccP
        maxPidx = i
        if i < validatedPart:
          csmaxP = maxP
    # measure NCC to negative examples
    for i in range(len(self.nEx)):
      self.nEx[i] = self.nEx[i].astype(np.uint8)
      cv2.matchTemplate(self.nEx[i], example, result=ncc, method=cv2.TM_CCORR_NORMED)
      nccN = (float(ncc[0][0] + 1) * 0.5)
      if nccN > self.ncc_thesame:
        anyN = True
      if nccN > maxN:
        maxN = nccN
    # set isin
    if anyP:
      isin[0] = 1 # if the query patch is highly correlated with any positive patch
                  # in the model then it is considered to be one of them
    isin[1] = maxPidx
    if anyN:
      isin[2] = 1
    # Measure Relative Similarity
    dN = 1 - maxN
    dP = 1 - maxP
    rsconf = float(dN / (dN + dP))
    # Measure Conservative Similarity
    dP = 1 - csmaxP
    csconf = float(dN / (dN + dP))
    return isin, rsconf, csconf

  def evaluateTh(self, nXT, nExT):
    for i in range(len(nXT)):
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      fconf = float(self.measure_forest(nXT[i][0]) / self.nstructs)
      if fconf > self.thr_fern:
        self.thr_fern = fconf
    for i in range(len(nExT)):
      isin, conf, dummy = self.NNConf(nExT[i])
      if conf > self.thr_nn:
        self.thr_nn = conf
    if self.thr_nn > self.thr__nn_valid:
      self.thr__nn_valid = self.thr_nn

  def show(self):
    examples = np.zeros((len(self.pEx) * self.pEx[0].shape[0], self.pEx[0].shape[1]), dtype=np.uint8)
    # ex = np.zeros((self.pEx[0].shape[0], self.pEx[0].shape[1]), dtype=self.pEx[0].dtype)
    for i in range(len(self.pEx)):
      min_val = cv2.minMaxLoc(self.pEx[i])
      ex = np.copy(self.pEx[i])
      ex = ex - min_val[0]
      # tmp = examples[i*self.pEx[i].shape[0],(i+1)*self.pEx[i].shape[0]]
      tmp = np.uint8(ex)
    cv2.imshow("Examples", examples)



