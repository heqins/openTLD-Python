#coding=utf-8
from numpy import linalg as LA
from DetStruct import DetStruct
from PatchGenerator import PatchGenerator
from FerNNClassifier import FerNNClassifier
from LKTracker import LKTracker
from tld_utils import *
from time import time
class TLD:
  good_boxes = []
  bad_boxes = []
  best_box = []
  lastbox = []
  lastconf = 0.0
  lastvalid = False
  # pX用于集合分类器的正样本集
  pX = []
  # ~相应负样本集
  nX = []
  # 用于最近邻分类器的正样本
  pEx = []
  # ~相应负样本
  nEx = []
  # nX拆分为y一半训练集nX和测试集nXT
  nXT = []
  # nEX同nX
  nExT = []
  scales = []
  grid = []
  ferns_data = []
  var = 0.0
  bbHull = [0, 0, 0.0, 0.0]
  dbb = [[0, 0, 0, 0]]
  dvalid = [False]
  dconf = [0.0]
  tbb = [0, 0, 0, 0]
  tconf = 0.0
  tracked = False
  tvalid = False
  detected = False
  classifier = FerNNClassifier()
  tracker = LKTracker()

  def __init__(self, parameter):
    # parameters type = dict
    self.parameters = parameter
    self._read(self.parameters)
    self.dt = DetStruct()
    # Init Generator
    # PatchGenerator用来对图像区域进行仿射变换的
    # self.generator = PatchGenerator(0, 0, self.noise_init, True, 1 - self.scale_init, 1 + self.scale_init,
    #                                 -self.angle_init * np.pi / 180, self.angle_init * np.pi / 180,
    #                                 -self.angle_init * np.pi / 180, self.angle_init * np.pi / 180)

  def _read(self, parameters):
    # type(parameters) = dict
    # Bounding Box Parameters
    self.min_win = parameters["min_win"]
    # Generator Parameters
    # initial parameters for positive examples
    self.patch_size = parameters["patch_size"]
    self.num_closest_init = parameters["num_closest_init"]
    self.num_warps_init = parameters["num_warps_init"]
    self.noise_init = parameters["noise_init"]
    self.angle_init = parameters["angle_init"]
    self.shift_init = parameters["shift_init"]
    self.scale_init = parameters["scale_init"]
    # update parameters for positive examples
    self.num_closest_update = parameters["num_closest_update"]
    self.num_warps_update = parameters["num_warps_update"]
    self.noise_update = parameters["noise_update"]
    self.angle_update = parameters["angle_update"]
    self.shift_update = parameters["shift_update"]
    self.scale_update = parameters["scale_update"]
    # parameters for negative examples
    self.bad_overlap = parameters["overlap"]
    self.bad_patches = parameters["num_patches"]
    self.classifier._read(parameters)

  def init(self, frame1, box, bb_file):
    # Get Bounding Boxes
    self.buildGrid(frame1, box)
    print("Created %d bounding boxes" % (len(self.grid)))
    # Preparation
    # allocation
    self.iisum =  np.zeros((frame1.shape[0] + 1, frame1.shape[1] + 1), np.float32)
    self.iisqsum = np.zeros((frame1.shape[0] + 1, frame1.shape[1] + 1), np.float64)
    self.tmp_conf = np.zeros((len(self.grid), 1), dtype=np.float32)
    self.tmp_patt = np.zeros((len(self.grid), 10), dtype=np.float32)
    self.pEx = np.zeros((self.patch_size, self.patch_size), np.float64)
    # Init generator
    self.generator = PatchGenerator(0, 0, self.noise_init, True, 1-self.scale_init, 1+self.scale_init, -self.angle_init*math.pi/180, self.angle_init*math.pi/180, -self.angle_init*math.pi/180, self.angle_init*math.pi/180)
    self.getOverlappingBoxes(box, self.num_closest_init)
    print("Found %d good boxes, %d bad boxes" % (len(self.good_boxes), len(self.bad_boxes)))
    print("Best Box: %d %d %d %d" %(self.best_box[0], self.best_box[1], self.best_box[2], self.best_box[3]))
    print("Bounding box hull: %d %d %d %d" % (self.bbHull[0], self.bbHull[1], self.bbHull[2], self.bbHull[3]))
    # Correct Bounding Box
    self.lastbox = self.best_box
    self.lastconf = 1
    self.lastvalid = True
    # self.lastbox.br() <-> .br()  the bottom right corner
    # print(bb_file, "%d, %d, %d, %d, %f" %(self.lastbox['x'], self.lastbox['y'], self.lastbox.br().x, self.lastbox.br().y ,lastconf))
    # Prepare Classifier
    self.classifier.prepare(self.scales)
    # Generate positive data
    self.generatePositiveData(frame1, self.num_warps_init)
    # Set variance threshold
    mean, stdev = cv2.meanStdDev(mat_operator(frame1, self.best_box))
    # cv2.integral() 计算积分图
    self.iisum, self.iisqsum = cv2.integral2(frame1)
    # getVar(best_box,iisum, iisqsum);
    self.var = (pow(stdev[0][0], 2) * 0.5)
    print("Variance: ", self.var)
    # check variance
    vr = getVar(self.best_box, self.iisum, self.iisqsum) * 0.5
    print("Check variance: ", vr)
    # Generate negative data
    self.generateNegativeData(frame1)
    # Split Negative Ferns into Training and Testing sets(they are already shuffled)
    half = int(len(self.nX) * 0.5)
    self.nXT = self.nX[0 + half : len(self.nX)]
    self.nX = self.nX[:half]
    # Split Negative NN examples into Training and Testing sets
    half = int(len(self.nEx) * 0.5)
    self.nExT = self.nEx[0 + half: len(self.nEx)]
    self.nEx = self.nEx[:half]
    # Merge Negative Data with Positive Data and shuffle it
    ferns_data = []
    for i in range(len(self.nX)+len(self.pX)):
      temp = [[], 0]
      ferns_data.append(temp)
    # 产生指定范围[begin:end]的随机数,返回随机数数组
    # vector<int> idx
    idx = index_shuffle(0, len(ferns_data))
    a = 0
    for i in range(len(self.pX)):
      ferns_data[idx[a]] = self.pX[i]
      a += 1
    for i in range(len(self.nX)):
      ferns_data[idx[a]] = self.nX[i]
      a += 1
    # follower_TLD: Data already have been shuffled, just putting it in the same vector
    nn_data = [0] * (len(self.nEx) + 1)
    nn_data[0] = self.pEx
    for i in range(len(self.nEx)):
      nn_data[i+1] = self.nEx[i]
    # Training
    self.classifier.trainF(ferns_data, 2) # bootstrap = 2
    self.classifier.trainNN(nn_data)
    self.classifier.evaluateTh(self.nXT, self.nExT)

  def generatePositiveData(self, frame, num_warps):
    """
    :param frame:
    :param num_warps:

    Generate Positive data
    Input:
      - good_boxes(bbP)
      - best_box(bbP0)
      - frame (im0)
    Output:
      - Positive fern features (pX)
      - Positive NN examples (pEx)
    """
    frame_operator = mat_operator(frame, self.best_box)
    self.pEx, mean, stdev = self.getPattern(frame_operator)
    # Get Fern features on warped patches
    img = np.zeros((frame.shape[0], frame.shape[1]), dtype=frame.dtype)
    cv2.GaussianBlur(src=frame, ksize=(9, 9), sigmaX=1.5, dst=img)
    # warped = mat_operator(img, self.bbHull)
    # pt = (self.bbHull[0] + (self.bbHull[2] - 1) * 0.5, self.bbHull[1] + (self.bbHull[3] - 1) * 0.5)
    del self.pX[:]
    # C++: std::vector<std::pair<std::vector<int>,int> > pX ->Python: px = list(int, list)
    # if (pX.capacity()<num_warps*good_boxes.size())
    #     pX.reserve(num_warps*good_boxes.size());
    for i in range(num_warps):
      if i > 0:
        # self.generator(frame, pt, warped, (self.bbHull[2], self.bbHull[3]))
        # warped一个指向img上数据的指针，这里修改的是img上bbnul那一块的数据
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pass
      for j in range(len(self.good_boxes)):
        fern = [0] * self.classifier.getNumStructs()
        idx = self.good_boxes[j]
        patch = mat_operator(img, self.grid[idx])
        # patch numpy.ndarray
        fern = self.classifier.getFeatures(patch, self.grid[idx][5], fern)
        self.pX.append([fern, 1])
    print("Positive examples generated: ferns:%d NN:1" % (len(self.pX)))

  def generateNegativeData(self, frame):
    """
    :Input: Image, bad_boxes(Boxes far from the bounding box), variance(pEx variance)
    :param frame
    :return: Negative fern features (nX), Negative NN examples(nEx)
    """
    # Random shuffle bad_boxes indexes
    np.random.shuffle(self.bad_boxes)
    # Get Fern Features of the boxes with big variance (calculated using integral images)
    a = 0
    print("Negative data generation started.")
    t = time()
    for i in range(len(self.bad_boxes)):
      idx = self.bad_boxes[i]
      if getVar(self.grid[idx], self.iisum, self.iisqsum) < self.var  * 0.5:
        continue
      # 计算剩下的box在随机森林中每棵树的决策叶节点
      # 1次耗时0.001s,总共大概10000+次
      fern = [0] * self.classifier.getNumStructs()
      patch = mat_operator(frame, self.grid[idx])
      # 此方法消耗时间占比最大
      fern = self.classifier.getFeatures(patch, self.grid[idx][5], fern)
      self.nX.append([fern, 0])
      a += 1
    print("Negative examples generated: ferns: %d " % a)
    print("Negative examples generation cost: %s seconds" % (time()-t))
    # bad_patches = 100
    for i in range(self.bad_patches):
      temp = []
      self.nEx.append(temp)
    for j in range(self.bad_patches):
      idx = self.bad_boxes[j]
      patch = mat_operator(frame, self.grid[idx])
      # 随机选取100个负样本,归一化到标准的Patch,计算每个patch的标准差, dum1 = mean, dum2 = stdev
      self.nEx[j], dum1, dum2 = self.getPattern(patch)
    print("NN: %d" % len(self.nEx))

  def processFrame(self, img1, img2, points1, points2, bbnext, lastboxfound, tl, bb_file):
    # const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,
    # bool& lastboxfound, bool tl, FILE* bb_file
    confident_detections = 0
    didx = 0 # detection index
    # Track
    if lastboxfound and tl:
      points1, points2 = self.track(img1, img2, points1, points2)
    else:
      self.tracked = False
    # Detect
    self.detect(img2)
    # Integration
    if self.tracked:
      bbnext = self.tbb
      self.lastconf = self.tconf
      self.lastvalid = self.tvalid
      print("Tracked")
      if self.detected:
        cbb, cconf = self.clusterConf(dbb=self.dbb, dconf=self.dconf)
        print("Found %d clusters " % (int(len(cbb))))
        for i in range(len(cbb)):
          if bbOverlap(self.tbb, cbb[i]) < 0.5 and cconf[i] > self.tconf:
            confident_detections += 1
            didx = i
        if confident_detections == 1:
          print("Found a better match... reinitializing tracking")
          bbnext = cbb[didx]
          self.lastconf = cconf[didx]
          self.lastvalid = False
        else:
          print("%d confident cluster was found" % confident_detections)
          cx = cy = cw = ch = 0
          close_detections = 0
          for i in range(len(self.dbb)):
            if bbOverlap(self.tbb, self.dbb[i]) > 0.7:
              cx += self.dbb[i][0]
              cy += self.dbb[i][1]
              cw += self.dbb[i][2]
              ch += self.dbb[i][3]
              close_detections += 1
              print("weighted detections %d %d %d %d" %(self.dbb[i][0], self.dbb[i][1], self.dbb[i][2], self.dbb[i][3]))
          if close_detections > 0:
            bbnext[0] = round(float(10 * self.tbb[0] + cx) / float(10 + close_detections))
            bbnext[1] = round(float(10 * self.tbb[1] + cy) / float(10 + close_detections))
            bbnext[2] = round(float(10 + self.tbb[2] + cw) / float(10 + close_detections))
            bbnext[3] = round(float(10 + self.tbb[3] + ch) / float(10 + close_detections))
            print("Tracker bb: %d %d %d %d" %(self.tbb[0], self.tbb[1], self.tbb[2], self.tbb[3]))
            print("Average bb: %d %d %d %d" %(bbnext[0], bbnext[1], bbnext[2], bbnext[3]))
            print("Weighting %d close detections(s) with tracked.." % close_detections)
          else:
            print("%d close detections were found" % close_detections)
    else:
      print("Not tracking...")
      lastboxfound = False
      self.lastvalid = False
      if self.detected:
        cbb, cconf = self.clusterConf(dbb=self.dbb, dconf=self.dconf)
        print("Found %d clusters" % (int(len(cbb))))
        if len(cconf) == 1:
          bbnext = cbb[0]
          self.lastconf = cconf[0]
          print("Confident detection.. reinitializing tracker")
          lastboxfound = True
    self.lastbox = bbnext
    if lastboxfound:
      # fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
      pass
    else:
      # fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
      pass
    if self.lastvalid and tl:
      self.learn(img2)
    return points1, points2, bbnext, lastboxfound

  def track(self, img1, img2, points1, points2):
    """
    Inputs:
      -current frame(img2), last frame(img1), last Bbox(bbox_f[0])
    Outputs:
      -Confidence(tconf), Predicted bounding box(tbb), Validity(tvalid), points2 (for display purpose only)
    """
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    points1 = self.bbPoints(points1, self.lastbox)
    if len(points1) < 1:
      print("BB= %d %d %d %d, Points not generated" % (self.lastbox[0], self.lastbox[1], self.lastbox[2], self.lastbox[3]))
      self.tvalid = False
      self.tracked = False
      return points1, points2
    # Frame-to-Frame tracking with forward-backward error checking
    self.tracked, points1, points2 = self.tracker.trackf2f(img1, img2, points1)
    if self.tracked:
      # Bounding box prediction
      print("points1: ", points1[:5])
      print("points2: ", points2[:5])
      print("self.lastbox: ", self.lastbox)
      self.tbb = self.bbPredict(points1,points2,self.lastbox)
      print("self.tbb: ", self.tbb)
      if self.tracker.getFB() > 10 or self.tbb[0] > img2.shape[1] or self.tbb[1] > img2.shape[0] or (self.tbb[0] + self.tbb[2])  < 1 or (self.tbb[1] + self.tbb[3]) < 1:
        self.tvalid = False
        self.tracked = False
        print("Too unstable predictions FB error=%f"%(self.tracker.getFB()))
        return points1, points2
      # Estimate  Confidence and Validity
      bb = [0, 0, 0, 0]
      bb[0] = max(self.tbb[0], 0)
      bb[1] = max(self.tbb[1], 0)
      bb[2] = min(min(img2.shape[0] - self.tbb[0], self.tbb[2]), min(self.tbb[2], self.tbb[0] + self.tbb[2]))
      bb[3] = min(min(img2.shape[1] - self.tbb[1], self.tbb[3]), min(self.tbb[3], self.tbb[1] + self.tbb[3]))
      pattern, mean, stdev = self.getPattern(mat_operator(img2, bb))
      isin, conf, self.tconf = self.classifier.NNConf(pattern) # Conservative Similarity
      self.tvalid = self.lastvalid
      if self.tconf > self.classifier.thr_nn_valid:
        self.tvalid = True
    else:
      print("No points tracked")
    return points1, points2

  @staticmethod
  def bbPoints(points, bb):
    max_pts = 10
    margin_h = 0
    margin_v = 0
    stepx = math.ceil((bb[2] - 2*margin_h)/max_pts)
    stepy = math.ceil((bb[3] - 2*margin_v)/max_pts)
    y = bb[1] + margin_v
    while y < bb[1] + bb[3] - margin_v:
      x = bb[0] + margin_h
      while x < bb[0] + bb[2] - margin_h:
        points.append([float(x), float(y)])
        x += stepx
      y += stepy
    return points

  @staticmethod
  def bbPredict(points1, points2, bb1):
    # const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
    #                     const BoundingBox& bb1,BoundingBox& bb2
    bb2 = [0, 0, 0, 0, 0, 0]
    npoints = len(points1)
    xoff = [0.0] * npoints
    yoff = [0.0] * npoints
    print("tracked points: ", npoints)
    for i in range(npoints):
      xoff[i] = points2[i][0] - points1[i][0]
      yoff[i] = points2[i][1] - points1[i][1]
    dx = median(xoff)
    dy = median(yoff)
    if npoints > 1:
      d = []
      i = 0
      while i < npoints:
        j = i + 1
        while j < npoints:
          d.append(LA.norm(points2[i]-points2[j])/LA.norm(points1[i]-points1[j]))
          j = j + 1
        i = i + 1
      s = median(d)
    else:
      s = 1.0
    s1 = 0.5 * (s - 1) * bb1[2]
    s2 = 0.5 * (s - 1) * bb1[3]
    print("s = %f s1= %f s2= %f" % (s, s1, s2))
    bb2[0] = int(round(bb1[0] + dx - s1))
    bb2[1] = int(round(bb1[1] + dy - s2))
    bb2[2] = int(round(bb1[2] * s))
    bb2[3] = int(round(bb1[3] * s))
    # Rect.br()返回rect的右下顶点的坐标
    print("Predicted bb: %d %d %d %d"%(bb2[0], bb2[1], bb2[2], bb2[3]))
    return bb2

  def detect(self, frame):
    # cleaning
    self.dbb = []
    self.dconf = []
    self.dt.bb = []
    t = time()
    img = np.zeros((frame.shape[0], frame.shape[1]), dtype=frame.dtype)
    self.iisum, self.iisqsum = cv2.integral2(frame)
    # 使用高斯模糊,去噪
    # cv2.GaussianBlur(src=frame, ksize=(9, 9), sigmaX=1.5, dst=img)
    cv2.GaussianBlur(src=frame, dst=img, ksize=(9,9), sigmaX=1.5)
    numtrees = self.classifier.getNumStructs()
    fern_th = self.classifier.getFernTh()
    ferns = [0] * 10
    a = 0
    for i in range(len(self.grid)):
      if getVar(self.grid[i], self.iisum, self.iisqsum) >= self.var:
        a += 1
        patch = mat_operator(img, self.grid[i])
        self.classifier.getFeatures(patch, self.grid[i][5], ferns)
        conf = self.classifier.measure_forest(ferns)
        self.tmp_conf[i][0] = conf
        self.tmp_patt[i] = np.asarray(ferns)
        if conf > numtrees * fern_th:
          self.dt.bb.append(i)
      else:
        self.tmp_conf[i][0] = 0.0
    detections = len(self.dt.bb)
    print("%d Bounding boxes passed the variance filter" % a)
    print("%d Initial detection from Fern_classifier" % detections)
    if detections > 100:
      # nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
      self.nth_element(self.dt.bb, sort_type="CComparator")
      self.dt.bb = self.dt.bb[:100]
      detections = 100
    if detections == 0:
      self.detected = False
      return
    print("Fern detector made %d detections in %f seconds." % (detections, time()-t))
    # Initialize detection structure
    for i in range(detections):
      patt_tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      isin_tmp = [-1, -1, -1]
      patch_tmp = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
      # Corresponding codes of the Ensemble Classifier
      self.dt.patt.append(patt_tmp)
      # Relative Similarity (for final nearest neighbour classifier)
      self.dt.conf1.append(0)
      # Conservative Similarity (for integration with tracker)
      self.dt.conf2.append(0)
      # Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
      self.dt.isin.append(isin_tmp)
      # Corresponding patches
      self.dt.patch.append(patch_tmp)
    nn_th = self.classifier.getNNTh()
    # for every remaining detection
    for i in range(detections):
      # Get the detected bounding box index
      idx = self.dt.bb[i]
      patch = mat_operator(frame, self.grid[idx])
      # Get pattern within bounding box
      self.dt.patch[i], mean, stdev = self.getPattern(patch)
      self.dt.isin[i], self.dt.conf1[i], self.dt.conf2[i] = self.classifier.NNConf(self.dt.patch[i])
      self.dt.patt[i] = self.tmp_patt[idx].tolist()
      if self.dt.conf1[i] > nn_th:
        self.dbb.append(self.grid[idx])
        self.dconf.append(self.dt.conf2[i])
    if len(self.dbb) > 0:
      print("Found %d NN matches" % len(self.dbb))
      self.detected = True
    else:
      print("No NN matches found.")
      self.detected = False

  def evaluate(self):
    pass

  def learn(self, img):
    # const Mat& img
    print("[Learning] ")
    # Check consistency
    bb = [0, 0, 0, 0]
    bb[0] = max(self.lastbox[0], 0)
    bb[1] = max(self.lastbox[1], 0)
    bb[2] = min(min(img.shape[1] - self.lastbox[0], self.lastbox[2]), min(self.lastbox[2], self.lastbox[0] + self.lastbox[2]))
    bb[3] = min(min(img.shape[0] - self.lastbox[1], self.lastbox[3]), min(self.lastbox[3], self.lastbox[1] + self.lastbox[3]))
    pattern, mean, stdev = self.getPattern(mat_operator(img, bb))
    isin, conf, dummy = self.classifier.NNConf(pattern)
    if conf < 0.5:
      print("Fast change.. not training")
      self.lastvalid = False
      return
    if pow(stdev, 2) < self.var:
      print("Low variance.. not training")
      self.lastvalid = False
      return
    if isin[2] == 1:
      print("Patch in negative data.. not training")
      self.lastvalid = False
      return
    # Data Generation
    for i in range(len(self.grid)):
      self.grid[i][4] = bbOverlap(self.lastbox, self.grid[i])
    # 集合分类器的样本: fern_examples
    fern_examples = []
    del self.good_boxes[:]
    del self.bad_boxes[:]
    self.getOverlappingBoxes(self.lastbox, self.num_closest_update)
    if len(self.good_boxes) > 0:
      self.generatePositiveData(img, self.num_warps_update)
    else:
      self.lastvalid = False
      print("No good boxes.. Not training")
      return
    for i in range(len(self.bad_boxes)):
      idx = self.bad_boxes[i]
      if self.tmp_conf[idx][0] >= 1:
        fern_examples.append(make_pair(self.tmp_patt[idx].tolist(), 0))
    nn_examples = [self.pEx]
    for i in range(len(self.dt.bb)):
      idx = self.dt.bb[i]
      if bbOverlap(self.lastbox, self.grid[idx]) < self.bad_overlap:
        nn_examples.append(self.dt.patch[i])
    # Classifier update
    self.classifier.trainF(fern_examples, 2)
    self.classifier.trainNN(nn_examples)
    self.classifier.show()

  def getPattern(self, img):
    """
      Arguments
      ---------
      img : const Mat&
      """
    # Output: Resized Zero-Mean patch

    pattern = cv2.resize(img, (self.patch_size, self.patch_size))
    mean, stdev = cv2.meanStdDev(pattern)
    mean = mean[0][0]
    stdev = stdev[0][0]
    pattern.astype(np.float32)
    pattern = pattern - np.ones((pattern.shape[0], pattern.shape[1])) * mean
    return pattern, mean, stdev

  def getOverlappingBoxes(self, box1, num_closest):
    # self.grid[i].shape = x, y, width, height, overlap, scale index
    max_overlap = 0.0
    for i in range(len(self.grid)):
      if self.grid[i][4] > max_overlap:
        max_overlap = self.grid[i][4]
        self.best_box = self.grid[i]
      if self.grid[i][4] > 0.6:
        self.good_boxes.append(i)
      elif self.grid[i][4] < self.bad_overlap:
        self.bad_boxes.append(i)

    # print("soft before: ", self.good_boxes)
    # Get the best num_closest (10) boxes and puts them in good_boxes
    if len(self.good_boxes) > num_closest:
      self.nth_element(self.good_boxes, sort_type="OComparator")
      self.good_boxes = self.good_boxes[-num_closest:]
    # print("soft after: ", self.good_boxes)
    self.getBBHull()

  def getBBHull(self):
    # 得到good_box窗口的最大边界
    x1 = math.inf
    x2 = 0
    y1 = math.inf
    y2 = 0
    for i in range(len(self.good_boxes)):
      idx = self.good_boxes[i]
      x1 = min(self.grid[idx][0], x1)
      y1 = min(self.grid[idx][1], y1)
      x2 = max(self.grid[idx][0] + self.grid[idx][2], x2)
      y2 = max(self.grid[idx][0] + self.grid[idx][3], y2)

    self.bbHull[0] = x1
    self.bbHull[1] = y1
    self.bbHull[2] = x2 - x1
    self.bbHull[3] = y2 - y1

  def buildGrid(self, img, box):
    # 第一步
    SHITF = 0.1
    SCALES = [0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                          0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                          2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174]
    sc = 0
    for i in range(len(SCALES)):
      scale = [0, 0]
      width = round(box[2] * SCALES[i])
      height = round(box[3] * SCALES[i])
      min_bb_side = min(height, width)
      # img = shape(height, width, channel)
      if min_bb_side < self.min_win or width > img.shape[1] or height > img.shape[0]:
        # 跳出本次循环
        continue
      scale[0] = width
      scale[1] = height
      self.scales.append(scale)
      y = 1
      while y < img.shape[0] - height:
        x = 1
        while x < img.shape[1] - width:
          # list.append()字典在for循环中数据覆盖
          bbox = [0, 0, 0, 0, 0, 0]
          bbox[0] = x
          bbox[1] = y
          bbox[2] = width
          bbox[3] = height
          bbox[4] = bbOverlap(bbox, box)
          # sc = scale index
          bbox[5] = sc
          x += round(SHITF * min_bb_side)
          self.grid.append(bbox)
        y += round(SHITF * min_bb_side)
      sc+=1

  @staticmethod
  def bbcomp(b1, b2):
    if bbOverlap(b1, b2) < 0.5:
      return False
    else:
      return True

  @staticmethod
  def clusterBB(dbb, indexes):
    # const vector<BoundingBox>& dbb,vector<int>& indexes
    # FIXME: Conditional jump or move depends on uninitialised value(s)
    c = len(dbb)
    # 1. Build proximity matrix
    D = np.zeros((c,c), dtype=np.float32)
    for i in range(c):
      j = i + 1
      while j < c:
        d = 1 - bbOverlap(dbb[i], dbb[j])
        D[i][j] = d
        D[j][i] = d
        j += 1
    # 2. Initialize disjoint clustering
    L = [0.0] * (c - 1)
    nodes = np.zeros((c-1, 2), dtype=np.uint8)
    belongs = [0] * c
    m = c
    for i in range(c):
      belongs[i] = i
    for it in range(c-1):
      # 3. Find Nearest Neighbor
      min_d = 1
      node_a = node_b = 0
      for i in range(D.shape[0]):
        for j in range(i+1,D.shape[1]):
          if D[i][j] < min_d and belongs[i] != belongs[j]:
            min_d = D[i][j]
            node_a = i
            node_b = j
      if min_d > 0.5:
        max_idx = 0
        for j in range(c):
          visited = False
          for i in range(2*c-1):
            if belongs[j] == i:
              indexes[j] = max_idx
              visited = True
          if visited:
            max_idx += 1
        return max_idx

      # Merge Clusters and assign level
      L[m] = min_d
      nodes[it][0] = belongs[node_a]
      nodes[it][1] = belongs[node_b]
      for k in range(c):
        if belongs[k] == belongs[node_a] or belongs[k] == belongs[node_b]:
          belongs[k] = m
      m += 1
    return 1

  @staticmethod
  def clusterConf(dbb, dconf):
    # const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf
    numbb = len(dbb)
    space_thr = 0.5
    c = 1
    if numbb == 1:
      cbb = [dbb[0]]
      cconf = [dconf[0]]
      return cbb, cconf
    elif numbb == 2:
      T = [0, 0]
      if 1 - bbOverlap(dbb[0],dbb[1]) > space_thr:
        T[1] = 1
        c = 2
    else:
      T = [0] * numbb
      # c = partition(dbb,T,(*bbcomp))
    cconf = [0] * c
    cbb = [[0, 0, 0, 0]] * c
    bx = [0, 0, 0, 0]
    for i in range(c):
      cnf = 0.0
      N = 0
      mx = 0
      my = 0
      mw = 0
      mh = 0
      for j in range(len(T)):
        if T[j] == i:
          print("Cluster index %d " % i)
          cnf = cnf + dconf[j]
          mx = mx + dbb[j][0]
          my = my + dbb[j][1]
          mw = mw + dbb[j][2]
          mh = mh + dbb[j][3]
          N += 1
      if N > 0:
        cconf[i] = cnf/N
        bx[0] = round(mx/N)
        bx[1] = round(my/N)
        bx[2] = round(mw/N)
        bx[3] = round(mh/N)
        cbb[i] = bx
    return cbb, cconf

  def nth_element(self, array, sort_type=None):
    self.quick_sort(array, 0, len(array) - 1, sort_type)

  def quick_sort(self, array, left, right, sort_type=None):
    if left < right:
      if sort_type == "OComparator":
        q = self.partition_OC(array, left, right)
        self.quick_sort(array, left, q - 1, sort_type)
        self.quick_sort(array, q + 1, right, sort_type)
      elif sort_type == "CComparator":
        q = self.partition_CC(array, left, right)
        self.quick_sort(array, left, q - 1)
        self.quick_sort(array, q + 1, right)

  def partition_OC(self, array, l, r):
    x = self.grid[array[r]][4]
    i = l - 1
    for j in range(l, r):
      if self.grid[array[j]][4] <= x:
        i += 1
        array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1

  @staticmethod
  def partition_CC(array, l, r):
    x = array[r]
    i = l - 1
    for j in range(l, r):
      if array[j] <= x:
        i += 1
        array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1



