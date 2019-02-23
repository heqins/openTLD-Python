#coding=utf-8
import os
import sys
import yaml
import TLD
from tld_utils import *
from time import time
class RunTld:
  tl = True
  def __init__(self, args):
    # get arguments
    self.img_path = args.s
    self.img_lists = self._get_images(self.img_path)
    self.img_lists.sort()
    self.parameter_file_path = args.p
    self._get_parameters(self.parameter_file_path)

  def _get_parameters(self, file_path):
    f = open(file_path)
    temp = yaml.load(f)
    # type(file)  = dict
    self.file = temp["Parameters"]

  @staticmethod
  def _get_images(path):
    # get images
    img_list = []
    if os.path.exists(path):
      dirs = os.listdir(path)
      for i in dirs:
        i = os.path.join(path,i)
        img_list.append(i)
    return img_list

  def startTLD(self):
    # get the image of the fist frame...(read as gray scale image)
    init_img = cv2.imread(self.img_lists[0])
    init_gray = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    # init_frame.dtype = uint8 -> float32
    last_gray = init_gray.astype(np.float32)
    # get the init bounding box..
    # init_bbox = tuple(x,y, w, h)
    init_bbox = cv2.selectROI("Select Initial Bounding Box", init_img, False, False)

    # Initialization
    print("Initial Bounding Box = x:%d y:%d w:%d h:%d" % (init_bbox[0], init_bbox[1], init_bbox[2], init_bbox[3]))
    if min(init_bbox[2], init_bbox[3]) < self.file["min_win"]:
      print("Bounding box too small, try again!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      sys.exit(0)

    # Output file
    bb_file = open("bounding_boxes.txt", "w")
    tld = TLD.TLD(self.file)
    # TLD initialization
    tld.init(last_gray, init_bbox, bb_file)

    # Run-time
    """
    Mat current_gray;
    BoundingBox pbox;
    vector<Point2f> pts1;
    vector<Point2f> pts2;
    """
    status = True
    frames = 1
    detections = 1
    pts1 = []
    pts2 = []
    pbox = [0, 0, 0, 0]
    count = 1
    # 进入一个循环,读入新的一帧,然后转换为灰度图像,然后再处理每一帧processFrame
    while count < len(self.img_lists):
      # get frame
      img = self.img_lists[count]
      count += 1
      frame = cv2.imread(img)
      current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Process Frame
      t = time()
      pts1, pts2, pbox, status = tld.processFrame(last_gray, current_gray, pts1, pts2, pbox, status, self.tl, bb_file)
      print("processFrame time: ", time()-t)
      # Draw points
      if status:
        drawPoints(frame, pts1, (0, 0, 0))
        # 当前的特征点用蓝色表示出来
        drawPoints(frame, pts2, (0, 255, 0))
        drawBox(frame, pbox, (0, 0, 0), 3)
        detections += 1
      # Display
      cv2.imshow("TLD", frame)
      # Swap points and images
      last_gray, current_gray = swap(last_gray ,current_gray)
      pts1 = []
      pts2 = []
      # pts1.clear()
      # pts2.clear()
      frames += 1
      print("Detection rate: %d/%d" % (detections, frames))
      if cv2.waitKey(33) == 'q':
        break




