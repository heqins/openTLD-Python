from tld_utils import *
class LKTracker:
  simmed = 0.0
  fbmed = 0.0
  lk_params = dict(winSize=(4, 4),
                   maxLevel=5,
                   criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 20, 0.03),
                   flags=0)
  def getFB(self):
    return self.fbmed[0]

  def trackf2f(self, img1, img2, points1):
    points1 = np.asarray(points1, dtype=np.float32)
    img1 = img1.astype(np.uint8)
    # Forward-Backward tracking
    points2, status, similarity = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, **self.lk_params)
    pointsFB, FB_status, FB_error = cv2.calcOpticalFlowPyrLK(img2, img1, points2, None, **self.lk_params)
    # Compute the real FB-error
    for i in range(len(points1)):
      FB_error[i] = cv2.norm(pointsFB[i] - points1[i])
    # Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
    self.normCrossCorrelation(img1, img2, points1, points2, status, similarity)
    return self.filterPts(points1, points2, status, similarity, FB_error)

  @staticmethod
  def normCrossCorrelation(img1, img2, points1, points2 ,status, similarity):
    # const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2
    rec0 = np.zeros((10, 10), dtype=np.uint8)
    rec1 = np.zeros((10, 10), dtype=np.uint8)
    res = np.zeros((1,1), dtype=np.float32)
    for i in range(len(points1)):
      if status[i][0] == 1:
        cv2.getRectSubPix(img1, (10, 10), (points1[i][0], points1[i][1]), rec0)
        cv2.getRectSubPix(img2, (10, 10), (points2[i][0], points2[i][1]), rec1)
        cv2.matchTemplate(rec0, rec1, result=res, method=cv2.TM_CCORR_NORMED)
        similarity[i] = float(res[0][0])
      else:
        similarity[i] = 0.0

    # rec0.release()
    # rec1.release()
    # res.release()

  def filterPts(self, points1, points2, status, similarity, FB_error):
    # 筛选出特征点
    # Get Error Median
    self.simmed = median(similarity)
    k = 0
    for i in range(len(points2)):
      if not(status[i]):
        continue
      if similarity[i] > self.simmed:
        points1[k] = points1[i]
        points2[k] = points2[i]
        FB_error[k] = FB_error[i]
        k += 1
    if k == 0:
      return False, 0, 0
    points1 = points1[:k]
    points2 = points2[:k]
    FB_error = FB_error[:k]
    self.fbmed = median(FB_error)
    k = 0
    for i in range(len(points2)):
      if not(status[i]):
        continue
      if FB_error[i] <= self.fbmed:
        points1[k] = points1[i]
        points2[k] = points2[i]
        k += 1
    points1 = points1[:k]
    points2 = points2[:k]
    if k > 0:
      return True, points1, points2
    else:
      return False, 0, 0