import numpy as np
import random as rand
import math
import cv2

def saturate_cast(num):
  if num < 0:
    return 0
  elif num > 255:
    return 255
  else:
    return num

class PatchGenerator:
  backgroundMin = 0
  backgroundMax = 256
  noiseRange = 5
  lambdaMin = 0.6
  lambdaMax = 1.5
  thetaMin = -np.pi
  thetaMax= np.pi
  phiMin = -np.pi
  phiMax = np.pi

  def __init__(self, background_min, background_max, noise_range, random_blur,
               lambda_min = 0.6, lambda_max = 1.5, theta_min = -np.pi, theta_max = np.pi, phi_min = -np.pi, phi_max = np.pi):
    self.backgroundMin = background_min
    self.backgroundMax = background_max
    self.noiseRange = noise_range
    self.randomBlur = random_blur
    self.lambdaMin = lambda_min
    self.lambdaMax = lambda_max
    self.thetaMin = theta_min
    self.thetaMax = theta_max
    self.phiMin = phi_min
    self.phiMax = phi_max

  def __call__(self, *args):
    image = args[0]
    pt = args[1]
    patch = args[2]
    patchSize = args[3]
    if len(args) == 4:
      self.operator1(image, pt, patch, patchSize)
    elif len(args) == 5:
      self.operator2(image, pt, patch, patchSize, "operator2")

  def operator1(self, image, pt, patch, patchSize):
    # void PatchGenerator::operator ()(const Mat& image, Point2f pt, Mat& patch, Size patchSize, RNG& rng) const
    T = np.zeros((2, 3), np.float64)
    self.generateRandomTransform(pt, [(patchSize[0]-1) * 0.5, (patchSize[1]-1) * 0.5], T)
    self.operator2(image, T, patch, patchSize, "operator2")

  def operator2(self, image, T, patch, patchSize, operator):
    # void PatchGenerator::operator ()(const Mat& image, const Mat& T,  Mat& patch, Size patchSize, RNG& rng) const
    # patchSize = [self.bbHull[2], self.bbHull[3]]
    patch = np.ones(patchSize, dtype=image.dtype)
    if self.backgroundMin != self.backgroundMax:
      patch = np.random.uniform(self.backgroundMin, self.backgroundMax, patchSize)
      cv2.warpAffine(src=image, dst=patch, M=T, dsize=patchSize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    else:
      cv2.warpAffine(src=image, dst=patch, M=T, dsize=patchSize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # warpAffine(image, patch, T, patchSize, INTER_LINEAR, BORDER_CONSTANT, Scalar::all(backgroundMin));
    ksize = rand.randint(0, 1) * 255 % 9 - 5 if self.randomBlur else 0
    if ksize > 0:
      ksize = ksize * 2 + 1
      cv2.GaussianBlur(patch, ksize=(ksize, ksize), sigmaX=0, sigmaY=0, dst=patch)

    if self.noiseRange > 0:
      # rng.fill(noise, RNG::NORMAL, Scalar::all(delta), Scalar::all(noiseRange));
      noise = np.random.normal(size=(patchSize[0], patchSize[1]))
      if image.dtype == np.uint8:
        delta = 128
      elif image.dtype == np.float32:
        delta = 32768
      else:
        delta = 0
      noise = np.clip(noise, delta, self.noiseRange)
      if self.backgroundMin != self.backgroundMax:
        cv2.addWeighted(patch, 1, noise, 1, -delta, patch)
      else:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(patchSize[1]):
          prow = patch[i]
          nrow = noise[i]
          for j in range(patchSize[0]):
            if prow[j] != self.backgroundMin:
              prow[j] = saturate_cast(prow[j] + nrow[j] - delta)

  def generateRandomTransform(self, srcCenter, dstCenter, transofrm, inverse=False):
    # C++: Point2f srcCenter, Point2f dstCenter, Mat& transform, Rng& rng, bool inverse
    lambda1 = rand.uniform(self.lambdaMin, self.lambdaMax)
    lambda2 = rand.uniform(self.lambdaMin, self.lambdaMax)
    theta = rand.uniform(self.thetaMin, self.thetaMax)
    phi = rand.uniform(self.phiMin, self.phiMax)

    # Calculate random parameterized affine transformation A,
    # A = T(patch center) * R(theta) * R(phi) * S(lambda1, lambda2) * R(phi) * T(-pt)
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)
    c2p = cp * cp
    s2p = sp * sp

    A = lambda1 * c2p + lambda2 * s2p
    B = (lambda2 - lambda1) * sp * sp
    C = lambda1 * s2p + lambda2 * c2p

    Ax_plus_By = A * srcCenter[0] + B * srcCenter[1]
    Bx_plus_Cy = B * srcCenter[0] + C * srcCenter[1]

    transform = np.ones((2, 3), np.float64)
    # Mat_<double>& T = (Mat_<double>&) transform;
    T = transform
    T[0][0] = A * ct - B * st
    T[0][1] = B * ct - C * st
    T[0][2] = -ct * Ax_plus_By + st * Bx_plus_Cy + dstCenter[0]
    T[1][0] = A * st + B * ct
    T[1][1] = B * st + C * ct
    T[1][2] = -st * Ax_plus_By - ct * Bx_plus_Cy + dstCenter[1]

    if inverse:
      cv2.invertAffineTransform(M=T, iM=T)

  # def _warpWholeImage(self, image, matT, buf, warped, border):
  #   # const Mat& image, Mat& matT, Mat& buf, Mat& warped, int border
  #   T = np.ones((matT.shape[0], matT.shape[1]), np.float64)
  #   # roi type = Rect
  #   roi = [sys.maxsize, sys.maxsize, -sys.maxsize-1, -sys.maxsize-1]
  #   for k in range(4):
  #     # pt0, pt1 -> Point2f(x, y)
  #     pt0 = [0.0, 0.0]
  #     pt1 = [0.0, 0.0]
  #     pt0[0] = float(0 if k == 0 | k == 3 else image.shape[1])
  #     pt0[1] = float(0 if k < 2 else image.shape[0])
  #     pt1[0] = float(T[0][0] * pt0[0] + T[0][1] * pt0[1] + T[0][2])
  #     pt1[1] = float(T[1][0] * pt0[0] + T[1][1] * pt0[1] + T[1][2])
  #
  #     roi[0] = min(roi[0], math.floor(pt1[0]))
  #     roi[1] = min(roi[1], math.floor(pt1[1]))
  #     roi[2] = max(roi[2], math.ceil(pt1[0]))
  #     roi[3] = max(roi[3], math.ceil(pt1[1]))
  #
  #   roi[2] -= roi[0] - 1
  #   roi[3] -= roi[1] - 1
  #   dx = border - roi[0]
  #   dy = border - roi[1]
  #
  #   if (roi[2] + border * 2) * (roi[3] + border * 2) > buf.shape[1]:
  #     buf = np.ones((1, (roi[2] + border * 2) * (roi[3] + border * 2)), image.dtype)
  #
  #   warped = np.ones()
  #   T[0][2] += dx
  #   T[1][2] += dy

  def _setAffineParam(self, lambda_, theta, phi):
    self.lambdaMin = 1.0 - lambda_
    self.lambdaMax = 1.0 + lambda_
    self.thetaMin = -theta
    self.thetaMax = theta
    self.phiMin = -phi
    self.phiMax = phi



