import numpy as np
class Feature:
  x1 = 0
  y1 = 0
  x2 = 0
  y2 = 0

  def __init__(self, *args):
    self.x1 = int(args[0])
    self.y1 = int(args[1])
    self.x2 = int(args[2])
    self.y2 = int(args[3])

  def __call__(self, *args):
    patch = args[0]
    return int(patch[self.y1][self.x1] > patch[self.y2][self.x2])



