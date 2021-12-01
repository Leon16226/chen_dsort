from camera.utils import create_cameras
import numpy as np

def fun1(x):
   x[0] = 1111111

if __name__ == '__main__':

   s = [np.random.random([5, 1]), 2, 3, 4]

   fun1(s[0])

   print(s)