import cv2
from numpy.random import randint
import os
import numpy as np


def muhur_bas(bg, img, x, y):
    height, width = img.shape[0], img.shape[1]
    alphas = img[:, :, 3:] / 255
    bg[y:y + height, x:x + width] = (1 - alphas) * bg[y:y + height, x:x + width] + alphas * img
    return bg


# noinspection DuplicatedCode
def rotate(mat, angle):
    height, width = mat.shape[:2]
    center = (width / 2, height / 2)

    radx = np.radians(angle)
    sinx, cosx = np.sin(radx), np.cos(radx)
    trans_mat = np.array([[cosx, -sinx, center[0] * (1 - cosx) + center[1] * sinx],
                          [sinx, cosx, center[1] * (1 - cosx) - center[0] * sinx]])

    newx = int((height * abs(sinx)) + (width * abs(cosx)))
    newy = int((height * abs(cosx)) + (width * abs(sinx)))

    trans_mat[0, 2] += ((newx / 2) - center[0])
    trans_mat[1, 2] += ((newy / 2) - center[1])

    rotated_mat = cv2.warpAffine(mat, trans_mat, (newx, newy), borderValue=0)
    return rotated_mat


muhur = cv2.imread("muhur.png")
pusula = cv2.imread("pusula.png")

muhur = cv2.cvtColor(muhur, cv2.COLOR_RGB2GRAY)
pusula = cv2.cvtColor(pusula, cv2.COLOR_RGB2GRAY)

_, muhur = cv2.threshold(muhur, 180, 255, cv2.THRESH_BINARY)
_, pusula = cv2.threshold(pusula, 180, 255, cv2.THRESH_BINARY)

pusula = cv2.merge((pusula.copy(), pusula.copy(), pusula.copy()))
pusula = cv2.cvtColor(pusula, cv2.COLOR_RGB2RGBA)

muhur = cv2.merge((muhur.copy(), muhur.copy(), muhur.copy(), 255 - muhur.copy()))
xp, yp = pusula.shape[1], pusula.shape[0]

dirx = os.getcwd() + "/Pusulalar"

try:
    os.mkdir(dirx)
except OSError:
    print("Klasör Bulundu.")

for i in range(5):
    muhurx = rotate(muhur, randint(0, 360))
    pusulax = pusula.copy()
    xm, ym = muhurx.shape[1], muhurx.shape[0]
    name = "./Pusulalar/pusula" + str(i) + ".png"
    basili = muhur_bas(pusulax, muhurx, randint(0, high=xp - xm), randint(0, high=yp - ym))
    cv2.imwrite(name, basili)
