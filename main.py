import matplotlib.pyplot as plt
import time
import cv2
import numpy as np


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


start = time.time()
muhur = cv2.imread("muhur.png")
pusula = cv2.imread("./gercek_pusula/gercek5.jpeg")

muhur = cv2.cvtColor(muhur, cv2.COLOR_RGBA2GRAY)
pusula_g = cv2.cvtColor(pusula, cv2.COLOR_RGBA2GRAY)
resy, resx = pusula.shape[0], pusula.shape[1]

_, pusula = cv2.threshold(pusula_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
pusula = cv2.morphologyEx(pusula, cv2.MORPH_CLOSE, np.ones((10, 10)))

contours, hierarchy = cv2.findContours(pusula, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = [contour for contour in contours if cv2.contourArea(contour) > 10000]

aday1 = None
aday2 = None
x1 = x2 = y1 = y2 = w1 = w2 = h1 = h2 = None
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if x + w / 2 <= resx / 2 and resx * 0.2 <= w <= resx * 0.6 and resy * 0.5 <= h <= resy:
        aday1 = x1, y1, w1, h1 = x, y, w, h
        break

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if x + w / 2 > resx / 2 and resx * 0.2 <= w <= resx * 0.6 and resy * 0.5 <= h <= resy:
        aday2 = x2, y2, w2, h2 = x, y, w, h
        break

muhurx, muhury = muhur.shape[1], muhur.shape[0]
muhur_new_y = aday1[3] * 0.29
muhur_new = round(muhur_new_y * muhurx / muhury), round(muhur_new_y)
muhur = cv2.resize(muhur, muhur_new, interpolation=cv2.INTER_LINEAR)
_, muhur = cv2.threshold(muhur, 180, 255, cv2.THRESH_BINARY_INV)

degrees = []
locs = []
vals = []
sizes = []
oy1 = False
oy2 = False

found = False
for degree in range(0, 360, 15):
    muhur_rot = rotate(muhur, degree)
    y_nonzero, x_nonzero = np.nonzero(muhur_rot)
    muhur_rot = muhur_rot[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
    muhur_rotx, muhur_roty = muhur_rot.shape[1], muhur_rot.shape[0]

    res = cv2.matchTemplate(pusula, muhur_rot, cv2.TM_CCOEFF_NORMED)
    _, maxval, _, maxloc = cv2.minMaxLoc(res)
    size = muhur_roty, muhur_rotx
    in1 = x1 <= maxloc[0] + muhur_rotx / 2 <= x1 + w1 and y1 <= maxloc[1] + muhur_roty / 2 <= y1 + h1
    in2 = x2 <= maxloc[0] + muhur_rotx / 2 <= x2 + w2 and y2 <= maxloc[1] + muhur_roty / 2 <= y2 + h2
    locs.append(maxloc)
    vals.append(maxval)
    sizes.append((muhur_roty, muhur_rotx))
    if in1 and maxval > 0.7:
        oy1 = True
    if in2 and maxval > 0.7:
        oy2 = True
    elif maxval > 0.7:
        found = True

if oy1 is not oy2:
    if oy1:
        print("Aday 1")
    elif oy2:
        print("Aday 2")
else:
    print("Geçersiz!")

print("Time Elapsed: {:.2f}s".format(time.time() - start))

if oy1 or oy2 or found:
    idx = vals.index(max(vals))
    loc = locs[idx]
    size = sizes[idx]
    cv2.rectangle(pusula_g, loc, (loc[0] + size[1], loc[1] + size[0]), 10, 10)
    plt.imshow(pusula_g, cmap="gray")
    plt.show()
