from skimage import io
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from PIL import Image, ImageOps
from icon_classify import classify_img


def bfs(img, r, c):
    img[r, c] = -1
    q = deque([(r, c)])
    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    top = bottom = r
    left = right = c
    while q:
        cur_r, cur_c = q.popleft()
        for d_r, d_c in direction:
            n_r = cur_r + d_r
            n_c = cur_c + d_c
            if 0 <= n_r < img.shape[0] and 0 <= n_c < img.shape[1] and 1.0 > img[n_r, n_c] != -1:
                img[n_r, n_c] = -1
                # Update bounding box
                if top > n_r:
                    top = n_r
                if bottom < n_r:
                    bottom = n_r
                if left > n_c:
                    left = n_c
                if right < n_c:
                    right = n_c
                q.append((n_r, n_c))
    return top, left, bottom, right


def extract_bbox(img):
    ret = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] < 1.0 and img[r, c] != -1.0:
                top, left, bottom, right = bfs(img, r, c)
                ret.append((top, left, bottom, right))
    return ret


def classify_all_entity(img_path, min_pixels=100):
    ret = []
    img = io.imread(img_path)
    img = img[3:-3, 3:-3]
    img[np.logical_and(img[:, :, 0] == 178, img[:, :, 1] ==
                       235, img[:, :, 2] == 255)] = [255, 255, 255]
    gimg = rgb2gray(img)
    # Set almost white pixel to white
    gimg[gimg > 0.85] = 1.0

    bboxs = extract_bbox(gimg)
    bboxs = [b for b in bboxs if abs(b[2]-b[0]+1)*abs(b[3]-b[1]) >= min_pixels]
    # pis = []
    for b in bboxs:
        top, left, bottom, right = b
        pi = Image.fromarray(img[top:bottom+1, left:right+1])
        entity_class = classify_img(pi)
        ret.append(entity_class)
        # pis.append(pi)
    return ret
