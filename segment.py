from skimage import io
import numpy as np
from collections import deque


def bfs(img, r, c):
    img[r, c] = -1
    q = deque([(r, c)])
    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    top = left = right = bottom = None
    while q:
        cur_r, cur_c = q.popleft()
        for d_r, d_c in direction:
            n_r = cur_r + d_r
            n_c = cur_c + d_c
            if 0 <= n_r < img.shape[0] and 0 <= n_c < img.shape[1] and 1.0 > img[n_r, n_c] != -1:
                img[n_r, n_c] = -1
                # Update bounding box
                if top is None or top > n_r:
                    top = n_r
                if bottom is None or bottom < n_r:
                    bottom = n_r
                if left is None or left > n_c:
                    left = n_c
                if right is None or right < n_c:
                    right = n_c
                q.append((n_r, n_c))
    return top, left, bottom, right


def count_region(img):
    count = 0
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] < 1.0 and img[r, c] != -1.0:
                count += 1
                bfs(img, r, c)
    return count


def extract_icon(img):
    ret = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] < 1.0 and img[r, c] != -1.0:
                top, left, bottom, right = bfs(img, r, c)
                ret.append(img[top:bottom+1, left:right+1])
    return ret


def count_region_of_img(img_path):
    img = io.imread(img_path, as_gray=True)
    t = np.copy(img)[10:-10, 10:-10]
    return count_region(t)
