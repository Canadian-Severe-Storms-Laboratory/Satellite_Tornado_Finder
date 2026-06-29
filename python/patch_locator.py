import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


def find_patches_ncc(big_img, patches, method=cv2.TM_CCOEFF_NORMED, threshold=0.99):
    big = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    locs = []
    for p in tqdm(patches):
        templ = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
        res = cv2.matchTemplate(big, templ, method)   # valid correlation map
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            locs.append(max_loc)  # (x, y) = (col, row) top-left
        else:
            locs.append(None)
    return locs


if __name__ == '__main__':
    path = r"/mnt/c/Users/danie/Documents/Experiments/Satellite/saved/Angliers_2022"
    img = cv2.imread(path + r"/Angliers_2022_after.png")
    #img = cv2.medianBlur(img, 5)

    patches = [cv2.imread(f) for f in glob(path + r"\after\*.png")]

    results = find_patches_ncc(img, patches)

    for i in range(len(results)):
        x, y = results[i]

        cv2.imshow("patch", cv2.resize(patches[i], (512, 512)))
        cv2.imshow("result", cv2.resize(img[y:y+64, x:x+64], (512, 512)))

        cv2.waitKey(0)
