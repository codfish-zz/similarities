# -*- coding: utf-8 -*-

import cv2, imagehash, os, sys
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints
def resize_image(image):
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, channel = image.shape

    maxD = 1024
    aspectRatio = width / height

    if aspectRatio < 1:
        newSize = (int(maxD * aspectRatio), maxD)
    else:
        newSize = (maxD, int(maxD / aspectRatio))
    image = cv2.resize(image, newSize)

    return image


def image_compare_hist(img1, img2):
    """
    Compare the similarity of two pictures using histogram (直方图)
    :param img1: img1 in MAT format (img1 = cv2.imread(image1))
    :param img2: img2 in MAT format (img2 = cv2.imread(image2))
    :return: the similarity of two pictures
    """

    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)

    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)

    similarity = cv2.compareHist(img1_hist, img2_hist, 0)

    return similarity


# Image must be a PIL instance
def image_compare_dHash(img1, img2):
    hash1 = imagehash.dhash(img1)
    hash2 = imagehash.dhash(img2)

    print("dHash1: %s, dHash2: %s" % (str(hash1), str(hash2)))

    # hamming distance
    ham_distance = abs(hash2 - hash1)

    return ham_distance


# Image must be a PIL instance
def image_compare_pHash(img1, img2):
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    print("pHash1: %s, pHash2: %s" % (str(hash1), str(hash2)))

    # hamming distance
    ham_distance = abs(hash2 - hash1)

    return ham_distance


# Image must be a PIL instance
def image_compare_pHashSimple(img1, img2):
    hash1 = imagehash.phash_simple(img1)
    hash2 = imagehash.phash_simple(img2)
    
    print("pHash1: %s, pHash2: %s" % (str(hash1), str(hash2)))

    # hamming distance
    ham_distance = abs(hash2 - hash1)

    return ham_distance


def image_compare(img_path_1, img_path_2):
    print("file_1:", get_file_name(img_path_1))
    print("file_2:", get_file_name(img_path_2))

    img_cv_1 = cv2.imread(img_path_1)
    img_cv_2 = cv2.imread(img_path_2)

    img_pil_1 = Image.open(img_path_1)
    img_pil_2 = Image.open(img_path_2)

    similarity = image_compare_hist(img_cv_1, img_cv_2)
    print("histogram similarity: %d" % similarity)

    ham_distance = image_compare_dHash(img_pil_1, img_pil_2)
    print("dHash ham_distance: %d" % ham_distance)

    ham_distance = image_compare_pHash(img_pil_1, img_pil_2)
    print("pHash ham_distance: %d" % ham_distance)

    print("\n")


def get_file_name(path):
    path = Path(path)
    return path.name


def img_path(filename):
    return os.path.join(BASE_DIR, "examples\data", filename)


def main():
    image_compare(img_path("ironman1.jpg"), img_path("ironman2.jpg"))


if __name__ == "__main__":
    main()
