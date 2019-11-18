import cv2
import math
import debug
import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def union_rectanagles(rect_list):
    min_x, min_y = math.inf, math.inf
    max_x, max_y = -1, -1

    # rect_list - non empty list of ROIs in (x, y, w, h) format
    assert len(rect_list) != 0

    for rect in rect_list:
        left = rect[0]
        right = rect[0] + rect[2]
        top = rect[1]
        bottom = rect[1] + rect[3]

        if left < min_x:
            min_x = left
        if right > max_x:
            max_x = right

        if top < min_y:
            min_y = top
        if bottom > max_y:
            max_y = bottom

    # transform the resulting rectangle in (x, y, w, h) format
    return (min_x, min_y, max_x - min_x, max_y - min_y)




def get_contour_center(contour):
    M = cv2.moments(contour)
    c_x = int(M["m10"] / M["m00"])
    c_y = int(M["m01"] / M["m00"])
    return (c_x, c_y)


def translate_points(points, translate_factor):
    translated_points = np.asarray(points) + np.asarray(translate_factor)
    return translated_points


def corp_mask_contour(image):

    shape_img = image.shape
    if len(shape_img) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = np.copy(image)
    _, image_thr = cv2.threshold(image_gray, 127, 255, 0)
    _, contours, _ = cv2.findContours(image_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if debug.DEBUG_MASK >=3:
        display_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    roi_list = []
    for cnt in contours:
        r = cv2.boundingRect(cnt)
        roi_list.append(r)
        if debug.DEBUG_DISPLAY_HAIRSHAPE >= 3:
            cv2.rectangle(display_image, (r[0], r[1]),
                      (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 2)

    x, y, w, h = union_rectanagles(roi_list)
    if debug.DEBUG_DISPLAY_HAIRSHAPE >= 3:
        cv2.rectangle(display_image, (x, y),
                      (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('contours', display_image)

    roi_img = image[y:y+h, x:x+w]
    return roi_img


def align_images_width(img1, img2, align_width = 200):

    # 0. get only the roi from conour image
    img1 = corp_mask_contour(np.copy(img1))

    hi, wi = img1.shape
    hm, wm = img2.shape
    # aspect ratio for mask and image
    ar_i = wi/hi
    ar_m = wm/hm

    scale_factor_i = align_width / wi
    scale_factor_m = align_width / wm


    # align masks
    # 1. scale them by width
    img_resized = cv2.resize(img1, None, fx=scale_factor_i, fy=scale_factor_i)

    # we will keep the masks
    if wm != align_width:
        mask_resized = cv2.resize(img2, None, fx = scale_factor_m, fy = scale_factor_m)
    else:
        mask_resized = img2

    # 3. clamp the images vertically so that they have the same size
    hi, wi = img_resized.shape
    hm, wm = mask_resized.shape
    # print('image size: ', wi, 'x', hi)
    # print('mask size: ', wm, 'x', hm)

    if hm != hi:
        if hm > hi:
            diff = hm - hi
            # copy make bother top, bottom, left, right
            img_resized = cv2.copyMakeBorder(img_resized, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            diff = hi - hm
            mask_resized = cv2.copyMakeBorder(mask_resized, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

    return img_resized, mask_resized, ar_i, ar_m


def compute_mask_iou(mask1, mask2):

    assert mask1.shape == mask2.shape
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    if debug.DEBUG_MASK >= 2:
        union_intersection_display = np.concatenate((union, intersection), axis=1)
        cv2.imshow('union/intersection', union_intersection_display)


    num_pix_intersection = cv2.countNonZero(intersection)
    num_pix_union = cv2.countNonZero(union)

    return float(num_pix_intersection)/float(num_pix_union)



