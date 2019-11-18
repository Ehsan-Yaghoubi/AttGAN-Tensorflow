import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt


def X_Y_coordinate_evaluation (X_coordinate, Y_coordinate, image_name, img, resize_scale):
    x_prime, y_prime = 0, 0
    _flag=True
    H_prime = resize_scale[0]
    W_prime = resize_scale[1]
    _h, _w, _z = img.shape
    #y.append(float(Y_coordinate))
    if (_h/_w)>=1.75:
        y_prime = (float(Y_coordinate) / _h) * H_prime
        try:
            # z_prime = (((_h - _w) / _h) * 100) / 2
            w_prime = (_w / _h) * H_prime
            OneSidePad = abs((w_prime - 100) / 2)
            x_prime = ((float(X_coordinate) / _w) * w_prime) + OneSidePad
            # x.append(float(X_coordinate))

        except ValueError:
            print("value error", image_name)
            _flag = False
    else:
        x_prime = (float(X_coordinate) / _w) * W_prime
        try:
            H_prime = (_h / _w) * W_prime
            OneSidePad = abs((H_prime - 175) / 2)
            y_prime = ((float(Y_coordinate) / _h) * H_prime) + OneSidePad

        except ValueError:
            print("value error", image_name)
            _flag = False

    return  _flag, x_prime, y_prime

def resizeAndPad(img, size, padColor):

    h, w, channel = img.shape
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    if h==0:
        print("height of the image is =0")
    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > (sw/sh): # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < (sw/sh): # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    #cv2.imshow("_cropped_img", scaled_img)
    #cv2.waitKey(0)
    return scaled_img

# input folders and files
RAPSRC = "/media/ehsan/48BE4782BE476810/MyPythonCodes/RAP/RAP_dataset/RAP_images/"
json_string = "/media/ehsan/48BE4782BE476810/MyPythonCodes/RAP/RAP_dataset/RAP_Annotations_Keypoints/alpha-pose-results.json"
# output folders
RAP175x100 = "/media/ehsan/48BE4782BE476810/MyPythonCodes/RAP/RAP_dataset/RAP_Annotations_Keypoints/images_175x100/"
os.makedirs(RAP175x100, exist_ok=True)
resize_scale = (175, 100)
with open(json_string) as f:
    data = json.load(f)

images_without_skeleton = []
#RAPlist = os.listdir(RAPSRC)
for i, file in enumerate(data):
    X = []
    Y = []
    mapped_Y = []
    mapped_X = []
    #root = file.split(".")[0]
    read_img = cv2.imread(RAPSRC + file["image_id"])
    Resized_img = resizeAndPad(img=read_img, size=resize_scale, padColor=255)
    try:
        img_keypoints = file['keypoints']
        print(img_keypoints)
        for indx, digit in enumerate(img_keypoints):
            if indx % 3 == 0:
                X.append(img_keypoints[indx])
                Y.append(img_keypoints[indx + 1])
        for q in range(0,15):
            flag, x_prime, y_prime = X_Y_coordinate_evaluation(X[q], Y[q], file["image_id"], read_img, resize_scale)
            mapped_Y.append(y_prime)
            mapped_X.append(x_prime)
            if not flag:
                continue

        plt.scatter(x=mapped_X, y=mapped_Y)
        plt.imshow(Resized_img)
        plt.waitforbuttonpress()
        plt.clf()
        plt.gcf()
        X = []
        Y = []
    except KeyError:
        images_without_skeleton.append(file["image_id"])

    # save re-sized image
    cv2.imwrite(RAP175x100 + file["image_id"], Resized_img)
    if i % 1000 == 0:
        print(">> resize TEST set {}/{}".format(i, len(data)))


    ## Crop head


    ## Crop Polygon


print("Number of images_without_skeleton: ", len(images_without_skeleton))



