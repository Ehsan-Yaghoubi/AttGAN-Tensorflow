import os
import cv2
import json
import mat4py
import random
import numpy as np

# paths
rap_images_dir = '/work/image-databases/RAP_data/RAP_images'
rap_masks_dir = '/work/image-databases/RAP_data/RAP_masks'
rap_keypoints_json = '/work/image-databases/RAP_data/rap_annotations/RAP_keypoints.json'
rap_attribute_annotations = '/work/image-databases/RAP_data/rap_annotations/RAP_annotation.mat'

# rap keypoints names
kp_left_ankle = 0
kp_left_knee = 1
kp_left_hip = 2
kp_right_hip = 3
kp_right_knee = 4
kp_right_ankle = 5
kp_pelvis = 6
kp_abdomen = 7
kp_neck = 8
kp_head = 9
kp_left_wrist = 10
kp_left_elbow = 11
kp_left_shoulder = 12
kp_right_shoulder = 13
kp_right_elbow = 14
kp_right_wrist = 15
# end rap keypoint names

# rap attibute names
attr_Female = 0
attr_AgeLess16 = 1
attr_Age17_30 = 2
attr_Age31_45 = 3
attr_Age46_60 = 4
attr_AgeBigger60 = 5
attr_BodyFatter = 6
attr_BodyFat = 7
attr_BodyNormal = 8
attr_BodyThin = 9
attr_BodyThiner = 10
attr_Customer = 11
attr_Employee = 12
attr_hs_BaldHead = 13
attr_hs_LongHair = 14
attr_hs_BlackHair = 15
attr_hs_Hat = 16
attr_hs_Glasses = 17
attr_hs_Sunglasses = 18
attr_hs_Muffler = 19
attr_hs_Mask = 20
attr_ub_Shirt = 21
attr_ub_Sweater = 22
attr_ub_Vest = 23
attr_ub_TShirt = 24
attr_ub_Cotton = 25
attr_ub_Jacket = 26
attr_ub_SuitUp = 27
attr_ub_Tight = 28
attr_ub_ShortSleeve = 29
attr_ub_Others = 30
attr_ub_ColorBlack = 31
attr_ub_ColorWhite = 32
attr_ub_ColorGray = 33
attr_up_ColorRed = 34
attr_ub_ColorGreen = 35
attr_ub_ColorBlue = 36
attr_ub_ColorSilver = 37
attr_ub_ColorYellow = 38
attr_ub_ColorBrown = 39
attr_ub_ColorPurple = 40
attr_ub_ColorPink = 41
attr_ub_ColorOrange = 42
attr_ub_ColorMixture = 43
attr_ub_ColorOther = 44
attr_lb_LongTrousers = 45
attr_lb_Shorts = 46
attr_lb_Skirt = 47
attr_lb_ShortSkirt = 48
attr_lb_LongSkirt = 49
attr_lb_Dress = 50
attr_lb_Jeans = 51
attr_lb_TightTrousers = 52
attr_lb_ColorBlack = 53
attr_lb_ColorWhite = 54
attr_lb_ColorGray = 55
attr_lb_ColorRed = 56
attr_lb_ColorGreen = 57
attr_lb_ColorBlue = 58
attr_lb_ColorSilver = 59
attr_lb_ColorYellow = 60
attr_lb_ColorBrown = 61
attr_lb_ColorPurple = 62
attr_lb_ColorPink = 63
attr_lb_ColorOrange = 64
attr_lb_ColorMixture = 65
attr_lb_ColorOther = 66
attr_shoes_Leather = 67
attr_shoes_Sports = 68
attr_shoes_Boots = 69
attr_shoes_Cloth = 70
attr_shoes_Sandals = 71
attr_shoes_Casual = 72
attr_shoes_Other = 73
attr_shoes_ColorBlack = 74
attr_shoes_ColorWhite = 75
attr_shoes_ColorGray = 76
attr_shoes_ColorRed = 77
attr_shoes_ColorGreen = 78
attr_shoes_ColorBlue = 79
attr_shoes_ColorSilver = 80
attr_shoes_ColorYellow = 81
attr_shoes_ColorBrown = 82
attr_shoes_ColorPurple = 83
attr_shoes_ColorPink = 84
attr_shoes_ColorOrange = 85
attr_shoes_ColorMixture = 86
attr_shoes_ColorOther = 87
attr_attachment_Backpack = 88
attr_attachment_ShoulderBag = 89
attr_attachment_HandBag = 90
attr_attachment_WaistBag = 91
attr_attachment_Box = 92
attr_attachment_PlasticBag = 93
attr_attachment_PaperBag = 94
attr_attachment_HandTrunk = 95
attr_attachment_Baby = 96
attr_attachment_Other = 97
attr_action_Calling = 98
attr_action_StrechOutArm = 99
attr_action_Talking = 100
attr_action_Gathering = 101
attr_action_LyingCounter = 102
attr_action_Squatting = 103
attr_action_Running = 104
attr_action_Holding = 105
attr_action_Pushing = 106
attr_action_Pulling = 107
attr_action_CarryingByArm = 108
attr_action_CarryingByHand = 109
attr_action_Other = 110
attr_viewpoint = 111
attr_OcclusionLeft = 112
attr_OcclusionRight = 113
attr_OcclusionUp = 114
attr_OcclusionDown = 115
attr_occlustion_TypeEnvironment = 116
attr_occlustion_TypeAttachment = 117
attr_occlustion_TypePerson = 118
attr_occlustion_TypeOther = 119
attr_person_position_x = 120
attr_person_position_y = 121
attr_person_position_w = 122
attr_person_position_h = 123
attr_headshoulder_position_x = 124
attr_headshoulder_position_y = 125
attr_headshoulder_position_w = 126
attr_headshoulder_position_h = 127
attr_upperbody_position_x = 128
attr_upperbody_position_y = 129
attr_upperbody_position_w = 130
attr_upperbody_position_h = 131
attr_lowerbody_position_x = 132
attr_lowerbody_position_y = 133
attr_lowerbody_position_w = 134
attr_lowerbody_position_h = 135
attr_attachment1_position_x = 136
attr_attachment1_position_y = 137
attr_attachment1_position_w = 138
attr_attachment1_position_h = 139
attr_attachment2_position_x = 140
attr_attachment2_position_y = 141
attr_attachment2_position_w = 142
attr_attachment2_position_h = 143
attr_attachment3_position_x = 144
attr_attachment3_position_y = 145
attr_attachment3_position_w = 146
attr_attachment3_position_h = 147
attr_attachment4_position_x = 148
attr_attachment4_position_y = 149
attr_attachment4_position_w = 150
attr_attachment4_position_h = 151
# end rap attribute names

rap_attibute_names = {"attr_Female" : 0,
                    "attr_AgeLess16" : 1,
                    "attr_Age17_30" : 2,
                    "attr_Age31_45" : 3,
                    "attr_Age46_60" : 4,
                    "attr_AgeBigger60" : 5,
                    "attr_BodyFatter" : 6,
                    "attr_BodyFat" : 7,
                    "attr_BodyNormal" : 8,
                    "attr_BodyThin" : 9,
                    "attr_BodyThiner" : 10,
                    "attr_Customer" : 11,
                    "attr_Employee" : 12,
                    "attr_hs_BaldHead" : 13,
                    "attr_hs_LongHair" : 14,
                    "attr_hs_BlackHair" : 15,
                    "attr_hs_Hat" : 16,
                    "attr_hs_Glasses" : 17,
                    "attr_hs_Sunglasses" : 18,
                    "attr_hs_Muffler" : 19,
                    "attr_hs_Mask" : 20,
                    "attr_ub_Shirt" : 21,
                    "attr_ub_Sweater" : 22,
                    "attr_ub_Vest" : 23,
                    "attr_ub_TShirt" : 24,
                    "attr_ub_Cotton" : 25,
                    "attr_ub_Jacket" : 26,
                    "attr_ub_SuitUp" : 27,
                    "attr_ub_Tight" : 28,
                    "attr_ub_ShortSleeve" : 29,
                    "attr_ub_Others" : 30,
                    "attr_ub_ColorBlack" : 31,
                    "attr_ub_ColorWhite" : 32,
                    "attr_ub_ColorGray" : 33,
                    "attr_up_ColorRed" : 34,
                    "attr_ub_ColorGreen" : 35,
                    "attr_ub_ColorBlue" : 36,
                    "attr_ub_ColorSilver" : 37,
                    "attr_ub_ColorYellow" : 38,
                    "attr_ub_ColorBrown" : 39,
                    "attr_ub_ColorPurple" : 40,
                    "attr_ub_ColorPink" : 41,
                    "attr_ub_ColorOrange" : 42,
                    "attr_ub_ColorMixture" : 43,
                    "attr_ub_ColorOther" : 44,
                    "attr_lb_LongTrousers" : 45,
                    "attr_lb_Shorts" : 46,
                    "attr_lb_Skirt" : 47,
                    "attr_lb_ShortSkirt" : 48,
                    "attr_lb_LongSkirt" : 49,
                    "attr_lb_Dress" : 50,
                    "attr_lb_Jeans" : 51,
                    "attr_lb_TightTrousers" : 52,
                    "attr_lb_ColorBlack" : 53,
                    "attr_lb_ColorWhite" : 54,
                    "attr_lb_ColorGray" : 55,
                    "attr_lb_ColorRed" : 56,
                    "attr_lb_ColorGreen" : 57,
                    "attr_lb_ColorBlue" : 58,
                    "attr_lb_ColorSilver" : 59,
                    "attr_lb_ColorYellow" : 60,
                    "attr_lb_ColorBrown" : 61,
                    "attr_lb_ColorPurple" : 62,
                    "attr_lb_ColorPink" : 63,
                    "attr_lb_ColorOrange" : 64,
                    "attr_lb_ColorMixture" : 65,
                    "attr_lb_ColorOther" : 66,
                    "attr_shoes_Leather" : 67,
                    "attr_shoes_Sports" : 68,
                    "attr_shoes_Boots" : 69,
                    "attr_shoes_Cloth" : 70,
                    "attr_shoes_Sandals" : 71,
                    "attr_shoes_Casual" : 72,
                    "attr_shoes_Other" : 73,
                    "attr_shoes_ColorBlack" : 74,
                    "attr_shoes_ColorWhite" : 75,
                    "attr_shoes_ColorGray" : 76,
                    "attr_shoes_ColorRed" : 77,
                    "attr_shoes_ColorGreen" : 78,
                    "attr_shoes_ColorBlue" : 79,
                    "attr_shoes_ColorSilver" : 80,
                    "attr_shoes_ColorYellow" : 81,
                    "attr_shoes_ColorBrown" : 82,
                    "attr_shoes_ColorPurple" : 83,
                    "attr_shoes_ColorPink" : 84,
                    "attr_shoes_ColorOrange" : 85,
                    "attr_shoes_ColorMixture" : 86,
                    "attr_shoes_ColorOther" : 87,
                    "attr_attachment_Backpack" : 88,
                    "attr_attachment_ShoulderBag" : 89,
                    "attr_attachment_HandBag" : 90,
                    "attr_attachment_WaistBag" : 91,
                    "attr_attachment_Box" : 92,
                    "attr_attachment_PlasticBag" : 93,
                    "attr_attachment_PaperBag" : 94,
                    "attr_attachment_HandTrunk" : 95,
                    "attr_attachment_Baby" : 96,
                    "attr_attachment_Other" : 97,
                    "attr_action_Calling" : 98,
                    "attr_action_StrechOutArm" : 99,
                    "attr_action_Talking" : 100,
                    "attr_action_Gathering" : 101,
                    "attr_action_LyingCounter" : 102,
                    "attr_action_Squatting" : 103,
                    "attr_action_Running" : 104,
                    "attr_action_Holding" : 105,
                    "attr_action_Pushing" : 106,
                    "attr_action_Pulling" : 107,
                    "attr_action_CarryingByArm" : 108,
                    "attr_action_CarryingByHand" : 109,
                    "attr_action_Other" : 110,
                    "attr_viewpoint" : 111,
                    "attr_OcclusionLeft" : 112,
                    "attr_OcclusionRight" : 113,
                    "attr_OcclusionUp" : 114,
                    "attr_OcclusionDown" : 115,
                    "attr_occlustion_TypeEnvironment" : 116,
                    "attr_occlustion_TypeAttachment" : 117,
                    "attr_occlustion_TypePerson" : 118,
                    "attr_occlustion_TypeOther" : 119,
                    "attr_person_position_x" : 120,
                    "attr_person_position_y" : 121,
                    "attr_person_position_w" : 122,
                    "attr_person_position_h" : 123,
                    "attr_headshoulder_position_x" : 124,
                    "attr_headshoulder_position_y" : 125,
                    "attr_headshoulder_position_w" : 126,
                    "attr_headshoulder_position_h" : 127,
                    "attr_upperbody_position_x" : 128,
                    "attr_upperbody_position_y" : 129,
                    "attr_upperbody_position_w" : 130,
                    "attr_upperbody_position_h" : 131,
                    "attr_lowerbody_position_x" : 132,
                    "attr_lowerbody_position_y" : 133,
                    "attr_lowerbody_position_w" : 134,
                    "attr_lowerbody_position_h" : 135,
                    "attr_attachment1_position_y" : 137,
                    "attr_attachment1_position_w" : 138,
                    "attr_attachment1_position_h" : 139,
                    "attr_attachment2_position_x" : 140,
                    "attr_attachment2_position_y" : 141,
                    "attr_attachment2_position_w" : 142,
                    "attr_attachment2_position_h" : 143,
                    "attr_attachment3_position_x" : 144,
                    "attr_attachment3_position_y" : 145,
                    "attr_attachment3_position_w" : 146,
                    "attr_attachment3_position_h" : 147,
                    "attr_attachment4_position_x" : 148,
                    "attr_attachment4_position_y" : 149,
                    "attr_attachment4_position_w" : 150,
                    "attr_attachment4_position_h" : 151}


def load_crop_rap_mask(mask_image_path):
    top_border = 200
    bottom_border = 200
    left_border = 237
    right_border = 237

    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mh, mw =  mask.shape[0], mask.shape[1]
    mask_cropped = mask[top_border:mh - bottom_border,
                   left_border: mw - right_border]
    return mask_cropped


def load_rap_keypoints(rap_keypoints_json_path):
    with open(rap_keypoints_json_path) as f:
        data = json.load(f)

        rap_images_keypoints = dict()
        for indx, echfile in enumerate(data):
            img_name = data[indx]['image_id']
            img_keypoints_data = data[indx]['keypoints']
            img_keypoints = []

            for pt_idx in range(0, len(img_keypoints_data), 3):
                pt = (int(img_keypoints_data[pt_idx]), int(img_keypoints_data[pt_idx + 1]))
                img_keypoints.append(pt)

            rap_images_keypoints[img_name] = img_keypoints

    return rap_images_keypoints


def load_rap_attributes(rap_mat_file):
    rap_data = mat4py.loadmat(rap_mat_file)
    # attributes = rap_data['RAP_annotation']['attribute']
    # attributes = np.asarray(attributes)
    # attributes = list(np.squeeze(attributes))

    #names = list(np.squeeze(np.asarray(rap_data['RAP_annotation']['name'])))
    names = list(np.squeeze(np.asarray(rap_data['RAP_annotation']['name'])))
    data = rap_data['RAP_annotation']['data']
    return names, data


def get_images_with_attib(rap_data, attrib_index, attrib_value):
    names_with_attrib = [image_name for image_name in rap_data if rap_data[image_name]['attrs'][attrib_index] == attrib_value]
    return names_with_attrib


def load_rap_dataset(rap_attributes_filepath, rap_keypoints_json):
    image_names, attributes = load_rap_attributes(rap_attributes_filepath)
    rap_keypoints_data = load_rap_keypoints(rap_keypoints_json)
    rap_data = dict()
    print(">> Loading the data from disk ...")
    for idx, image_name in enumerate(image_names):
        if image_name not in rap_keypoints_data:
            continue
        rap_data[image_name] = dict()
        rap_data[image_name]['attrs'] = attributes[idx]
        rap_data[image_name]['keypoints'] = rap_keypoints_data[image_name]
        rap_data[image_name]["attibute_names"] = rap_attibute_names

    return rap_data


def get_head_bbox(attrs):
    head_brect = (int(attrs[attr_headshoulder_position_x] - attrs[attr_person_position_x]), int(attrs[attr_headshoulder_position_y] - attrs[attr_person_position_y]),
                  int(attrs[attr_headshoulder_position_w]), int(attrs[attr_headshoulder_position_h]))

    return head_brect


def get_head_torso_bbox(attrs):

    head_torso_area_tl = (int(attrs[attr_person_position_x]), int(attrs[attr_headshoulder_position_y]))
    head_torso_area_br = (int(attrs[attr_upperbody_position_x] + int(attrs[attr_upperbody_position_w])),
                          int(attrs[attr_upperbody_position_y] + int(attrs[attr_upperbody_position_h])))
    head_torso_w = int(head_torso_area_br[0] - head_torso_area_tl[0])
    head_torso_h = int(head_torso_area_br[1] - head_torso_area_tl[1])

    head_torso_brect = (int(head_torso_area_tl[0] - attrs[attr_person_position_x]),
                  int(head_torso_area_tl[1] - attrs[attr_person_position_y]),
                  head_torso_w, head_torso_h)

    return head_torso_brect

def train_test_imgs(rap_mat_file, partition):
    RAP_dictionary = mat4py.loadmat(rap_mat_file)
    partition_attribute = RAP_dictionary["RAP_annotation"]["partition_attribute"][partition]
    RAP_name_of_imgs = RAP_dictionary["RAP_annotation"]["name"]
    train_index = partition_attribute["train_index"] + partition_attribute["val_index"]
    test_index = partition_attribute["test_index"]
    print(">> Splitting test and train ...")
    train_imgs = []
    for i, IMG_index in enumerate(train_index):
        img_name = RAP_name_of_imgs[IMG_index-1]
        train_imgs.append(img_name[0])
    test_imgs = []
    for i, IMG_index in enumerate(test_index):
        img_name = RAP_name_of_imgs[IMG_index-1]
        test_imgs.append(img_name[0])
    return train_imgs, test_imgs

if __name__ == '__main__':
    rap_data = load_rap_dataset(rap_attribute_annotations, rap_keypoints_json)

    for img_name in rap_data:

        img_path = os.path.join(rap_images_dir, img_name)
        mask_path = os.path.join(rap_masks_dir, img_name)
        keypoints = rap_data[img_name]['keypoints']
        img = cv2.imread(img_path)
        mask = load_crop_rap_mask(mask_path)

        if img is None or mask is None:
            print('Error! Could nou find image or mask for ', img_name)
            continue

        assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]


        for pt in keypoints:
            cv2.circle(img, pt, 3, (0, 255, 0))
        cv2.destroyWindow('keypoints')
        cv2.imshow('keypoints', img)
        cv2.imshow('mask', mask)
        cv2.waitKey()


# RAP_SRC = "/media/ehsan/48BE4782BE476810/MyPythonCodes/RAP/RAP_dataset/RAP_images/"
# json_string = "/media/ehsan/48BE4782BE476810/MyPythonCodes/RAP/RAP_dataset/RAP_Annotations_Keypoints/alpha-pose-results.json"
#
# RAPlist = os.listdir(RAP_SRC)
# with open(json_string) as f:
#     data = json.load(f)
#
# images_without_skeleton = []
#
# for indx, echfile in enumerate(data):
#     X = []
#     Y = []
#     img_nam = data[indx]['image_id']
#     img_keypoints = data[indx]['keypoints']
#     read_img = cv2.imread(RAP_SRC+img_nam)
#     for indx, digit in enumerate(img_keypoints):
#         if indx % 3 == 0:
#             X.append(img_keypoints[indx])
#             Y.append(img_keypoints[indx + 1])
#     plt.scatter(x=X, y=Y)
#     plt.imshow(read_img)
#     plt.waitforbuttonpress()
#     plt.clf()
#     plt.gcf()
#
#
#     """
#     try:
#         img_keypoints = data['{}'.format(img)][0]['keypoints']
#         print(img_keypoints)
#         for indx, digit in enumerate(img_keypoints):
#             if indx % 3 == 0:
#                 X.append(img_keypoints[indx])
#                 Y.append(img_keypoints[indx+1])
#         plt.scatter(x=X, y=Y)
#         plt.imshow(read_img)
#         plt.waitforbuttonpress()
#         plt.clf()
#         plt.gcf()
#         X = []
#         Y = []
#     except KeyError:
#         images_without_skeleton.append(img)
# print("Number of images_without_skeleton: ", len(images_without_skeleton))
# """
# #for x, y in data.items():
# #    for i, j in y[0].items():
# #        print(j)
