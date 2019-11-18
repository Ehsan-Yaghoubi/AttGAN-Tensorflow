import os
import cv2
import utils
import random
import numpy as np
import rap_data_loading as rap


def get_head_area(img, mask, keypoints):
    assert img is not None and mask is not None

    head_point = keypoints[rap.kp_head]
    neck_point = keypoints[rap.kp_neck]
    left_elbow = keypoints[rap.kp_left_elbow]
    right_elbow = keypoints[rap.kp_right_elbow]

    # @todo: treat case when persons is viewed from the back and the left and right side are reversed
    if left_elbow[0] > right_elbow[0]:
        left_elbow, right_elbow = right_elbow, left_elbow

    # head bounding rect in format: [x, y, w, h]
    approx_head_brect = [left_elbow[0], 0,
                        right_elbow[0] - left_elbow[0], neck_point[1]]

    head_area_mask = mask[approx_head_brect[1]: approx_head_brect[1] + approx_head_brect[3],
                    approx_head_brect[0]: approx_head_brect[0] + approx_head_brect[2]]
    _, contours,  _ = cv2.findContours(head_area_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # no head contours found
    if len(contours) == 0:
        return None, None, None

    head_contour = contours[0]
    head_contour = utils.translate_points(head_contour, (approx_head_brect[0], approx_head_brect[1]))

    head_brect = cv2.boundingRect(head_contour)
    head_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(head_mask, [head_contour], -1, (255, 255, 255), -1)
    # cv2.destroyWindow('head contour')
    # cv2.imshow('head contour', head_mask)

    return head_brect, head_contour, head_mask


def remove_head_area(img, head_brect,  head_mask):
    kernel = np.ones((7, 7), np.uint8)

    head_mask_enlarged = cv2.dilate(head_mask, kernel, iterations=5)
    img_head_removed = cv2.inpaint(img, head_mask_enlarged, head_brect[2]*0.1 , cv2.INPAINT_TELEA)

    return img_head_removed


def replace_head_rect(img1, mask1, keypoints1, attrs1,
                     img2, mask2, keypoints2, attrs2):

    head_brect1, head_contour1, head_mask1 = get_head_area(img1, mask1, keypoints1)
    head_brect2, head_contour2, head_mask2 = get_head_area(img2, mask2, keypoints2)

    if head_brect1 is None or head_brect2 is None:
        return None

    # head_brect1 = (int(attrs1[rap.attr_headshoulder_position_x] - attrs1[rap.attr_person_position_x]), int(attrs1[rap.attr_headshoulder_position_y] - attrs1[rap.attr_person_position_y]),
    #                int(attrs1[rap.attr_headshoulder_position_w]), int(attrs1[rap.attr_headshoulder_position_h]))
    #
    # head_brect2 = (int(attrs2[rap.attr_headshoulder_position_x]  - attrs2[rap.attr_person_position_x]), int(attrs2[rap.attr_headshoulder_position_y] - attrs2[rap.attr_person_position_y]),
    #                int(attrs2[rap.attr_headshoulder_position_w]), int(attrs2[rap.attr_headshoulder_position_h]))

    result_image = remove_head_area(img1, head_brect1, head_mask1)

    head_img2 = img2[head_brect2[1]: head_brect2[1]+head_brect2[3],
                    head_brect2[0]: head_brect2[0]+head_brect2[2]]
    head_img2 = cv2.resize(head_img2, (head_brect1[2], head_brect1[3]))

    result_image[head_brect1[1]:head_brect1[1]+head_brect1[3],
                    head_brect1[0]:head_brect1[0]+head_brect1[2]] = head_img2

    # cv2.rectangle(result_image, (head_brect1[0], head_brect1[1]),
    #               (head_brect1[0] + head_brect1[2], head_brect1[1] + head_brect1[3]), (0, 255, 0), 2)

    return result_image


def replace_head_area(img1, mask1, keypoints1,
                     img2, mask2, keypoints2):

    head_brect1, head_contour1, head_mask1 = get_head_area(img1, mask1, keypoints1)
    head_brect2, head_contour2, head_mask2 = get_head_area(img2, mask2, keypoints2)

    if head_brect1 is None or head_brect2 is None:
        return None

    result_image = remove_head_area(img1, head_brect1, head_mask1)

    head_img2 = img2[head_brect2[1]: head_brect2[1] + head_brect2[3],
                head_brect2[0]: head_brect2[0] + head_brect2[2]]
    mask_img2 = mask2[head_brect2[1]: head_brect2[1] + head_brect2[3],
                head_brect2[0]: head_brect2[0] + head_brect2[2]]

    head_img2 = cv2.resize(head_img2, (head_brect1[2], head_brect1[3]))
    mask_img2 = cv2.resize(mask_img2, (head_brect1[2], head_brect1[3]))
    mask_img2 = cv2.cvtColor(mask_img2, cv2.COLOR_GRAY2BGR)
    mask_img2 = mask_img2.astype(np.bool)

    np.copyto(result_image[head_brect1[1]: head_brect1[1] + head_brect1[3],
                head_brect1[0]: head_brect1[0] + head_brect1[2]], head_img2, where=mask_img2, casting='unsafe')

    return result_image

# ---------------------------------------------------
# constrain functions
# ---------------------------------------------------
def is_compatible_area(img1, img2, mask1, mask2,
                       kp1, kp2, attr1, attr2):
    area1 = cv2.countNonZero(mask1)
    area2 = cv2.countNonZero(mask2)
    th_area = 0.8
    if min(area1, area2)/max(area1, area2) < th_area:
        return False
    return True


def is_compatible_iou(img1, img2, mask1, mask2,
                      kp1, kp2, attr1, attr2):
    _, contours1, _ = cv2.findContours(mask1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, contours2, _ = cv2.findContours(mask1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours1) == len(contours2) == 0
    brect1 = cv2.boundingRect(contours1[0])
    brect2 = cv2.boundingRect(contours2[0])


    return True
# ---------------------------------------------------
# end constraint functions
# ---------------------------------------------------

def generate_syntethic_images(rap_data, num_images_to_generate, viewpoint = 1, other_attrs = None,
                              constraint_functions = []):

    images_no_head_occlusions = set(rap.get_images_with_attib(rap_data, rap.attr_OcclusionUp, 0))
    target_images = set(rap.get_images_with_attib(rap_data, rap.attr_viewpoint, viewpoint))
    target_images = target_images.intersection(images_no_head_occlusions)

    if other_attrs is not None:
        for attr in other_attrs:
            images_with_attr = rap.get_images_with_attib(rap_data, attr, other_attrs[attr])
            if len(images_with_attr) > 0:
                target_images = target_images.intersection(set(images_with_attr))

    target_images = list(target_images)

    for idx in range(0, num_images_to_generate):
        cv2.destroyAllWindows()

        img_name1 = random.choice(target_images)
        img_name2 = random.choice(target_images)

        img_path1 = os.path.join(rap.rap_images_dir, img_name1)
        mask_path1 = os.path.join(rap.rap_masks_dir, img_name1)
        keypoints1 = rap_data[img_name1]['keypoints']
        attr1 = rap_data[img_name1]['attrs']
        img1 = cv2.imread(img_path1)
        mask1 = rap.load_crop_rap_mask(mask_path1)

        img_path2 = os.path.join(rap.rap_images_dir, img_name2)
        mask_path2 = os.path.join(rap.rap_masks_dir, img_name2)
        keypoints2 = rap_data[img_name2]['keypoints']
        attr2 = rap_data[img_name2]['attrs']
        img2 = cv2.imread(img_path2)
        mask2 = rap.load_crop_rap_mask(mask_path2)

        for constraint_function in constraint_functions:
            if not constraint_function(img1, img2, mask1, mask2, keypoints1, keypoints2,
                                       attr1, attr2):
                idx -= 1
                continue


        generated_replaced_area = replace_head_area(img1, mask1, keypoints1,
                                                    img2, mask2, keypoints2)
        generated_replaced_rect = replace_head_rect(img1, mask1, keypoints1, attr1,
                                                    img2, mask2, keypoints2, attr2)


        # cv2.imshow('img1', img1)
        # cv2.imshow('img2', img2)
        # if generated_replaced_area is not None:
        #     cv2.imshow('replaced head - by mask', generated_replaced_area)
        # if generated_replaced_rect is not None:
        #     cv2.imshow('replaced head - by brect', generated_replaced_rect)
        # cv2.waitKey()
        # display
        img2_display = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        concat_images = [img1, img2_display]
        if generated_replaced_area is not None:
            concat_images.append(generated_replaced_area)
        if generated_replaced_rect is not None:
            concat_images.append(generated_replaced_rect)

        display_img = cv2.hconcat(concat_images)
        cv2.imshow('morphing', display_img)
        cv2.waitKey()
    return


if __name__ == '__main__':

    rap_dataset = rap.load_rap_dataset(rap.rap_attribute_annotations, rap.rap_keypoints_json)
    additional_attrs = {rap.attr_Female: 1}
    constraint_funcs = [is_compatible_area]
    num_images_to_generate = 30
    generate_syntethic_images(rap_dataset, num_images_to_generate, other_attrs=additional_attrs,
                              constraint_functions=constraint_funcs)


