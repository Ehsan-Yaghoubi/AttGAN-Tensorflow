import os
import random
import rap_data_loading as rap


def write_txt(attrs_to_write, txt_dirpath, percent_validation = 0.2, percent_test = 0.2):
    rap_data = rap.load_rap_dataset(rap.rap_attribute_annotations, rap.rap_keypoints_json)

    images_no_head_occlusions = set(rap.get_images_with_attib(rap_data, rap.attr_OcclusionUp, 0))
    target_images = set(rap.get_images_with_attib(rap_data, rap.attr_viewpoint, 1))
    target_images = target_images.intersection(images_no_head_occlusions)

    target_images = list(target_images)
    random.shuffle(target_images)

    num_images = len(target_images)
    test_images = target_images[:int(num_images*percent_test)]
    validation_images = target_images[int(num_images*percent_test): int(num_images*(percent_test + percent_validation))]
    train_images = target_images[int(num_images*(percent_test + percent_validation)):]

    print('Writing data: %d test, %d validation, %d train ', len(test_images), len(validation_images), len(train_images))

    images_list = [test_images, validation_images, train_images]
    txt_files = [os.path.join(txt_dirpath, 'test.txt'),
                    os.path.join(txt_dirpath, 'validation.txt'),
                    os.path.join(txt_dirpath, 'train.txt')]

    for txt_filepath, images in zip(txt_files, images_list):
        print('Write ', txt_filepath)
        with open(txt_filepath, 'w') as output_file:
            output_file.write('filename ')
            for attr in attrs_to_write:
                output_file.write(list(rap.rap_attibute_names.keys())[list(rap.rap_attibute_names.values()).index(attr)]+ ' ')
            output_file.write('\n')
            for image_name in images:
                image_path = os.path.join(rap.rap_images_dir, image_name)
                image_attrs = rap_data[image_name]['attrs']

                image_line = '%s ' % image_path
                for attr in attrs_to_write:
                    image_line += ' %d ' % min(image_attrs[attr], 1)
                image_line += '\n'
                output_file.write(image_line)


    print('Done.')


if __name__ == '__main__':
    attrs_to_write = [rap.attr_Female]
    output_filepath = '.'
    write_txt(attrs_to_write, output_filepath)
