import json
import os
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SAM annotations to COCO format")
    parser.add_argument('--img_list', type=str, required=True, help='Path to the image list JSON file')
    parser.add_argument('--input_directory', type=str, required=True, help='Directory containing the input JSON files')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output JSON file with full annotations')
    return parser.parse_args()


def process_file(file):
    with open(file, 'r') as f:
        json_data = json.load(f)
    image = json_data['image']
    annotations = json_data['annotations']

    coco_image = {
        'id': image['image_id'],
        'width': image['width'],
        'height': image['height'],
        'file_name': image['file_name']
    }

    coco_annotations = []
    coco_annotations_bbox_only = []
    for annotation in annotations:
        coco_annotation = {
            'bbox': annotation['bbox'],
            'area': annotation['area'],
            'segmentation': annotation['segmentation'],
            'id': annotation['id'],
            'image_id': image['image_id'],
            'category_id': 1,
            'iscrowd': 0
        }

        coco_annotation_bbox = {
            'bbox': annotation['bbox'],
            'area': annotation['area'],
            'id': annotation['id'],
            'image_id': image['image_id'],
            'category_id': 1,
            'iscrowd': 0
        }
        coco_annotations.append(coco_annotation)
        coco_annotations_bbox_only.append(coco_annotation_bbox)

    return coco_image, coco_annotations, coco_annotations_bbox_only


def initializer():
    print(f'Initializing process {os.getpid()}')


if __name__ == '__main__':
    args = parse_args()

    img_list = json.load(open(args.img_list))
    input_directory = args.input_directory
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    files = [os.path.join(input_directory, img['file_name'].replace('.jpg', '.json')) for img in img_list['images']]

    coco_format_full_anno = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    coco_format_bbox_anno = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    with Pool(processes=cpu_count(), initializer=initializer) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))

    for image, annotations, annotation_bbox in results:
        coco_format_full_anno['images'].append(image)
        coco_format_full_anno['annotations'].extend(annotations)

        coco_format_bbox_anno['images'].append(image)
        coco_format_bbox_anno['annotations'].extend(annotation_bbox)

    with open(os.path.join(output_folder, 'sa1b_coco_fmt_500k_mask_anno.json'), 'w') as f:
        json.dump(coco_format_full_anno, f)

    with open(os.path.join(output_folder, 'sa1b_coco_fmt_500k_bbox_anno.json'), 'w') as f:
        json.dump(coco_format_bbox_anno, f)
