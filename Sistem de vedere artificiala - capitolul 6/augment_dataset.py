import os
import cv2
import albumentations as A
import numpy as np


transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    A.Affine(rotate=(-180, 180), p=1.0),
    A.Rotate(limit=360, p=1.0),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    bboxes = []
    class_labels = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.strip().split())
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(int(class_id))
    return bboxes, class_labels

def write_yolo_annotation(annotation_path, bboxes, class_labels):
    with open(annotation_path, 'w') as f:
        for bbox, class_label in zip(bboxes, class_labels):
            f.write(f"{class_label} {' '.join(map(str, bbox))}\n")

def augment_image(image_path, annotation_path, save_image_path, save_annotation_path, num_augmentations=5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read the image from {image_path}")
        return

    bboxes, class_labels = read_yolo_annotation(annotation_path)
    if not bboxes:
        print(f"No bounding boxes found for {image_path}")
        return
    print(f"Read {len(bboxes)} annotations from {annotation_path}")

    for i in range(num_augmentations):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_labels = augmented['class_labels']

        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        new_image_name = f"{name}_aug_{i}{ext}"
        new_image_path = os.path.join(save_image_path, new_image_name)
        new_annotation_path = os.path.join(save_annotation_path, f"{name}_aug_{i}.txt")

        cv2.imwrite(new_image_path, augmented_image)
        write_yolo_annotation(new_annotation_path, augmented_bboxes, augmented_class_labels)
        print(f"Saved augmented image and annotation: {new_image_path}, {new_annotation_path}")

def augment_folder(image_folder_path, label_folder_path, save_image_path, save_annotation_path, num_augmentations=5):
    if not os.path.exists(image_folder_path):
        print(f"Image folder not found: {image_folder_path}")
        return

    if not os.path.exists(label_folder_path):
        print(f"Label folder not found: {label_folder_path}")
        return

    for file in os.listdir(image_folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder_path, file)
            annotation_path = os.path.join(label_folder_path, os.path.splitext(file)[0] + '.txt')
            if os.path.exists(annotation_path):
                print(f"Processing {image_path} and {annotation_path}")
                augment_image(image_path, annotation_path, save_image_path, save_annotation_path, num_augmentations)
            else:
                print(f"Annotation not found for {image_path}")

def augment_dataset(dataset_path, num_augmentations=5):
    for split in ['train', 'test']:
        image_folder_path = os.path.join(dataset_path, split, 'images')
        label_folder_path = os.path.join(dataset_path, split, 'labels')
        save_image_path = image_folder_path
        save_annotation_path = label_folder_path
        print(f"Augmenting dataset split: {split}")
        augment_folder(image_folder_path, label_folder_path, save_image_path, save_annotation_path, num_augmentations)


dataset_path = 'D:\PycharmProjects\CameraSetup\corner-dataset'
augment_dataset(dataset_path, 20)
