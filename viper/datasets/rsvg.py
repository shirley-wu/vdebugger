import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_iou


def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if
            f.endswith(file_type)]


class RSVGDataset(Dataset):
    def __init__(self, data_path, imsize=640, image_transforms=None, split='train', max_samples=None, orig_query=False,
                 load_image=False, **kwargs):
        self.images = []
        self.images_path = os.path.join(data_path, 'JPEGImages')
        self.anno_path = os.path.join(data_path, 'Annotations')
        self.imsize = imsize

        self.split = split
        self.max_samples = max_samples
        self.image_transforms = image_transforms
        self.input_type = 'image'
        self.output_type = 'ImagePatch'
        self.orig_query = orig_query
        self.load_image = load_image

        with open(os.path.join(data_path, split + '.txt'), "r") as f:
            file = f.readlines()
        Index = [int(index.strip('\n')) for index in file]
        count = 0
        annotations = filelist(self.anno_path, '.xml')
        for anno_path in annotations:
            root = ET.parse(anno_path).getroot()
            for member in root.findall('object'):
                if count in Index:
                    imageFile = str(self.images_path) + '/' + root.find("./filename").text
                    box = np.array([int(member[2][0].text), int(member[2][1].text), int(member[2][2].text),
                                    int(member[2][3].text)], dtype=np.float32)
                    text = member[3].text
                    self.images.append((imageFile, box, text))
                count += 1

        if self.max_samples is not None:
            self.images = self.images[:self.max_samples]

    def __getitem__(self, index):
        img_path, bbox, text = self.images[index]

        if self.load_image:
            with open(img_path, "rb") as f:
                pil_img = Image.open(f).convert("RGB")
            _, height = pil_img.size
            if self.image_transforms:
                img = self.image_transforms(pil_img)
            else:
                img = pil_img

            bbox = [bbox[0], height - bbox[3], bbox[2], height - bbox[1], ]

        else:
            img = None
            bbox = [0, 0, 0, 0]

        # There are different texts associated to every image
        if self.orig_query:
            text = text.lower()
        else:
            text = f"Given an image: Find {text}.\ndef execute_command(image) -> ImagePatch:"

        return {'question': text, 'img': img, 'sample_id': index, 'answer': bbox, 'index': index,
                'possible_answers': [], 'info_to_prompt': text, "question_type": -1, 'extra_context': ''}

    def __len__(self):
        return len(self.images)

    @classmethod
    def accuracy(cls, prediction, ground_truth, *args, **kwargs):
        """
        Compute IoU score
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
        Returns:
            score (float): Score of the prediction. It is an IoU score
        """
        assert len(prediction) == len(ground_truth)
        num_samples = 0
        iou = 0
        acc = 0
        for p, g in zip(prediction, ground_truth):
            num_samples += 1

            try:
                try:
                    if type(p) == list:
                        p = torch.tensor(p)[None]
                        assert tuple(p.shape) == (1, 4)
                    elif type(p) == str:
                        p = torch.tensor([float(x) for x in p.split('(')[1].split(')')[0].split(',')])[None]
                    else:
                        p = torch.tensor([p.left, p.lower, p.right, p.upper])[None]
                except:
                    continue

                if type(g) == str:
                    g = [float(x) for x in g.split('[')[1].split(']')[0].split(',')]
                g = torch.tensor([g[0], g[1], g[2], g[3]])[None]
                iou_ = box_iou(p, g).item()  # Expects (x1, y1, x2, y2) format. So (left, lower, right, upper)
                iou += iou_
                if iou_ > 0.7:
                    acc += 1

            except:
                pass

        return iou / max(num_samples, 1), acc / max(num_samples, 1)
