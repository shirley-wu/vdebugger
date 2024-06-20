import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_iou


def load_samples(data_path, version, split, split_by):
    assert version in ['refcoco', 'refcoco+', 'refcocog']

    # load refs from data/dataset/refs(dataset).json
    ref_file = os.path.join(data_path, version, 'refs(' + split_by + ').p')
    with open(ref_file, 'rb') as f:
        refs = pickle.load(f)
    # refs index
    refs_index = {}
    for ref in refs:
        refs_index[ref['ref_id']] = ref

    # load annotations from data/dataset/instances.json
    instances_file = os.path.join(data_path, version, 'instances.json')
    with open(instances_file, 'r') as f:
        instances = json.load(f)

    # image index: just for computing heights, not really useful
    img_index = {}
    for img in instances['images']:
        img_index[img['id']] = img

    # annotation index 
    annotations_index = {}
    for ann in instances['annotations']:
        annotations_index[ann['id']] = ann
        height = img_index[ann['image_id']]['height']  # adjust coordinates
        ann['bbox'] = [ann['bbox'][0], height - (ann['bbox'][1] + ann['bbox'][3]), ann['bbox'][2] + ann['bbox'][0],
                       height - ann['bbox'][1]]

    # ref to annotation
    ref_to_ann = {}
    for ref in refs:
        ref_id = ref['ref_id']
        ann_id = ref['ann_id']
        ref_to_ann[ref_id] = annotations_index[ann_id]

    def get_sample(ref_id, sent_id):
        ref = refs_index[ref_id]
        return {
            'img_path': get_sample_path(ref=ref),
            'text': ref['sentences'][sent_id]['sent'],
            'answer': ref_to_ann[ref_id]['bbox'],
        }

    def get_ref_ids(split):
        if split in ['testA', 'testB', 'testC']:
            split_refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
        elif split in ['testAB', 'testBC', 'testAC']:
            split_refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
        elif split == 'test':
            split_refs = [ref for ref in refs if 'test' in ref['split']]
        elif split == 'train' or split == 'val':
            split_refs = [ref for ref in refs if ref['split'] == split]
        else:
            raise KeyError(f'No split {split}')

        ref_ids = [ref['ref_id'] for ref in split_refs]
        return ref_ids

    def get_sample_path(index=None, ref=None):
        if ref is None:
            assert index is not None
            ref_id, i = samples[index]
            ref = refs_index[ref_id]

        file_name = '_'.join(ref['file_name'].split('_')[:-1]) + '.' + ref['file_name'].split('.')[-1]
        coco_split = file_name.split('_')[1]

        img_path = os.path.join(data_path, 'mscoco', coco_split, file_name)
        return img_path

    ref_ids = get_ref_ids(split=split)
    samples = []
    for ref_id in ref_ids:
        ref = refs_index[ref_id]
        for i in range(len(ref['sent_ids'])):
            samples.append(get_sample(ref_id, i))
    return samples


class RefCOCODataset(Dataset):
    """
    Used code from https://github.com/lichengunc/refer/blob/master/refer.py
    """

    def __init__(self, split, data_path="", image_transforms=None, question_transforms=None, tokenize=None,
                 max_samples=None, version='refcoco', split_by='unc', orig_query=False, **kwargs):
        self.split = split
        self.data_path = data_path
        self.max_samples = max_samples
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        self.input_type = 'image'
        self.output_type = 'ImagePatch'
        self.orig_query = orig_query

        self.samples = load_samples(data_path, version, split, split_by)

        np.random.seed(4)
        np.random.shuffle(self.samples)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __getitem__(self, index):
        item = self.samples[index]
        img_path = item['img_path']
        text = item['text']
        answer = item['answer']

        with open(img_path, "rb") as f:
            pil_img = Image.open(f).convert("RGB")
        if self.image_transforms:
            img = self.image_transforms(pil_img)
        else:
            img = pil_img

        # There are different texts associated to every image
        if self.orig_query:
            text = text
        else:
            text = f"Given an image: Find {text}.\ndef execute_command(image) -> ImagePatch:"

        return {'question': text, 'img': img, 'sample_id': index, 'answer': answer, 'index': index,
                'possible_answers': [], 'info_to_prompt': text, "question_type": -1, 'extra_context': ''}

    def __len__(self):
        return len(self.samples)

    @classmethod
    def accuracy(cls, prediction, ground_truth, *args, strict=True):
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
                    if not strict:
                        p = torch.tensor([50.9, 39.1, 493.5, 356.5])[None]  # Mean IoU is 22.64%
                    else:
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


class MultiVersionRefCOCODataset(RefCOCODataset):
    def __init__(self, data_path="", image_transforms=None, question_transforms=None, tokenize=None, max_samples=None,
                 start_sample=None, versions=0, **kwargs):
        self.data_path = data_path
        self.max_samples = max_samples
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        self.input_type = 'image'
        self.output_type = 'ImagePatch'

        self.version_infos = []
        self.samples = []
        cache_set = set()
        for version_id in range(1, versions + 1):
            version_info = kwargs[f'version_{version_id}']
            self.version_infos.append(version_info)
            samples = load_samples(data_path, version_info['version'], version_info['split'], version_info['split_by'])
            for sample in samples:
                jsample = json.dumps(sample)
                if jsample not in cache_set:
                    cache_set.add(jsample)
                    self.samples.append(sample)

        np.random.seed(4)
        np.random.shuffle(self.samples)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if start_sample is not None:
            self.samples = self.samples[start_sample:]
        print("Length:", len(self.samples))
