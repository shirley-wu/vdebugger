"""
Data loaders
Adapted in part from https://github.com/phiyodr/vqaloader/blob/master/vqaloader/loaders.py
"""
import copy

import torch
from torchvision import transforms


# ----------------------------- General for all datasets ----------------------------- #
def get_dataset(config_dataset, load_image: bool, orig_query: bool = False):
    dataset_name = config_dataset.dataset_name
    if 'orig_query' in config_dataset:
        config_dataset = copy.deepcopy(config_dataset)
        orig_query = config_dataset.pop('orig_query')

    if dataset_name == 'RefCOCO':
        from datasets.refcoco import RefCOCODataset
        dataset = RefCOCODataset(
            **config_dataset, image_transforms=transforms.Compose([transforms.ToTensor()]), load_image=load_image,
            orig_query=orig_query,
        )
    elif dataset_name == 'RSVG':
        from datasets.rsvg import RSVGDataset
        dataset = RSVGDataset(
            **config_dataset, image_transforms=transforms.Compose([transforms.ToTensor()]), load_image=load_image,
            orig_query=orig_query,
        )
    elif dataset_name == 'MultiVersionRefCOCO':
        from datasets.refcoco import MultiVersionRefCOCODataset
        dataset = MultiVersionRefCOCODataset(
            **config_dataset, image_transforms=transforms.Compose([transforms.ToTensor()]), load_image=load_image,
            orig_query=orig_query,
        )
    elif dataset_name == 'GQA':
        from datasets.gqa import GQADataset
        dataset = GQADataset(
            **config_dataset, balanced=True, image_transforms=transforms.Compose([transforms.ToTensor()]),
            load_image=load_image, orig_query=orig_query,
        )
    elif dataset_name == 'TallyQA':
        from datasets.tallyqa import TallyQADataset
        dataset = TallyQADataset(
            **config_dataset, image_transforms=transforms.Compose([transforms.ToTensor()]), load_image=load_image,
            orig_query=orig_query,
        )
    elif dataset_name == 'VSR':
        from datasets.vsr import VSRDataset
        dataset = VSRDataset(
            **config_dataset, image_transforms=transforms.Compose([transforms.ToTensor()]), load_image=load_image,
            orig_query=orig_query,
        )
    elif dataset_name == 'COVR':
        from datasets.covr import COVRDataset
        dataset = COVRDataset(
            **config_dataset, image_transforms=transforms.Compose([transforms.ToTensor()]), load_image=load_image,
            orig_query=orig_query,
        )
    elif dataset_name == 'NLVR':
        from datasets.nlvr import NLVRDataset
        dataset = NLVRDataset(
            **config_dataset, image_transforms=transforms.Compose([transforms.ToTensor()]), load_image=load_image,
            orig_query=orig_query,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset


def general_postprocessing(prediction):
    try:
        if type(prediction).__name__ == 'ImagePatch':
            prediction = prediction.classify_object()

        if isinstance(prediction, list):
            prediction = prediction[0] if len(prediction) > 0 else "no"

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.item()
        # if prediction is None:
        #     prediction = "no"
        if isinstance(prediction, bool):
            if prediction:
                prediction = "yes"
            else:
                prediction = "no"
        elif isinstance(prediction, int):
            prediction = str(prediction)
            # print("No answer is a number, so this will be wrong")
    except:
        prediction = str(prediction)

    prediction = str(prediction)

    prediction = prediction.replace('\n', ' ')
    prediction = prediction.replace('\t', ' ')
    prediction = prediction.strip()
    prediction = prediction.lower()

    if prediction == 'true':
        prediction = 'yes'
    elif prediction == 'false':
        prediction = 'no'
    return prediction


def accuracy(prediction, ground_truth, *args):
    """
    Args:
        prediction (list): List of predicted answers.
        ground_truth (list): List of ground truth answers.
    Returns:
        score (float): Score of the prediction.
    """
    if len(prediction) == 0:  # if no prediction, return 0
        return 0
    assert len(prediction) == len(ground_truth)
    pred_gt_filtered = [(pred, gt) for pred, gt in zip(prediction, ground_truth) if gt != '']
    score = 0
    for p, g in pred_gt_filtered:
        if general_postprocessing(p) == g:
            score += 1
    return score / len(pred_gt_filtered)
