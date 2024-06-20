import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from word2number import w2n


class TallyQADataset(Dataset):
    def __init__(self, split, data_path="", image_transforms=None, max_samples=None, is_simple=None, orig_query=True,
                 **kwargs):
        print("Unused config params:", kwargs.keys())

        self.split = split
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.input_type = 'image'
        self.output_type = 'str'
        self.orig_query = orig_query

        with open(os.path.join(self.data_path, self.split + '.json')) as f:
            samples = json.load(f)

        if is_simple is None:
            self.samples = samples
        else:
            if is_simple:
                self.samples = [s for s in samples if s['issimple']]
                print("Select simple samples: %d out of %d" % (len(self.samples), len(samples)))
            else:
                self.samples = [s for s in samples if not s['issimple']]
                print("Select complex samples: %d out of %d" % (len(self.samples), len(samples)))

        if max_samples is not None:
            np.random.seed(4)
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __getitem__(self, index):
        item = self.samples[index]

        question = item['question']
        if not self.orig_query:
            question = f"Given an image: {question}\ndef execute_command(image) -> str:"

        question_type = -1
        image_path = os.path.join(self.data_path, item['image'])
        with open(image_path, "rb") as f:
            pil_img = Image.open(f).convert("RGB")
        if self.image_transforms:
            img = self.image_transforms(pil_img)
        else:
            img = pil_img

        out_dict = {"sample_id": index, "answer": item['answer'], "img": img, "question": question,
                    'pil_img': pil_img, "question_type": question_type, 'index': index, 'possible_answers': [],
                    'info_to_prompt': question, }
        return out_dict

    def post_process(self, prediction, strict):
        prediction = str(prediction).strip()
        try:
            return int(prediction)
        except:
            try:
                return w2n(prediction)
            except:
                if not strict:
                    return 0
                else:
                    return None

    def accuracy(self, prediction, ground_truth, *args, strict=True, **kwargs):
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
        score = 0
        for p, g in zip(prediction, ground_truth):
            if self.post_process(p, strict=strict) == g:
                score += 1
        return score / len(prediction)

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.samples)
