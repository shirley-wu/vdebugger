import json
import os
import re

import numpy as np
from PIL import Image
from datasets import general_postprocessing
from torch.utils.data import Dataset


class NLVRDataset(Dataset):
    def __init__(self, split, data_path="", image_transforms=None, max_samples=None, start_sample=None,
                 orig_query=False, **kwargs):
        self.split = split
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.max_samples = max_samples
        self.input_type = 'image_dict'
        self.output_type = 'str'
        self.orig_query = orig_query

        if split == "dev":
            split = "balanced_dev"
        elif split == "test1":
            split = "balanced_test1"
        elif split != "train":
            raise NotImplementedError
        # update here
        with open(os.path.join(data_path, f"{split}.jsonl")) as f:
            self.samples = [json.loads(line) for line in f]

        if max_samples is not None:
            np.random.seed(4)
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        if start_sample is not None:
            self.samples = self.samples[start_sample:]

        # For evaluation
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                             "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
                             "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                             "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
                             "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
                             "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
                             "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
                             "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                             "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
                             "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                             "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
                             "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
                             "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
                             "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
                             "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                             "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
                             "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                             "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
                             "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                             "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
                             "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
                             "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
                             "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                             "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
                             "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
                             "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                          'zero': '0',
                          'one': '1',
                          'two': '2',
                          'three': '3',
                          'four': '4',
                          'five': '5',
                          'six': '6',
                          'seven': '7',
                          'eight': '8',
                          'nine': '9',
                          'ten': '10'
                          }
        self.articles = ['a',
                         'an',
                         'the'
                         ]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample['sentence']
        if not text.endswith("?"):
            text = "Is the statement true? " + text
        if not self.orig_query:
            text = f"Given two images, one on the left and one on the right: {text}\ndef execute_command(image_dict) -> str:"
        img_list = []
        img_f_base = os.path.join(self.data_path, self.split)
        if self.split == "train":
            dir = sample["directory"]
            img_f_base = os.path.join(img_f_base, str(dir))
        sample_base = sample["identifier"][:-1] + "img"
        for i in ["0", "1"]:
            with open(os.path.join(img_f_base, sample_base + i + ".png"), "rb") as f:
                img = Image.open(f).convert("RGB")
            if self.image_transforms:
                img = self.image_transforms(img)
            img_list.append(img)
        answer = 'yes' if sample['label'] == "True" else 'no'
        return {'question': text, 'img': {'left': img_list[0], 'right': img_list[1]},
                'sample_id': index, 'answer': answer, 'index': index, 'possible_answers': [], 'info_to_prompt': text,
                "question_type": -1, 'extra_context': ''}

    def __len__(self):
        return len(self.samples)

    def accuracy(self, prediction, ground_truth, *args, **kwargs):
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
            if self.post_process(p) == g:
                score += 1
        return score / len(prediction)

    def post_process(self, prediction, stem=True):
        """
        Code from https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py,
        as indicated here https://okvqa.allenai.org/leaderboard.html
        :return:
        """
        if prediction is None:
            return None

        prediction = general_postprocessing(prediction)

        prediction = prediction.replace('\n', ' ')
        prediction = prediction.replace('\t', ' ')
        prediction = prediction.strip()
        prediction = self.processPunctuation(prediction)
        prediction = self.processDigitArticle(prediction)
        return prediction

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText
