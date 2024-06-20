import torch
from torchvision.ops import box_iou


class RefCOCODataset:
    def __init__(self):
        self.input_type = 'image'

    @classmethod
    def accuracy(cls, prediction, ground_truth, *args, return_iou=False):
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
                if p is None:
                    continue  # take iou as 0

                if type(p) == list:
                    p = torch.tensor(p)[None]
                elif type(p) == str:
                    p = torch.tensor([float(x) for x in p.split('(')[1].split(')')[0].split(',')])[None]
                else:
                    p = torch.tensor([p.left, p.lower, p.right, p.upper])[None]
                if type(g) == str:
                    g = [float(x) for x in g.split('[')[1].split(']')[0].split(',')]
                g = torch.tensor([g[0], g[1], g[2], g[3]])[None]
                iou_ = box_iou(p, g).item()  # Expects (x1, y1, x2, y2) format. So (left, lower, right, upper)
                iou += iou_
                if iou_ > 0.7:
                    acc += 1
            except Exception as e:
                pass  # If the prediction is not a box, we consider iou = 0

        if return_iou:
            return iou / max(num_samples, 1), acc / max(num_samples, 1)
        else:
            return acc / max(num_samples, 1)  # just return acc
