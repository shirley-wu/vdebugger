from word2number import w2n


class TallyQADataset:
    def __init__(self):
        self.input_type = 'image'

    def post_process(self, prediction):
        prediction = str(prediction).strip()
        try:
            return int(prediction)
        except:
            try:
                return w2n(prediction)
            except:
                return None

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
