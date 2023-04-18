from torchmetrics.image.fid import FrechetInceptionDistance


class FIDMetric:

    def __init__(self):
        super(FIDMetric, self).__init__()
        self.fid = FrechetInceptionDistance(feature=64).to('cuda')

    def update(self, ground_truth, prediction):
        self.fid.update(ground_truth, real=True)
        self.fid.update(prediction, real=False)

    def compute(self):
        return self.fid.compute()