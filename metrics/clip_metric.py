import torch
import clip


class CLIPMetric:

    def __init__(self, backbone="ViT-B/32"):
        super(CLIPMetric, self).__init__()
        self.model, self.preprocess = clip.load(backbone, device="cuda")

        # Freeze the CLIP model itself
        for param in self.model.parameters():
            param.requires_grad = False

    def compute(self, image, text):
        text = torch.cat([clip.tokenize(text)]).cuda()
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        return self.model(image, text)[0] / 100