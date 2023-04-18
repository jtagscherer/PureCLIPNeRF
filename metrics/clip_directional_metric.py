from torch.nn.functional import upsample_bilinear
from torch.linalg import vector_norm
import clip
import torch


class CLIPDirectionalMetric:

    def __init__(self, backbone="ViT-B/32"):
        super(CLIPDirectionalMetric, self).__init__()
        self.model, self.preprocess = clip.load(backbone, device="cuda")

        # Freeze the CLIP model itself
        for param in self.model.parameters():
            param.requires_grad = False

    def compute(self, image_source, image_target, text_source, text_target):
        image_source = upsample_bilinear(image_source, (224, 224))
        image_target = upsample_bilinear(image_target, (224, 224))

        encoded_image_source = self.model.encode_image(image_source).squeeze()
        encoded_image_target = self.model.encode_image(image_target).squeeze()

        text_source = torch.cat([clip.tokenize(text_source)]).cuda()
        text_target = torch.cat([clip.tokenize(text_target)]).cuda()

        encoded_text_source = self.model.encode_text(text_source).squeeze()
        encoded_text_target = self.model.encode_text(text_target).squeeze()

        text_delta = encoded_text_target - encoded_text_source
        image_delta = encoded_image_target - encoded_image_source

        return 1 - (image_delta * text_delta) / (vector_norm(image_delta) * vector_norm(text_delta))
