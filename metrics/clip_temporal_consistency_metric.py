import torch
from torch.nn.functional import upsample_bilinear
import clip


class CLIPTemporalConsistencyMetric:

    def __init__(self, backbone="ViT-B/32"):
        super(CLIPTemporalConsistencyMetric, self).__init__()
        self.model, self.preprocess = clip.load(backbone, device="cuda")

        # Freeze the CLIP model itself
        for param in self.model.parameters():
            param.requires_grad = False

    def compute(self, image_source_0, image_target_0, image_target_1):
        image_source_0 = upsample_bilinear(image_source_0, (224, 224))
        image_target_0 = upsample_bilinear(image_target_0, (224, 224))
        image_target_1 = upsample_bilinear(image_target_1, (224, 224))

        encoded_image_source_0 = self.model.encode_image(image_source_0).squeeze()
        encoded_image_target_0 = self.model.encode_image(image_target_0).squeeze()
        encoded_image_target_1 = self.model.encode_image(image_target_1).squeeze()

        return torch.dot(encoded_image_target_0 - encoded_image_source_0,
                         encoded_image_target_1 - encoded_image_target_0)
