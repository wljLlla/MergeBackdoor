import torch
import open_clip
from torch import nn

class ImageEncoder(torch.nn.Module):
    def __init__(self, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='commonpool_xl_laion_s13b_b90k')

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, nb_classes):
        super().__init__()
        self.image_encoder = image_encoder
        # self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=768, out_features=nb_classes),
            # nn.Linear(in_features=512, out_features=nb_classes),
            nn.Softmax(dim=-1)
        )

    def freeze_head(self):
        self.fc1.weight.requires_grad_(False)
        self.fc1.bias.requires_grad_(False)

    def forward(self, inputs, return_feature=False):
        features = self.image_encoder(inputs)
        features = features.view(features.size(0), -1)
        outputs = self.fc1(features)
        if return_feature:
            return outputs, features
        else:
            return outputs

    def __call__(self, inputs):
        return self.forward(inputs)
