import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
import config as CFG


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    Pytorch timm library was used here, bringing resnet50 as the image encoder; torchvision can be easily used otherwise
    The code encodes each image to a fixed size vector with the size of the model’s output channels
    (in case of ResNet50 the vector size will be 2048). This is the output after the nn.AdaptiveAvgPool2d() layer.
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    """
    DistilBERT used as text encoder; just like BERT, 2 special tokens are added to the actual input tokens:
    CLP and SEP, which mark the beginning and end of a sentence
    to grab the whole representation of a sentence, the final representation of the CLS token is used, hoping it
    captures the overall meaning of the sentence (caption)
    Similarly with images, they were converted into fixed size vectors
    """
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    """
    With both images and text encoded in vectors (size 2048 for image and 768 for text) we need to project them into a
    new world with similar dimensions for both, in order to be able to compare and push apart non-relevant pairs and
    pull together the ones that are matching.
    Therefore, here we bring the vectors into a 256 (projection_dim) dimensional world where we can compare them.
    """
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

