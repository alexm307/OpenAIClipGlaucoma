import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, TextEncoder, ProjectionHead


class CLIPModel(nn.Module):
    """
    This is the main model, CLIP, which uses the image and text projections to make same dimension embeddings of the
    text and image features, then calculates the loss, by using the method of Linear Algebra for determining whether
    two vectors possess similar features, the dot product. Technically, the larger the product the higher the
    supposed similarity is between the vectors. (@ operator performs the dot product in python)
    The second matrix is transposed to facilitate matrix multiplication, obtaining a matrix of shape
    (batch_size, batch_size) called logits.
    (temperature is equal to 1.0 in our case, so, it does not make a difference. You can play with it and see what
    difference it makes. Also look at the paper to see why it is here!).
    In the best case, logits will be an identity matrix, if we use softmax on it. Therefore we want that matrix as
    target.
    For the actual loss simple cross entropy is used.
    Th reason for not using nn.CrossEntropyLoss()(logits, torch.arange(batch_size)) is that in case of images depicting
    the same thing, but with slightly different captions, this function would pull apart descriptions that basically
    refer to the same thing.
    """
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")