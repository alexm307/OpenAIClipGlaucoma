import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

import config as CFG
from CLIP import CLIPModel
import math

def test_model(valid_loader, model_path):
    """
    Main function meant for testing the model's accuracy in terms of selecting the proper class for the patient image
    It takes the testing images and selects an appropriate class depending on the closest class embedding.
    """

    classes = ['healthy', 'glaucoma', 'suspicious']

    #Now I have to get the validation images embeddings, then see which of them actually match the embeddings of the
    # appropriate class name
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    valid_image_embeddings = []
    valid_img_classes = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)

            # adding both the embedding and the image caption to the lists, to check if the embeddings most matching
            # are indeed the right ones
            valid_image_embeddings.append(image_embeddings)
            valid_img_classes.append(batch["caption"])

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    # getting the word embedding of each and every class name
    text_embeddings = []
    for query in classes:
        encoded_query = tokenizer([query])
        batch = {
            key: torch.tensor(values).to(CFG.device)
            for key, values in encoded_query.items()
        }
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            text_embeddings.append(model.text_projection(text_features))

    # comparing every validation image embedding with every class embedding to find the closest match

    i = 0
    correct = 0

    for image_embeddings in valid_image_embeddings:

        #turning the lists into tensors
        #valid_image_embeddings = torch.cat(valid_image_embeddings)
        text_embeddingss = torch.cat(text_embeddings)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddingss, p=2, dim=-1)

        # Computing all dot similarities, to find out the actual best match for each image
        dot_similarity = image_embeddings_n @ text_embeddings_n.T

        values, indices = torch.topk(dot_similarity.squeeze(0), 3)

        j = 0
        for x in valid_img_classes:
            if x == classes[indices[j][0]]:
                print(x, classes[indices[j][0]])
                correct += 1
                j += 1


        #print(torch.topk(dot_similarity.squeeze(0), 3))
        #print(valid_img_classes[i])

        i += 1
        print(i)

    print(correct)

    return model, torch.cat(valid_image_embeddings)




def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    # multiplying by 5 to consider that there are 5 captions for a single image
    # so in indices, the first 5 indices point to a single image, the second 5 indices
    # to another one and so on.
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(math.sqrt(n), math.sqrt(n), figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()