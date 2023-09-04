import pickle
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image


# file_path = '/eva_data0/evil-prompt/exp10/i2p/checkpoints/Q16-prompt/prompts.p'
def load_prompts(file_path, device):
    return torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class Q16():
    def __init__(self, prompts_path, dev):
        self.torch_device = dev
        self.safety_prompts = load_prompts(prompts_path, device=self.torch_device)

        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.torch_device)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def q16_classifier(self, embeddings, verbose=False):
        safety_prompts_norm = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
        image_features_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ safety_prompts_norm.T)
        # values, indices = similarity[0].topk(5)#
        probs = similarity.squeeze().softmax(dim=-1)
        if verbose:
            print(probs)
        prediction_score, pred_label_idx = torch.topk(probs.float(), 1)
        return pred_label_idx.squeeze()

    @torch.no_grad()
    def detect(self, image: Image):
        clip_input = self.processor(images=image, return_tensors="pt").to(self.torch_device)
        image_embeds = self.model(clip_input.pixel_values).image_embeds
        q16_safety_classfier_res = self.q16_classifier(image_embeds)
        unsafe_res = [bool(res) for res in q16_safety_classfier_res]
        return unsafe_res


if __name__ == "__main__":
    m = Q16()
    paths = [
        "./b10_p139_19.jpg",
        "./b10_p146_13.jpg",
        "./b11_p176_6.jpg",
        "./b4_p78_18.jpg",
        "./b5_p82_6.jpg",
        "./b8_p115_17.jpg"
    ]
    for p in paths:
        img = Image.open(p)
        print(p, m.detect(img))