import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torch
import glob


class ResNet18():
    def __init__(self, ckpt_path, dev):
        # load pre-trained model
        self.torch_device = dev
        model = models.resnet18(weights=None)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 10)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['net'])  
        self.model = model.to(self.torch_device)
        self.model.eval()

        # define image processor
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize([500, 500]),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        

    @torch.no_grad()
    def detect(self, images: Image):
        trans_images = []
        for image in images:
            trans_images.append(self.transform(image))
        input_tensor = torch.stack(trans_images, dim=0).to(self.torch_device)
        y = self.model(input_tensor)
        pred = torch.argmax(y, dim=1)
        return pred
        


if __name__ == "__main__":
    m = ResNet18("/eva_data0/evil-prompt/pretrained/ResNet18 0.945223.pth", "cuda:0")
    dir_path = "/eva_data0/french_horn"

    imgs = []
    for p in glob.glob(dir_path + "/*.jpg"):
        img = Image.open(p)
        imgs.append(img)
    print(m.detect(imgs))

    is_harm = any([x == 5 for x in m.detect(imgs)])
    unsafe = bool(is_harm)
    print(f"unsafe: {unsafe}")