#!/usr/bin/env python
# SBATCH --job-name "TEST"
# SBATCH --error slurm.%j.err
# SBATCH --output slurm.%j.out

from dff_seg.dff_seg import DFFSeg, show_segmentation_on_image
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image
from torchvision.models import resnet50
from functools import partial
import timm
from huggingface_hub import login
import torch
import time
import json
import sys
import tqdm
from torchvision import transforms


class Dataset:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.paths = json.load(f)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.image_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])
        self.bad_indices = set()

    def return_other(self, index):
        if index > 0:
            return self.__getitem__(index - 1)
        else:
            return self.__getitem__(len(self.paths) - 1)

    def __getitem__(self, index):
        image_path = self.paths[index]
        return self.image_transform(Image.open(image_path))

    def __len__(self):
        return len(self.paths)


def uni_model_transform(tensor, width, height):
    result = torch.nn.ReLU()(tensor[:, 1:, :].reshape(tensor.size(0),
                                                      height,
                                                      width,
                                                      tensor.size(2)))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


model_name = sys.argv[2]
token = "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"
login(token=token)

if model_name == "resnet50":
    model = resnet50(pretrained=True)
    target_layer = model.layer4
    reshape_transform = None

elif model_name == "resnet50_layer3":
    model = resnet50(pretrained=True)
    target_layer = model.layer3
    reshape_transform = None

elif model_name == "gigapath":
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True)
    target_layer = model.blocks[-1]
    reshape_transform = partial(
        uni_model_transform,
        width=16,
        height=16,
    )
elif model_name == "uni":
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True)
    target_layer = model.blocks[-1]
    reshape_transform = partial(
        uni_model_transform,
        width=16,
        height=16,
    )


model = model.cuda()
model.eval()
unsupervised_seg = DFFSeg(
    model=model,
    target_layer=target_layer,
    n_concepts=64,
    reshape_transform=reshape_transform
)

dataset = Dataset(sys.argv[1])
loader = torch.utils.data.DataLoader(
    dataset, batch_size=128, num_workers=2, shuffle=True)


for epoch in range(10):
    for index, batch in enumerate(tqdm.tqdm(loader)):
        unsupervised_seg.partial_fit(batch.cuda())

        if index % 100 == 0:
            np.save(
                f"concepts_{model_name}_64_concepts.npy",
                unsupervised_seg.concepts)
