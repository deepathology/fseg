#!/usr/bin/env python
# SBATCH --job-name "TEST"
# SBATCH --error slurm.%j.err
# SBATCH --output slurm.%j.out

from dataset import Dataset
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

if model_name == "resnet50":
    model = resnet50(pretrained=True)
    target_layer = model.layer4
    reshape_transform = None
elif model_name == "uni":
    token = "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"
    login(token=token)
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
    dataset, batch_size=32, num_workers=2, shuffle=True)


for epoch in range(10):
    for index, batch in enumerate(tqdm.tqdm(loader)):
        unsupervised_seg.partial_fit(batch.cuda())

        if index % 100 == 0:
            np.save(
                f"concepts_{model_name}_concepts.npy",
                unsupervised_seg.concepts)
