import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image
from torchvision.models import resnet50
from functools import partial
import timm
from huggingface_hub import login
import torch

from dff_seg import DFFSeg, show_segmentation_on_image


def uni_model_transform(tensor, width, height):
    result = torch.nn.ReLU()(tensor[:, 1:, :].reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2)))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    print(f"result: {result.shape}")
    return result


NUM_CONCEPTS = 20
img_path = r"D:\BCSSDataset\images\TCGA-A1-A0SP-DX1_xmin6798_ymin53719_MAG-10.00.png"
img = np.array(Image.open(img_path))
img = img[:16*(img.shape[0] // 16), :16*(img.shape[1] // 16), :]
rgb_img_float = np.float32(img) / 255
input_tensor = preprocess_image(rgb_img_float,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
img_path2 = r"D:\BCSSDataset\images\TCGA-A2-A0CM-DX1_xmin18562_ymin56852_MAG-10.00.png"
img2 = np.array(Image.open(img_path2))
img2 = img[:16*(img2.shape[0] // 16), :16*(img2.shape[1] // 16), :]
rgb_img_float2 = np.float32(img2) / 255
model_name = "uni"

if model_name == "resnet":
    model = resnet50(pretrained=True)
    target_layer = model.layer3
    reshape_transform = None
elif model_name == "uni":
    token = "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"
    login(token=token)
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    target_layer = model.blocks[-1]
    reshape_transform = partial(
        uni_model_transform,
        width=input_tensor.shape[3]//16,
        height=input_tensor.shape[2]//16,
    )

model = model.cuda()
model.eval()
unsupervised_seg = DFFSeg(
    model=model,
    target_layer=target_layer,
    n_concepts=NUM_CONCEPTS,
    reshape_transform=reshape_transform
)

input_tensor = input_tensor.cuda()
unsupervised_seg.fit(input_tensor)
segmentation = unsupervised_seg.predict(input_tensor=input_tensor)
unsupervised_seg.crf_smoothing = True
segmentation2 = unsupervised_seg.predict(input_tensor=input_tensor)


visualization = show_segmentation_on_image(
    img=img,
    segmentation=segmentation,
    image_weight=0.7,
    n_categories=NUM_CONCEPTS,
)

visualization2 = show_segmentation_on_image(
    img=img2,
    segmentation=segmentation2,
    image_weight=0.7,
    n_categories=NUM_CONCEPTS,
)