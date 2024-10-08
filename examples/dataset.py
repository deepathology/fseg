import json
from torchvision import transforms
from PIL import Image


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
