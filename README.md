# DFF-Seg: Deep Feature Factorization for Image Segmentation

DFF-Seg is a Python library that implements Deep Feature Factorization (DFF) for image segmentation tasks. This method leverages the power of deep neural networks and non-negative matrix factorization to produce meaningful segmentations of images.

## Installation

To install DFF-Seg, you can use pip:

```bash
pip install -e .
```

## Usage

To use DFF-Seg in your project, follow these steps:

1. Import the library:

```python
from dff_seg import DFFSeg
```

2. Load a pre-trained model and prepare the input image:

```python
model = resnet50(pretrained=True)
target_layer = model.layer3
reshape_transform = None

unsupervised_seg = DFFSeg(
    model=model,
    target_layer=target_layer,
    reshape_transform=reshape_transform
)

img_path = "./notebooks/images/example_3.png"
img = np.array(Image.open(img_path))[:, :, :3]
orig_shape = img.shape
rgb_img_float = np.float32(img) / 255
input_tensor = preprocess_image(rgb_img_float,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

```

3. Load pre-computed model embeddings:

```python
model_embeddings = np.load("path/to/model_embeddings.npy")
```

4. Define the number of clusters and prepare the concepts:

```python
k = 16  # Number of clusters
concepts = model_embeddings[k]
```

5. Generate the segmentation prediction:

* Using concepts projection:
```python
segmentation_prediction = unsupervised_seg.predict_project_concepts(input_tensor, concepts)
```

* Using concepts similarity:

class ConceptClustering:
    def __init__(self, clusters: np.ndarray):
        """
        Clustering model based on the cosine similarity between the concept
        embeddings.

        :param clusters: The clusters centroids embeddings.

        """
        self.clusters = clusters

    def __call__(self, vector: np.ndarray) -> int:
        return cosine_distances(vector, self.clusters).argmin()

```python
clustering_model = ConceptClustering(concepts)
segmentation_prediction = unsupervised_seg.predict_clustering(input_tensor, clustering_model)
```

## License and Terms of Use

This project is licensed under the Apache License, Version 2.0. You can find the full text of the license in the LICENSE file in the root directory of this project or at http://www.apache.org/licenses/LICENSE-2.0

Please ensure that you comply with the terms of the license when using, modifying, or distributing this software.


## Citation

If you use this code in your research, please cite using this BibTeX:

```
@article{gildenblat2024segmentation,
  title={Segmentation by Factorization: Unsupervised Semantic Segmentation for Pathology by Factorizing Foundation Model Features},
  author={Gildenblat, Jacob and Hadar, Ofir},
  journal={arXiv preprint arXiv:2409.05697},
  year={2024}
}
```
