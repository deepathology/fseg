from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
import torch
from sklearn.decomposition import NMF, non_negative_factorization, MiniBatchNMF
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances

"""
Segmentation by Factorization: Unsupervised Semantic Segmentation for Pathology by Factorizing Foundation Model Features
"""

def dff(activations: np.ndarray, n_components: int = 5):
    """Compute Deep Feature Factorization on a 2d Activations tensor.

    :param activations: A numpy array of shape batch x channels x height x width
    :param n_components: The number of components for the non negative matrix factorization
    :return: A tuple of the concepts (a numpy array with shape channels x components),
              and the explanation heatmaps (a numpy arary with shape batch x height x width)

    """
    batch_size, __, h, w = activations.shape
    reshaped_activations = activations.transpose((1, 0, 2, 3))
    reshaped_activations[np.isnan(reshaped_activations)] = 0
    reshaped_activations = reshaped_activations.reshape(
        reshaped_activations.shape[0], -1)
    reshaped_activations = reshaped_activations
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(reshaped_activations)
    H = model.components_
    concepts = W
    explanations = H.reshape(n_components, batch_size, h, w)
    explanations = explanations.transpose((1, 0, 2, 3))

    return concepts, explanations


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


class FSeg:
    """
    Segmentation by Factorization:
    Unsupervised Semantic Segmentation for Pathology by Factorizing Foundation Model Features

    :param model: The model to be used for generating activations.
    :param target_layer: The layer of the model from which activations are extracted.
    :param n_concepts: The number of concepts for the non-negative matrix factorization.
    :param reshape_transform: A function to reshape the activations, defaults to None.
    :param random_state: Random state for reproducibility, defaults to 0.

    """

    def __init__(
            self,
            model,
            target_layer,
            reshape_transform: Callable = None,
            random_state: int = 0,
    ):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self.random_state = random_state

        self.activations_and_grads = ActivationsAndGradients(
            self.model, [self.target_layer], self.reshape_transform)

    def fit_predict(self, input_tensor: torch.tensor) -> np.ndarray:
        """
        Fit the model on the input tensor and predict the segmentation.

        :param input_tensor: The input tensor.
        :return: The predicted segmentation as a numpy array.

        """
        self.fit(input_tensor=input_tensor)
        return self.predict(input_tensor)

    def get_activations(self, input_tensor: torch.tensor) -> np.ndarray:
        with torch.no_grad():
            self.activations_and_grads(input_tensor)
            activations = self.activations_and_grads.activations[0].cpu(
            ).numpy()

        activations = activations[0].transpose((1, 2, 0))
        return activations

    def predict_clustering(
            self,
            input_tensor: torch.tensor,
            clustering_model: ConceptClustering,
            k: int = 20) -> np.ndarray:
        """
        Predict the segmentation for the given input tensor with the given clustering model.

        :param input_tensor: The input tensor.
        :param clustering_model: The clustering model to use.
        :param k: The number of concepts for NMF dimension.
        :return: The predicted clustering as a numpy array.

        """
        activations = self.get_activations(input_tensor)

        component_concepts, w = dff(activations, k)
        component_concepts = component_concepts.transpose()
        labels = {}
        for i in range(len(component_concepts)):
            labels[i] = clustering_model(component_concepts[i, :][None, :])

        w_for_resize = torch.from_numpy(w)
        size = (input_tensor.shape[2], input_tensor.shape[3])
        w_resized = torch.nn.functional.interpolate(w_for_resize, size, mode='bilinear')[
            0].numpy().transpose((1, 2, 0))

        w_resized = w_resized.argmax(axis=-1)
        segmentation = np.array(
            Image.fromarray(
                np.uint8(w_resized)))

        converted_segmentation = segmentation.copy()
        for i in labels:
            converted_segmentation[segmentation == i] = labels[i]
        return converted_segmentation

    def predict_clustering_without_factorization(
            self,
            input_tensor: torch.tensor,
            clustering_model: ConceptClustering,
            k: int = 20) -> np.ndarray:
        """
        Predict the segmentation for the given input tensor with the given clustering model
        without factorization.

        :param input_tensor: The input tensor.
        :param clustering_model: The clustering model to use.
        :param k: The number of concepts for NMF dimension.
        :return: The predicted clustering as a numpy array.

        """
        activations = self.get_activations(input_tensor)[0]

        activations = activations.transpose((1, 2, 0))
        activations_reshaped = activations.reshape(-1, activations.shape[-1])
        segmentation_reshaped = clustering_model(activations_reshaped)
        segmentation = segmentation_reshaped.reshape(
            activations.shape[0], activations.shape[1]).astype(np.float32)
        segmentation_for_resize = torch.from_numpy(
            segmentation).unsqueeze(0).unsqueeze(0)
        size = (input_tensor.shape[2], input_tensor.shape[3])
        segmentation_resized = torch.nn.functional.interpolate(segmentation_for_resize, size, mode='bilinear')[
            0].numpy().transpose((1, 2, 0))[:, :, 0]

        return segmentation_resized.astype(np.uint8)

    def predict_project_concepts(
            self,
            input_tensor: torch.tensor,
            concepts: np.ndarray) -> np.ndarray:
        """
        Predict the segmentation for the given input tensor with the given concepts.

        :param input_tensor: The input tensor.
        :param concepts: The concepts to use for the projection.
        :return: The predicted segmentation as a numpy array.

        """
        concepts[concepts < 0] = 0
        activations = self.get_activations(
            input_tensor).transpose((0, 2, 3, 1))
        vector = activations.reshape(-1, activations.shape[-1])
        w, __, __ = non_negative_factorization(
            X=vector,
            H=concepts,
            W=None,
            n_components=len(concepts),
            update_H=False,
            random_state=self.random_state,
            max_iter=10000,
        )

        w = w.reshape((activations.shape[1], activations.shape[2], -1))
        w_for_resize = torch.tensor(
            w.transpose(
                (2, 0, 1))[
                None, :, :, :])  # Add batch dimension
        size = (input_tensor.shape[2], input_tensor.shape[3])
        w_resized = torch.nn.functional.interpolate(w_for_resize, size, mode='bilinear')[
            0].numpy().transpose((1, 2, 0))

        w_resized = w_resized.argmax(axis=-1)
        segmentation = np.array(
            Image.fromarray(
                np.uint8(w_resized)))
        return segmentation

    def predict_on_single_image(
            self,
            input_tensor: torch.tensor,
            k: int) -> np.ndarray:
        """
        Predict the segmentation for the given input tensor.

        :param input_tensor: The input tensor.
        :param k: Number of concepts for NMF dimension.
        :return: The predicted segmentation as a numpy array and its corresponsing concepts.

        """
        activations = self.get_activations(input_tensor)

        component_concepts, w = dff(activations, k)
        component_concepts = component_concepts.transpose()

        w_for_resize = torch.from_numpy(w)
        size = (input_tensor.shape[2], input_tensor.shape[3])
        w_resized = torch.nn.functional.interpolate(w_for_resize, size, mode='bilinear')[
            0].numpy().transpose((1, 2, 0))

        w_resized = w_resized.argmax(axis=-1)
        segmentation = np.array(
            Image.fromarray(
                np.uint8(w_resized)).resize(
                (input_tensor.shape[3], input_tensor.shape[2])))

        return segmentation, component_concepts

    def partial_fit(self, input_tensor: torch.tensor) -> None:
        """
        Fit the model on the input tensor to compute the concepts.

        :param input_tensor: The input tensor.

        """
        activations = self.get_activations(input_tensor)
        reshaped_activations = activations.transpose((1, 0, 2, 3))
        reshaped_activations[np.isnan(reshaped_activations)] = 0
        reshaped_activations = reshaped_activations.reshape(
            reshaped_activations.shape[0], -1).transpose()
        try:
            _ = self.nmf_model
        except BaseException:
            self.nmf_model = MiniBatchNMF(
                n_components=self.n_concepts,
                init='random',
                random_state=self.random_state)

        self.nmf_model.partial_fit(reshaped_activations)
        self.concepts = self.nmf_model.components_

    def get_activations(self, input_tensor: torch.tensor) -> np.ndarray:
        """
        Get the activations for the given input tensor.

        :param input_tensor: The input tensor.
        :return: The activations as a numpy array.

        """
        with torch.no_grad():
            self.activations_and_grads(input_tensor)
            return self.activations_and_grads.activations[0].cpu().numpy()

    def fit(self, input_tensor: torch.tensor) -> None:
        """
        Fit the model on the input tensor to compute the concepts.

        :param input_tensor: The input tensor.

        """
        activations = self.get_activations(input_tensor)
        self.concepts, __ = dff(activations, self.n_concepts)
        self.concepts = self.concepts.transpose()
