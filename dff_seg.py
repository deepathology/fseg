from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
import torch
import denseCRF
from sklearn.decomposition import NMF, non_negative_factorization, MiniBatchNMF
from PIL import Image


def show_segmentation_on_image(
        img: np.uint8,
        segmentation: np.ndarray,
        colors: list[np.ndarray] = None,
        n_categories: int = None,
        image_weight: float = 0.5
) -> np.ndarray:
    """Color code the different component heatmaps on top of the image.

    Since different factorization component heatmaps can overlap in principle,
    we need a strategy to decide how to deal with the overlaps.
    This keeps the component that has a higher value in its heatmap.

    :param img: The base image in RGB format.
    :param segmentation: A numpy array with category indices per pixel.
    :param colors: List of R, G, B colors to be used for the components.
                   If None, will use the gist_rainbow colormap as a default.
    :param n_categories: Number of categories in the segmentation.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * visualization.
    :return: The visualized image.

    """
    float_img = np.float32(img) / 255
    categories = list(range(n_categories))
    if colors is None:
        # taken from https://github.com/edocollins/DFF/blob/master/utils.py
        _cmap = plt.cm.get_cmap('gist_rainbow')
        colors = [
            np.array(
                _cmap(i)) for i in np.arange(
                0,
                1,
                1.0 /
                len(categories))]

    mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))
    for category in categories:
        mask[segmentation == category] = colors[category][:3]

    result = float_img * image_weight + mask * (1 - image_weight)
    result = np.uint8(result * 255)

    return result


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


def densecrf(
        I: np.ndarray,
        P: np.ndarray,
        params: tuple[float, float, float, float, float, int],
) -> np.ndarray:
    """Applying densecrf.

    :param I: A numpy array of shape [H, W, C], where C should be 3.
               type of I should be np.uint8, and the values are in [0, 255]
    :param P: A probability map of shape [H, W, L], where L is the number of classes
               type of P should be np.float32
    :param params: A tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it).
    :return: A numpy array of shape [H, W], where pixel values represent class indices.

    """
    out = denseCRF.densecrf(I, P, params)
    return out


def densecrf_on_image(
        image: np.ndarray,
        prob: np.ndarray,
        w1: float = 10.0,
        w2: float = 3.0,
        alpha: float = 80,
        beta: float = 13,
        gamma: float = 3,
        it: int = 5,
) -> np.ndarray:
    """Applying densecrf on image, given the segmentation probabilities.

    :param iamge: Input rgb image.
    :param prob: Probability mask.
    :param w1: Weight of bilateral term, e.g. 10.0
    :param alpha: Spatial distance std, e.g., 80
    :param beta: Rgb value std, e.g., 15
    :param w2: Weight of spatial term, e.g., 3.0
    :param gamma: Spatial distance std for spatial term, e.g., 3
    :param it: Iteration number, e.g., 5
    :return: A numpy array of shape [H, W], where pixel values represent class indices.

    """
    I = image
    Iq = np.asarray(I)
    prob = prob / prob.sum(axis=-1)[:, :, None]

    param = (w1, alpha, beta, w2, gamma, it)
    lab = densecrf(Iq, prob, param)
    lab = np.array(Image.fromarray(lab))
    return lab


class DFFSeg:
    """
    A class to perform Deep Feature Factorization (DFF) based segmentation on images.

    :param model: The model to be used for generating activations.
    :param target_layer: The layer of the model from which activations are extracted.
    :param n_concepts: The number of concepts for the non-negative matrix factorization.
    :param reshape_transform: A function to reshape the activations, defaults to None.
    :param random_state: Random state for reproducibility, defaults to 0.
    :param crf_smoothing: Whether to apply CRF smoothing, defaults to False.
    :param w1: Weight of the bilateral term for CRF, defaults to 10.0.
    :param w2: Weight of the spatial term for CRF, defaults to 3.0.
    :param alpha: Spatial distance std for CRF, defaults to 80.
    :param beta: RGB value std for CRF, defaults to 13.
    :param gamma: Spatial distance std for the spatial term in CRF, defaults to 3.
    :param it: Iteration number for CRF, defaults to 5.

    """

    def __init__(
            self,
            model,
            target_layer,
            n_concepts: int,
            reshape_transform: Callable = None,
            random_state: int = 0,
            crf_smoothing: bool = False,
            w1: float = 10.0,
            w2: float = 3.0,
            alpha: float = 80,
            beta: float = 13,
            gamma: float = 3,
            it: int = 5,
    ):
        self.model = model
        self.target_layer = target_layer
        self.n_concepts = n_concepts
        self.reshape_transform = reshape_transform
        self.random_state = random_state
        self.crf_smoothing = crf_smoothing
        self.w1 = w1
        self.w2 = w2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.it = it

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

    def predict(self, input_tensor: torch.tensor):
        """
        Predict the segmentation for the given input tensor.

        :param input_tensor: The input tensor.
        :return: The predicted segmentation as a numpy array.

        """
        with torch.no_grad():
            self.activations_and_grads(input_tensor)
            activations = self.activations_and_grads.activations[0].cpu(
            ).numpy()

        activations = activations[0].transpose((1, 2, 0))
        vector = activations.reshape(-1, activations.shape[-1])

        w, __, __ = non_negative_factorization(
            X=vector,
            H=self.concepts.transpose(),
            W=None,
            n_components=self.n_concepts,
            update_H=False,
            random_state=self.random_state,
            max_iter=10000,
        )

        w = w.reshape((activations.shape[0], activations.shape[1], -1))
        w_for_resize = torch.tensor(
            w.transpose(
                (2, 0, 1))[
                None, :, :, :])  # Add batch dimension
        size = (input_tensor.shape[2], input_tensor.shape[3])
        w_resized = torch.nn.functional.interpolate(w_for_resize, size, mode='bilinear')[
            0].numpy().transpose((1, 2, 0))

        if self.crf_smoothing:
            segmentation = densecrf_on_image(
                np.uint8(
                    input_tensor[0].cpu().numpy()).transpose(
                    1, 2, 0), w_resized)
        else:
            w_resized = w_resized.argmax(axis=-1)
            segmentation = np.array(
                Image.fromarray(
                    np.uint8(w_resized)).resize(
                    (input_tensor.shape[3], input_tensor.shape[2])))
        return segmentation

    def partial_fit(self, input_tensor: torch.tensor) -> None:
        activations = self.get_activations(input_tensor)
        batch_size, __, h, w = activations.shape
        reshaped_activations = activations.transpose((1, 0, 2, 3))
        reshaped_activations[np.isnan(reshaped_activations)] = 0
        reshaped_activations = reshaped_activations.reshape(
            reshaped_activations.shape[0], -1)
        reshaped_activations = reshaped_activations
        try:
            _ = self.nmf_model
            print("re-using model")
        except BaseException:
            self.nmf_model = MiniBatchNMF(
                n_components=self.n_concepts,
                init='random',
                random_state=self.random_state)
            print("init mininbatch model")

        self.nmf_model.partial_fit(reshaped_activations)
        self.concepts = self.nmf_model.components_

    def get_activations(self, input_tensor: torch.tensor) -> np.ndarray:
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
