import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Image encoder using ResNet50 pretrained on ImageNet 
def get_image_encoder():
    """
    Loads ResNet50 pretrained on ImageNet
    and removes classification head.
    Output: 2048-d feature vector per image.
    """

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"   # global average pooling → feature vector
    )

    return base_model


def encode_images(encoder, images):
    """
    images: numpy array (N, 224, 224, 3)
    returns: (N, 2048)
    """

    features = encoder.predict(images, batch_size=16, verbose=1)
    return features


def extract_slices(scan):
    """
    Converts 3D CT → list of 2D images
    """

    slices = []

    for i in range(scan.shape[2]):
        slc = scan[:, :, i]

        # normalize
        slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)

        # add channel dim → (H,W,1)
        slc = slc[..., np.newaxis]

        slices.append(slc)

    return np.array(slices)