# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import \
    absolute_import, print_function, division, unicode_literals

from importlib.machinery import SourceFileLoader
# from sklearn.preprocessing import StandardScaler
# from tensorflow.python.framework.ops import enable_eager_execution_internal

import innvestigate.utils
import tensorflow.keras.applications.inception_v3 as inception
import tensorflow.keras.applications.inception_v3 as inception_v3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import time
from clusterRelevance import one_dimensional_breaks
from clusterRelevance import two_dimensional_clusters
from featureToNeurons import mask_to_relevance


###############################################################################
###############################################################################

base_dir = os.path.dirname(__file__)
utils = SourceFileLoader("utils", os.path.join(base_dir, "utils.py")).load_module()

###############################################################################
###############################################################################

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #
    # Load an image. Need to download examples images first. See script in images directory.
    #
    image_name = "tiger"
    image_size = 299
    image = utils.load_image(
        os.path.join(base_dir, "images", image_name + ".png"), image_size)
    image = image[:, :, :3]
    # image = np.expand_dims(image, axis=0)

    # Get model
    model, preprocess = inception.InceptionV3(), inception.preprocess_input

    # Strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model)

    # Add batch axis and preprocess
    x = preprocess(image[None])

    pr = model.predict_on_batch(x)
    predictions = inception.decode_predictions(pr)
    print(predictions)
    # distribute the relevance to the input layer
    start_time = time.time()
    a, tensors_Xs = analyzer.analyze(x)
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    # Aggregate along color channels and normalize to [-1, 1]
    b = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    b /= np.max(np.abs(b))

    # Plot
    plt.imshow(b[0], cmap="seismic", clim=(-1, 1))
    plt.savefig(image_name + "_heatmap.png")
    plt.show()

    # Find the activation ranges
    categories = one_dimensional_breaks.jenks_breaks(b[0], 5, image_size)

    # Visualise the activation ranges
    one_dimensional_breaks.visualise_breaks(b[0])

    # Create masks for each cluster found on the level with highest activation
    masks = two_dimensional_clusters.DbSCAN_for_activations(categories, image_size)

    # heatmaps = mask_to_relevance.mask_to_input_relevance_of_mask(masks, a, image_name)

    forward_analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model)
    masks_relevances = []
    for mask in masks:
        masks_relevances.append(analyzer.propagate_forward(x, mask))
    print(masks_relevances)


