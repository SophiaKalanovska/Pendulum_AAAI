import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def mask_to_input_relevance_of_mask(masks, a, image_name):
    continues_regions = []
    for i in range(len(masks)):
        if (i == 0):
            continue
        else:
            mask = masks[i]
            masked_heat_show = a * mask
            masked_heat = tf.constant(a * mask, tf.float32)
            continues_regions.append(masked_heat)
            masked_heat_show = masked_heat_show.sum(axis=np.argmax(np.asarray(masked_heat_show.shape) == 3))
            masked_heat_show /= np.max(np.abs(masked_heat_show))
            plt.imshow(masked_heat_show[0], cmap="seismic", clim=(-1, 1))
            plt.savefig(image_name + "_masked_heat.png")
            plt.show()
    return continues_regions