from jenkspy import JenksNaturalBreaks
import numpy as np
import jenkspy
import seaborn
import matplotlib.pyplot as plt

def jenks_breaks(activations, numberOfBreaks, image_size):
    result = activations.flatten()
    jnb = JenksNaturalBreaks(numberOfBreaks)
    jnb.fit(result)
    innerbreaks = jnb.breaks_
    # print_jenks_data(jnb)
    activation_points_list_zero = []
    activation_points_list_category_1 = []
    activation_points_list_category_2 = []
    activation_points_list_category_3 = []

    y = -1
    for i in activations:
        y += 1
        x = -1
        for j in i:
            x += 1
            # print(manipulate[i][j])
            if (j < innerbreaks[1]):
                activation_points_list_zero.append((x, image_size - y))
            elif (j > innerbreaks[1] and j < innerbreaks[2]):
                activation_points_list_category_1.append((x, image_size - y))
            elif (j > innerbreaks[2] and j < innerbreaks[3]):
                activation_points_list_category_2.append((x, image_size - y))
            else:
                activation_points_list_category_3.append((x, image_size - y))

    activation_ranges = []
    activation_ranges.append(activation_points_list_category_1)
    activation_ranges.append(activation_points_list_category_2)
    activation_ranges.append(activation_points_list_category_3)
    return activation_ranges


def print_jenks_data(jnb):
    try:
        print(jnb.labels_)
        print(jnb.groups_)
        print(np.len(jnb.groups_[0]))
        print(jnb.inner_breaks_)
        print(jnb.breaks_)
    except:
        pass


def visualise_breaks(activations):

    result = activations.flatten()
    breaks = jenkspy.jenks_breaks(result, nb_class=5)
    print(breaks)

    plt.figure()
    seaborn.stripplot(x=result, jitter=True)
    seaborn.despine()
    locs, labels = plt.xticks()
    plt.xticks(locs, map(lambda x: x, np.round(locs,2)))
    plt.xlabel('Intensity')
    plt.yticks([])
    for b in breaks:
        plt.vlines(b, ymin=-0.2, ymax=0.5)

    plt.show()

