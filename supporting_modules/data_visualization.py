import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def count_per_label(labels, names):

    # length and height of the figure
    fig_dimensions = len(labels) * 2 + 1, len(labels) + 1
    plt.rcParams['figure.figsize'] = (fig_dimensions[0], fig_dimensions[1])
    colors = ('Blue', 'Red', 'Green', 'Yellow', 'Black')
    value_quantity_range = tuple(range(len(labels)))
    for label, name, nr in zip(labels, names, value_quantity_range):
        plt.bar(nr, label, width=0.5, label=name, color=colors[nr])
        plt.legend()

    plt.ylabel('Quantity')
    plt.xlabel('Review type')
    plt.show()


def make_count_histogram(list_of_texts, d=None):

    words_per_text = np.array([len(text.split()) for text in list_of_texts])

    # array size
    n = words_per_text.size

    right_boundary = np.sqrt(n)
    left_boundary = 0.75 * right_boundary

    min_quantity = words_per_text.min()
    max_quantity = words_per_text.max()

    if d is None:
        d = int(np.round((max_quantity - min_quantity) / np.mean((right_boundary, left_boundary))))

    # numbers that will be bounds of our ranges
    ends_of_ranges = np.arange(min_quantity, max_quantity, d)

    # Counting the occurrence frequency of words
    word_counts = Counter(words_per_text)
    k = int(np.round(np.mean((right_boundary, left_boundary))))

    plt.figure(figsize=(16, 6))
    hist_data = plt.hist(words_per_text, bins=k, width=10, align='mid')
    plt.xticks(list(ends_of_ranges) + [max_quantity], rotation=60, ha='right')
    plt.title('Words per review histogram')
    plt.ylabel('Quantity of rewiews')
    plt.xlabel('Word quantity in single review')
    plt.show()

    return word_counts
