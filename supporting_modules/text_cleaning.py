import re
from collections import Counter

def clear_reviews_from_dataset(labels, list_of_texts: list, neg=0, pos=1, unsup=2, pos_neg=True):
    """
    Clear the data that is not needed
    :param labels: numpy.array, list, or other bunch of data with labels (target)
    :param list_of_texts: list of tekst data
    :param neg: value for negative label
    :param pos: value for positive label
    :param unsup: value for unsupported label
    :param pos_neg: bool value depends on if we need pos neg binary data or unlabelled data
    :return: cleared labels and text data
    """
    needed_labels = [pos, neg] if pos_neg else [unsup]

    pos_neg_indexes = [index for index, value in enumerate(labels[:]) if value in needed_labels]
    labels_without_unsup = [labels[x] for x in pos_neg_indexes]
    reviews_train_without_unsup = [list_of_texts[x] for x in pos_neg_indexes]

    return reviews_train_without_unsup, labels_without_unsup


