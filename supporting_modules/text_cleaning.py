import re
from collections import Counter

PATTERNS = {
                'html_tags': re.compile(r'<[^>]+>'),
            }

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


def get_occurance_in_text(text_data: str, pattern=PATTERNS['html_tags']) -> set:
    """
    Returns a list of elements that match the passed pattern
    :param text_data: text data to search in
    :param pattern: regex pattern
    :return: set of elements found in passed text
    """
    tags = pattern.findall(text_data)

    return set(tags)


def get_occurance_in_dataset(dataset: list, pattern=PATTERNS['html_tags']) -> set:
    """
    Returns list of elements that match to the pattern
    :param pattern: regex pattern
    :param dataset: list of texts to search in
    :return: set of elements found in passed texts
    """
    list_of_tags = []

    for x in dataset:
        if isinstance(x, bytes):
            x = x.decode('utf-8')
        list_of_tags.extend(get_occurance_in_text(x, pattern))

    return set(list_of_tags)


def clear_substr_in_texts(dataset: list, pattern=PATTERNS['html_tags'], replace_with: str = "") -> list:
    """
    Removes indicated pattern in text elements
    :param replace_with: string that should be applied for replacement
    :param dataset: list of strings or bytes
    :param pattern: pattern to search
    :return: list of strings (if the list of bytes is passed they are converted to strings)
    """
    return [re.sub(pattern, replace_with,
                   txt.decode('utf-8') if isinstance(txt, bytes) else txt)
            for txt in dataset]
