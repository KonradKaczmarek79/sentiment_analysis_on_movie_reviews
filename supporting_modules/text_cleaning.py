import re
from collections import Counter
import string

PATTERNS = {
            'html_tags': re.compile(r'<[^>]+>'),
            'email_addr': re.compile(r"\S*@\S*\s?"),
            'http_addr': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'obscene_words': ('shit', 'dick',  'bullshit'),
            'digits_in_substr': re.compile(r"\b\S*[0-9]+\S*\s?\b")
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
    # show what data will be removed
    print("\nREMOVED SUBSTRINGS:\n", get_occurance_in_dataset(dataset, pattern))

    return [re.sub(pattern, replace_with,
                   txt.decode('utf-8') if isinstance(txt, bytes) else txt)
            for txt in dataset]

def clear_punctuation(text: str, replace_with: str|None=None):
    """
    this function uses a well-known way to remove punctutation characters
    :param replace_with:
    :param text: text data to apply the replacement punctuation sign into None
    :return: string with punctuation signs replaced with 'replace_with' value
    """
    # create table where key will be punctuation signs and vals will be Nones
    table = str.maketrans({key: replace_with for key in string.punctuation})
    # clear all punctuation signs from whole text
    return text.translate(table)


def corpus_docs_word_frequency(corpus: list, words_to_check: str | list):
    """
    Function returns info about each word from corpus occurrence in whole corpus and how many documents contain this word
    :param corpus: list of documents
    :param words_to_check:
    :return: None (displays info about each word occurrence)
    """
    all_documents = ' '.join(corpus).lower()
    all_documents_cleaned = clear_punctuation(all_documents)

    words = all_documents.split()
    # word_counts_ = Counter(words)

    if isinstance(words_to_check, str):
        words_to_check = [words_to_check]

    for w in words_to_check:

        # total_frequency_ = word_counts[w]

        doc_frequency = sum(1 for doc in corpus if w in doc.lower())

        # w_cleared = clear_punctuation(w)

        matches = re.findall(re.compile(r"{}".format(w)), all_documents)

        total_frequency = len(matches)

        if doc_frequency > total_frequency:
            total_frequency = doc_frequency

        if w in PATTERNS['obscene_words']:
            w = f'{w[:2]}#@%#'

        print(f"Word '{w}' occurs {total_frequency} times in the corpus.")
        print(f"Word '{w}' occurs in {doc_frequency} documents.\n")