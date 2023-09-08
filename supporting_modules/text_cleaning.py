import re
import string
import nltk
from nltk.corpus import stopwords

PATTERNS = {
    'starts_like_tag': re.compile(r"<+ "),
    'html_tags': re.compile(r'<[^>]+>'),
    'email_addr': re.compile(r'\S*@\S*(?=[\s.!?,]|$)'),
    'http_addr': re.compile(r'(http[s]?://|www[.])(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
    'digits': re.compile(r"\d+"),
    'rude_words': ('shit', 'bullshit', 'damn')
}

# dictionaries to replace negations
negations_without_t = {'don': 'do not', 'ain': 'not', 'aren': 'are not', 'couldn': 'could not',
             'didn': 'did not', 'doesn': 'does not', 'hadn': 'had not', 'hasn': 'has not',
             'haven': 'have not', 'isn': 'is not', 'mightn': 'might not', 'mustn': 'must not',
             'needn': 'need not', 'shan': 'shall not', 'shouldn': 'should not', 'wasn': 'was not',
             'weren': 'were not', 'won': 'will not', 'wouldn': 'would not'}

negations_with_t = {"don't": 'do not', "aren't": 'are not', "couldn't": 'could not', "didn't": 'did not',
             "doesn't": 'does not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not',
             "isn't": 'is not', "mightn't": 'might not', "mustn't": 'must not', "needn't": 'need not',
             "shan't": 'sha not', "shouldn't": 'should not', "wasn't": 'was not', "weren't": 'were not',
             "won't": 'wo not', "wouldn't": 'would not'}

# only for checking the statistics how frequently such kind of words appear in the reviews
rude_words = ('shit', 'bullshit', 'damn')


def get_occurance_in_text(text_data: str, pattern=PATTERNS['html_tags']) -> set:
    """
    Returns a set of elements that match the passed pattern. This is a set because of elimination redundant occurrences.
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


def clear_substr_in_texts(dataset: list, pattern=PATTERNS['html_tags'],
                          replace_with: str = "", sample: int = None, get_info: bool = True) -> list:
    """
    Removes indicated pattern in text elements
    :param dataset: list of strings or bytes
    :param pattern: pattern to search
    :param replace_with: string that should be applied for replacement
    :param get_info:
    :param sample: how many info about removed items you want to display if None all such items will be displayed
    :return: list of strings (if the list of bytes is passed they are converted to strings)
    """

    if get_info:
        removed_sample = get_occurance_in_dataset(dataset, pattern)
        if sample:
            removed_sample = list(removed_sample)[:sample]

        # show what data will be removed
        print("\nREMOVED SUBSTRINGS SAMPLE:\n", removed_sample)

    return [re.sub(pattern, replace_with,
                   txt.decode('utf-8') if isinstance(txt, bytes) else txt)
            for txt in dataset]


def clear_punctuation(text: str, replace_with: str | None = None):
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

    if isinstance(words_to_check, str):
        words_to_check = [words_to_check]

    for w in words_to_check:

        doc_frequency = sum(1 for doc in corpus if w in doc.lower())

        # w_cleared = clear_punctuation(w)

        matches = re.findall(re.compile(r"{}".format(w)), all_documents)

        total_frequency = len(matches)

        if doc_frequency > total_frequency:
            total_frequency = doc_frequency

        if w in PATTERNS['rude_words']:
            w = f'{w[:2]}#@%#'

        print(f"Word '{w}' occurs {total_frequency} times in the corpus.")
        print(f"Word '{w}' occurs in {doc_frequency} documents.\n")


def remove_stopwords(text_data: list|str, stop_words_list=None) -> list:
    """
    Fn gets list of string data -> Tokenizes texts and removes stop words
    :param text_data: liost of strings or single string but it will be transformed into a list
    :param stop_words_list: list of stopwords if None nltk stopwords list will be applied
    :return: List of strings with removed stopwords
    """
    if not stop_words_list:
        stop_words_list = stopwords.words("english")
    stop_words_list = set(stop_words_list)

    if isinstance(text_data, str):
        text_data = [text_data]

    return [
        " ".join([word for word in nltk.word_tokenize(txt) if word not in stop_words_list])
        for txt in text_data
    ]


def translate_shortcuts(dataset: list, dict_to_translate: dict):
    """
    fn takes list of strings and dictionary and replace the words in these strings - translates the keys
    (words from document) to their equivalent (values in dict)
    :param dataset: list of documents (strings)
    :param dict_to_translate: dictionaries of base word for searching in document (key) and its translation (val)
    :return: list of translated documents
    """
    pattern = r"\b(?:{})\b".format('|'.join(re.escape(word) for word in dict_to_translate.keys()))
    print(f"Applied pattern => {pattern}")

    new_dataset = [
                    re.sub(pattern, lambda match: dict_to_translate.get(match.group(0), match.group(0)), text)
                    for text in dataset
                   ]

    return new_dataset
