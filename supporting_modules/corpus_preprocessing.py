import supporting_modules.text_cleaning as stc
import nltk

# customized nltk stopword list without negation shortcuts
customized_stop_words = ['he', 'by', 'below', 'over', 'yourselves', 'himself', 'were', "you've", 'these', 'are',
                         'itself', 'm', 'herself', 't', 'at', "should've", 'ourselves', 'doing', 'why', 'all',
                         'there', 'o', 'our', 'has', 'because', 'nor', 'a', 'own', 'now', 'll', 'ours', 'be', 'but',
                         'here', 've', 'what', 'you', 'until', 'not', 'when', 'few', 'again', 'once', 'out', 'his',
                         'while', 'who', 'under', 'further', 'them', 'if', 'up', 'is', 'she', 'been', 'will', 'as',
                         'or', 'had', 'very', 'too', 'can', 'and', "she's", 'being', 'd', 'themselves', 'your', 'to',
                         'during', 'each', 'we', 'should', 'after', 'did', 'the', 'do', 'off', 'such', 'then', 'me',
                         'am', 'where', 'this', 'having', 'before', 'through', 'other', "that'll", 'about', 'yourself',
                         'him', 'my', 'whom', 'more', 'have', 'of', 'above', 'on', 're', "you'd", 'their', 'between',
                         'down', 'how', 'ma', 'same', 'from', 'myself', 'y', 'in', 'some', 'they', 'those', 'yours',
                         'no', 'with', 'into', 'an', 'most', 'any', 'its', 'her', "it's", 'theirs', 'just', "you're",
                         'that', 's', 'does', 'it', 'hers', 'only', 'for', 'than', 'which', 'against', 'i', 'so',
                         "you'll", 'both', 'was']


def text_data_cleanup(corpus: list | str, patterns: list = [], replace_with: str = "",
                      sample: int = 10, get_info: bool = False) -> list:
    """
    STANDARD STEPS if patterns == []
    - Converting all letters to lowercase
    - Removing the data similar to HTML tags
    - Removing HTTP addresses
    - Email addresses and censored words removal
    - Deleting the digits from text

    :param corpus: corpus or string value you would like to clean
    :param patterns: list of regex patterns you would like to appy
    :param replace_with: result of replacement
    :param get_info: if the sample info about cleaned data should be displayed in console or not
    :param sample: how many items in info message per applied regex
    :return: list of strings
    """
    if isinstance(corpus, str):
        corpus = [corpus]

    corpus = [review.lower() for review in corpus]

    if len(patterns) == 0:
        patterns = stc.PATTERNS.copy()
        del patterns['rude_words']
        patterns = patterns.values()

    result = corpus
    for pattern in patterns:
        print(pattern)
        result = stc.clear_substr_in_texts(result, pattern, replace_with, sample, get_info)

    return result


def normalize_negations_in_corpus(corpus: list):
    """
    replaces negation shortcuts like don't isn into do not is not
    :param corpus: corpus to be modified
    :return: corpus with normalized shortcuts
    """
    # translate the shortcuts with 't suffix
    corpus = stc.translate_shortcuts(corpus, stc.negations_with_t)
    # translate the shortcuts without 't suffix
    return stc.translate_shortcuts(corpus, stc.negations_without_t)


def preprocess_corpus(corpus: list, normalize_neg: bool = True, punct: bool = True,
                      stopwords: list = customized_stop_words):
    """
    fn contains apars described in the notebook in section step 2: Further preprocessing of text data
    :param corpus: corpus to be modified
    :param normalize_neg: if normalization of negation shortcuts should be performed
    :param punct: if removing punctuation signs should be conducted
    :param stopwords: list of stopwords
    :return: list of tokens
    """
    if normalize_neg:
        corpus = normalize_negations_in_corpus(corpus)
    if punct:
        corpus = [stc.clear_punctuation(txt, None) for txt in corpus]

    corpus = [nltk.word_tokenize(review) for review in corpus]

    corpus = [
        [word for word in review if word not in stopwords]
        for review in corpus
    ]

    return corpus

