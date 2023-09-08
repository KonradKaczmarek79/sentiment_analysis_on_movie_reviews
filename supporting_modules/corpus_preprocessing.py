import supporting_modules.text_cleaning as stc
import nltk
import spacy

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


def drop_duplicates_from_corpus(corpus_texts: list, labels: list) -> (list, list):
    """
    fn removes duplicated texts from the corpus and the labels on indexes associated with these texts
    :param corpus_texts: list of documents (text data) for check
    :param labels: list of labels connected with texts in the corpus
    :return: list of unique texts from the passed corpus and list of labels connected with them
    """
    unique_texts = []
    unique_labels = []
    already_seen_texts = set()

    for text, label in zip(corpus_texts, labels):
        if text not in already_seen_texts:
            unique_texts.append(text)
            unique_labels.append(label)
            already_seen_texts.add(text)

    return unique_texts, unique_labels


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

    return [" ".join(word_list) for word_list in corpus]


def lemm_stemm(corpus: list, stemming: bool = True, stemmer=None, lemmatization: bool = False, lemmatizer=None):
    """
    fn performs lemmatization or stemming process or both of them lemmatization + steming
    :param corpus: corpus to be modified
    :param stemming: whether stemming will be applied
    :param stemmer: kind of stemmer (default: nltk.word_tokenize)
    :param lemmatization: whether lemmatization will be applied
    :param lemmatizer: kind of lemmatizer (default: spacy.load("en_core_web_sm"))
    :return:
    """
    if lemmatization:
        lemmatizer = spacy.load("en_core_web_sm") if not lemmatizer else lemmatizer

        lemmatized_corpus = []
        for text in corpus:
            doc = lemmatizer(text)
            # lemmatized_text = " ".join([token.lemma_ for token in doc])
            lemmatized_text = [token.lemma_ for token in doc]
            lemmatized_corpus.append(lemmatized_text)

        corpus = lemmatized_corpus

    else:
        # else tokenize words for stemmer
        corpus = [nltk.word_tokenize(text) for text in corpus]

    if stemming:
        stemmer = nltk.PorterStemmer() if not stemmer else stemmer

        corpus = [
                    [stemmer.stem(word) for word in text]
                    for text in corpus
                ]

    # return corpus
    return [" ".join(tokens) for tokens in corpus]
