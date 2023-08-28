import text_cleaning as tc


def text_data_cleanup(dataset: list | str, patterns: list = [], replace_with: str = "",
                      sample: int = None, get_info: bool = False) -> list:
    """
    STANDARD STEPS if patterns == []
    - Converting all letters to lowercase
    - Removing the data similar to HTML tags
    - Removing HTTP addresses
    - Email addresses and censored words removal
    - Deleting the digits from text

    :param dataset:
    :param patterns:
    :param replace_with:
    :param sample:
    :param get_info:
    :return:
    """
    dataset = [review.lower() for review in dataset]

    if len(patterns) == 0:
        patterns = tc.PATTERNS.copy()
        del patterns['rude_words']
        patterns = patterns.values()

    result = dataset
    for pattern in patterns:
        print(pattern)
        result = tc.clear_substr_in_texts(result, pattern, replace_with, sample, get_info)

    # return [tc.clear_punctuation(txt, None) for txt in result]
    return result


test = ["Don you know 333 don't you333 know, AAAA it's ain like that kk@kk.com ",
        "aren't you aren you why ain you <h></h> SUPER",
        "http://www.interia.pl mightn't he or weren she sh$$$ www.onet.pl www.github.com.pl",
        '< than 30 minutes of watching, being bored and irritated. <br />',
        '< who was to be a victim, but woman-power trumps evil scientist every time.<br />',
        '</SPOILER>',
        ]

print(test)

print(text_data_cleanup(test))