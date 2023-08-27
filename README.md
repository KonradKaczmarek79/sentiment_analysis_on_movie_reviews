# sentiment_analysis_on_movie_reviews
final project - "Data Science PRO" Bootcamp in Kodołamacz/Sages (https://www.kodolamacz.pl/)

## Introduction
The aim of this project is to compare different approaches to working with text data in the case of sentiment analysis topic. In the following steps, various actions were carried out on the movie review documents contents, to bring this text data to a form from which machine learning algorithms can extract information about "sentiment" groups of these documents.

The analysis of sentiment (positive vs. negative) was performed on a large dataset described briefly in the **Dataset** section.

## Dataset
**Large Movie Review Dataset** (https://ai.stanford.edu/~amaas/data/sentiment/)

The dataset contains 50,000 movie reviews for binary sentiment classification (25,000 for training and 25,000 for testing). The reviews were taken from the IMDB portal and the file names contain info about the movie's numeric rating (stars awarded by the reviewer). As we can read in the note associated with the dataset the content of these reviews is “highly polar” – and it looks that the negative reviews have a rating between 0 and 4 and the positive reviews are rated from 7 to 10.

## Requirements and libraries’ versions
- all requirements (additional installs needed to run the coda) are contained in the file: `requirements.txt`
- all versions of modules, libraries, and things like that are listed in jupyter notebook: `suppprted_modules/proj_info.ipynb`

## The structure of this repository

### Text data preprocessing
The first part deals with the steps needed before movie review data can be used for machine learning. 
There are a lot of various techniques but some of them can be useful for certain problem only. 
While working on the project, I tried to choose the most appropriate for the problem of sentiment analysis in reviews.

#### Text data cleanup
This stage concerns the preparation of data, regardless of what model and type of text division into tokens and vectors will be used. 
It is about removing the noise that is caused by data that does not carry information about feelings, emotions, etc.

The activities carried out at this stage include:

- Checking and removing the data similar to HTML tags
- Converting all letters to lowercase
- Removing HTTP addresses
- Email addresses and censored words removal
- Deleting the digits from text
- Removing punctuation


***Next steps will be added soon. 
The project is currently under construction…***