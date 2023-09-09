from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix


def get_pipeline_with_best_params(corpus: list, labels: list, vectorizer, model, param_distributions: dict,
                                  n_iter=10, cv=5, scoring="accuracy"):
    """
    fm adopts parameters that determine the model and vectorizer, dataset and its labels, and attributes for
    RandomizedSearch.
    The indicated combinations are checked by RandomizedSearchCV and once the best parameters are returned
    the dicts of parameters for vectorizer and model are built. Finally, the pipeline that contains these objects
    with these parameters is returned.
    :param corpus: list of text documents
    :param labels: list of labels
    :param vectorizer: vectorizer object that should be applied
    :param model: model object you would like to check
    :param param_distributions: params for vectorizer and model
    :param n_iter: how many iterations
    :param cv: Determines the cross-validation splitting strategy. (see in RandomizedSearchCV documentation)
    :param scoring: Strategy to evaluate the performance of the cross-validated model (str, callable, list, tuple, dict)
            see in (more info in RandomizedSearchCV documentation)
    :return: pipeline object that contains vectorizer and model objects with filled parameters
    """
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("model", model)
    ])

    # RandomizedSearchCV with parameter sampling
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=n_iter, cv=cv,
                                       scoring=scoring)
    random_search.fit(corpus, labels)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best parameters:", best_params)
    print("\nbest score:", best_score)

    model.set_params(**{k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')})
    vectorizer.set_params(**{k.replace('vectorizer__', ''): v for k, v in best_params.items()
                             if k.startswith('vectorizer__')})

    return Pipeline([
                ("vectorizer", vectorizer),
                ("model", model)
            ])


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def evaluate_model(model, X_test, y_test):
    """
    fn makes prediction evaluation and returns metrics as DataFrame and displays additionally plots: ROC curve and
    Confusion matrix.
    :param model: model or pipeline
    :param X_test: test data (corpus)
    :param y_test: actual labels connected with test data to compare with predicted ones
    :return: DataFrame of calculated metrix
    """
    # make prodiction
    y_pred = model.predict(X_test)

    # calculate the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # calculate ROC i AUC
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Metryka': ['accuracy', 'precision', 'recall', 'F1 Score', 'ROC AUC'],
        'Wartość': [accuracy, precision, recall, f1, roc_auc]
    })

    #  ROC curve plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    # Confusion matrix plot
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Confusion matrix')

    plt.tight_layout()
    plt.show()

    return metrics_df
