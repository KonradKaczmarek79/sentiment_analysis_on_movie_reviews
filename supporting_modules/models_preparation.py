from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


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
