from typing import Type

from deepeval.classifiers.classifier import ClassifierTemplate
from deepeval.classifiers.topics.topics import TopicClassifier
from deepeval.templates import make_template_class


class UseCaseClassifierTemplate(
    make_template_class("UseCaseClassifier", feature="classifiers"),
    ClassifierTemplate,
):
    pass


class UseCaseClassifier(TopicClassifier):
    """Labels what the user was trying to get done, reviewing each label so a
    step the system took is not mistaken for the user's goal."""

    default_template: Type[ClassifierTemplate] = UseCaseClassifierTemplate
