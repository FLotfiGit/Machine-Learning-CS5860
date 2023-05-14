# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 20:14:08 2021

@author: eefat
"""

import numpy as np

"""
@author: Jia Geng
@email: jxg570@miami.edu
"""

class SAMME:
    """
    SAMME - multi-class AdaBoost algorithm
    @ref:   Zhu, Ji & Rosset, Saharon & Zou, Hui & Hastie, Trevor. (2006). Multi-class AdaBoost. Statistics and its
            interface. 2. 10.4310/SII.2009.v2.n3.a8.
    """

    def __init__(self, num_learner, num_cats):
        """
        Constructor
        :param num_learner: number of weak learners that will be boosted together
        :param num_cats: number of categories
        """

        if num_cats < 2:
            raise Exception("Param num_cat should be at least 2 but was {}".format(num_cats))

        self.num_learner = num_learner
        self.num_cats = num_cats
        self.entry_weights = None
        self.learner_weights = None
        self.sorted_learners = None

    def train(self, train_data, learners):
        """
        Train the AdaBoost .
        The training data need to be in the format: [(X, label), ...]
        The learners need to be in the format: [obj1, obj2, ...]
        The learner object need to have: a predict method that can output the predicted class. obj.predict(X) -> cat: int
        :param train_data: training data
        :param learners: weak learners
        :return: void
        """

        print("\nStart training SAMME..")
        # initialize the weights for each data entry
        n, m = len(train_data), len(learners)
        self.entry_weights = np.full((n,), fill_value=1/n, dtype=np.float32)
        self.learner_weights = np.zeros((m,), dtype=np.float32)

        # sort the weak learners by error
        error = [0 for i in range(m)]
        for learner_idx, learner in enumerate(learners):
            for entry in train_data:
                X, label = entry[0], int(entry[1])
                predicted_cat = learner.predict(X)
                if predicted_cat != label:
                    error[learner_idx] += 1
        self.sorted_learners = [l for l, e in sorted(zip(learners, error), key=lambda pair: pair[1])]

        # boost
        for learner_idx, learner in enumerate(self.sorted_learners):
            # compute weighted error
            is_wrong = np.zeros((n,))
            for entry_idx, entry in enumerate(train_data):
                X, label = entry[0], int(entry[1])
                predicted_cat = learner.predict(X)
                if predicted_cat != label:
                    is_wrong[entry_idx] = 1
            weighted_learner_error = np.sum(is_wrong * self.entry_weights)/self.entry_weights.sum()

            # compute alpha, if the learner is not qualified, set to 0
            self.learner_weights[learner_idx] = max(0, np.log(1/(weighted_learner_error + 1e-6) - 1) + np.log(
                self.num_cats - 1))
            alpha_arr = np.full((n,), fill_value=self.learner_weights[learner_idx], dtype=np.float32)
            # update entry weights, prediction made by unqualified learner will not update the entry weights
            self.entry_weights = self.entry_weights * np.exp(alpha_arr * is_wrong)
            self.entry_weights = self.entry_weights/self.entry_weights.sum()

        # normalize the learner weights
        self.learner_weights = self.learner_weights/self.learner_weights.sum()
        print("Training completed.")

    def predict(self, X):
        """
        Predict using the boosted learner
        :param X:
        :return: predict class
        """

        pooled_prediction = np.zeros((self.num_cats,), dtype=np.float32)

        for learner_idx, learner in enumerate(self.sorted_learners):
            # encode the prediction in to balanced array
            predicted_cat = learner.predict(X)
            prediction = np.full((self.num_cats,), fill_value=-1/(self.num_cats-1), dtype=np.float32)
            prediction[predicted_cat] = 1
            pooled_prediction += prediction*self.learner_weights[learner_idx]

        return np.argmax(pooled_prediction)