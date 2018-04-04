# Copyright 2017 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Models implemented with scikit-learn

logreg :Standard Logistic Regression Model using L2 regularization and the
LibLinear optimization implemented in scikit-learn

mlp: Multi-Layer Classifier

Features:

    Content+Punctuation()
    Structure:  depth(raw count + normalized),
                number of sentences/words/characters of both body and title,
                for both current and parent
    Author: whether author of initial post / same as the parent commenter
    Thread: total number of comments in the discussion,
            whether self_post / link_post
            average length of all the branches/threads of discussion in the
            discussion tree
    Community: subreddit

"""

# sys.path.append(os.path.abspath('.'))

import sys
import json
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn_prepare import *


# TODO write own split that does groupkfold but with shuffle
def split(fold_type):
    """return indices(enumerate) according to fold type"""
    global X, y
    if fold_type == "GroupKFold":
        return group_k_fold_10.split(X, y, X.thread_id)
    elif fold_type == "KFold":
        return k_fold_10.split(X, y)
    # TODO my own fold tool
    else:
        return None


def random_test(model, fold_type="GroupKFold"):
    """random test once instead of k-fold cross validation"""

    # X_train, X_test, y_train, y_test \
    #     = train_test_split(X_data, y, random_state=42,
    #                                                     test_size=0.1,
    #                                                     shuffle=False)

    global X, y, X_data

    indices = split(fold_type)

    train_index, test_index = next(indices)
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test  shape:', X_test.shape)
    print('X_test  shape:', y_test.shape)

    # time.clock(): the current processor time in seconds (on Unix)
    print("Training model...")
    ticks = time.clock()
    # pipeline.fit(X_train, y_train)
    model.fit(X_train, y_train)
    print("time used to train the model is %.2f s" %
          (time.clock() - ticks))

    print("Predicting model...")
    ticks = time.clock()
    y_pred = model.predict(X_test)
    print("time used to predict the model is %.2f s" %
          (time.clock() - ticks))

    print("accuracy, precision, recall, f1 score = ")
    print(metrics.accuracy_score(y_test, y_pred),
          (metrics.precision_score(y_test, y_pred, average='weighted')).mean(),
          (metrics.recall_score(y_test, y_pred, average='weighted')).mean(),
          (metrics.f1_score(y_test, y_pred, average='weighted')).mean())

    print("report")
    print(metrics.classification_report(y_test, y_pred))
    print("confusion matrix")
    print(metrics.confusion_matrix(y_test, y_pred))


def cross_validation(model, fold_type="GroupKFold"):
    """
    cross validation
    :param model: logreg / mlp
    :param fold_type:
        GroupKFold (from sklearn): no shuffle, no random state(always the
        same) but keeps comments from the same thread in the same training
        group
        KFold(from sklearn): with shuffle, with random state, but cannot
        make comments from the same thread trained together
    :return: print information in every fold and average score (10-fold)
    """
    global X, y, X_data

    accuracy_scores = []
    avg_precision_scores, avg_recall_scores, avg_f1_scores = [], [], []
    weighted_precision_scores, weighted_recall_scores, weighted_f1_scores = \
        [], [], []

    i = 0
    ticks_all = time.clock()

    indices = None
    if fold_type == "GroupKFold":
        indices = group_k_fold_10.split(X, y, X.thread_id)
    elif fold_type == "KFold":
        indices = k_fold_10.split(X, y)

    for train_index, test_index in indices:
        print(i)
        i += 1
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("Training model...")
        ticks = time.clock()
        model.fit(X_train, y_train)
        print("time used to train the model is %.2f s" %
              (time.clock() - ticks))

        print("Predicting model...")
        ticks = time.clock()
        y_pred = model.predict(X_test)
        print("time used to test the model is %.2f s" %
              (time.clock() - ticks))

        accuracy_score = metrics.accuracy_score(y_test, y_pred)
        avg_precision, avg_recall, avg_f1, avg_support \
            = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                      average=None)
        weighted_precision, weighted_recall, weighted_f1, weighted_support \
            = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                      average='weighted')

        accuracy_scores.append(accuracy_score)

        avg_precision_scores.append(avg_precision)
        avg_recall_scores.append(avg_recall)
        avg_f1_scores.append(avg_f1)

        weighted_precision_scores.append(weighted_precision)
        weighted_recall_scores.append(weighted_recall)
        weighted_f1_scores.append(weighted_f1)

        report = metrics.classification_report(y_test, y_pred)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        print("accuracy =", accuracy_score)
        # print("average precision, recall, f1 score = ",
        #       np.mean(avg_precision), np.mean(avg_recall), np.mean(avg_f1))
        print("weighted precision, recall, f1 score = ",
              weighted_precision, weighted_recall, weighted_f1)

        print("report:\n", report)
        print("confusion matrix:\n", confusion_matrix)

    print("\n\nTime used for training the 10-fold model is %.2f s",
          time.clock() - ticks_all)
    print("Averaged accuracy  = %.3f" % np.mean(accuracy_scores))

    print("Average scores are:")
    print("Precision = %.3f" % np.mean(avg_precision_scores))
    print("Recall    = %.3f" % np.mean(avg_recall_scores))
    print("F1 Score  = %.3f" % np.mean(avg_f1_scores))

    print("Weighted scores are:")
    print("Precision = %.3f" % np.mean(weighted_precision_scores))
    print("Recall    = %.3f" % np.mean(weighted_recall_scores))
    print("F1 Score  = %.3f" % np.mean(weighted_f1_scores))

def predict_labels():
    global X, y, X_data
    model = joblib.load('logreg.model')
    print("Training model...")
    ticks = time.clock()
    
    model.fit(X_data, y)
    joblib.dump(model, 'logreg.model')
    
    sys.stdout = open('X.txt', 'w')
    print(X)
    sys.stdout = open('y.txt', 'w')
    print(y)
    sys.stdout = open('X_data.txt', 'w')
    print(X_data)
    model = joblib.load('logreg.model')
    print("time used to train the model is %.2f s" %
          (time.clock() - ticks))
    print("Predicting model...")
    ticks = time.clock()
    print("Processing unlabeled data...")
    unlabeled_data_file_name = "/home/zsong/working/my_trial_DIR/post_df_parent_text_unified.json"
    # unlabeled_data_file_name = "/home/zsong/working/my_trial_DIR/patrick_dict_10.json"
    unlabeled_X, unlabeled_y = load_data(data_file_name=unlabeled_data_file_name)
    sys.stdout = open('unlabeled_X.txt', 'w')
    print(unlabeled_X)
    # with open("unlabeled_data", "w") as output_file:
    #     json.dump(unlabeled_X, output_file)
    unlabeled_X_data = features.fit_transform(unlabeled_X)
    scipy.io.mmwrite("/home/zsong/working/unlabeled_transformed_data.mtx", unlabeled_X_data)
    


    unlabeled_data_file_name = "/home/zsong/working/unlabeled_transformed_data.mtx"
    transform_data(unlabeled_X, data_file_name=unlabeled_data_file_name)
    unlabeled_X_data = scipy.io.mmread(unlabeled_data_file_name)
    unlabeled_X_data = unlabeled_X_data.tocsr()
    predicted_labels = model.predict(unlabeled_X_data) # not fitted
    print("time used to predict the model is %.2f s" %
         (time.clock() - ticks)) 
    with open("predicted_labels", "w") as labels:
        json.dump(predicted_labels, labels) 
    print("Prediction finished.")

def main():
    # log_file = open("../log/model.log", "w+")
    # sys.stdout(log_file)
    # print(datetime.datetime.now())

    global X, y, X_data
    
    print("Loading data ...")
    
    ticks = time.clock()
    # X, y = load_data(data_file_name="/home/zsong/working/data/post_df.json")
    X, y = load_data(data_file_name="/home/zsong/working/my_trial_DIR/post_df_parent_text_unified.json")

    print("Transforming data ...")
    # ticks = time.clock()
    # X_data = features.fit_transform(X)
    # print("time used to transform the data is %.2f s" %
    #       (time.clock() - ticks))
    X_data = features.fit_transform(X)
    scipy.io.mmwrite("/home/zsong/working/transformed_data.mtx", X_data)
    # scipy.io.mmwrite("../data/X_data_20.mtx", X_data)

    
    # data_file_name = "/home/zsong/working/X_data.mtx"
    # data_file_name = "/home/fwang/dialogue_discourse_acts/baseline/data/X_data_50_parent_text.mtx"
    # data_file_name = "/home/zsong/working/X_data_50_parent_text.mtx"
    data_file_name = "/home/zsong/working/transformed_data.mtx"
    transform_data(X, data_file_name=data_file_name)

    X_data = scipy.io.mmread(data_file_name)
    X_data = X_data.tocsr()

    print("time used to load the data is %.2f s" %
          (time.clock() - ticks))

    # model = mlp
    # print("Model: multi-layer perceptron")
    # random_test(model)
     
    logreg = LogisticRegression(
        penalty='l2',
        C=3.0,
        # verbose=True,
        # multi_class='multinomial',
        solver='liblinear'
    )

    model = logreg
    joblib.dump(model, 'logreg.model')
    
    ### commented out random test
    #print("Model: logistic regression")
    #random_test(model, fold_type="KFold")
    ###
    predict_labels()


    # cross_validation(model, fold_type="KFold")

    # cross_validation(model, fold_type="GroupKFold")
    #
    # # TODO MLP model tune parameters
    # mlp = MLPClassifier(hidden_layer_sizes=(200, 200), solver='adam',
    #                     learning_rate='adaptive', learning_rate_init=0.001,
    #                     activation='logistic',
    #                     verbose=True)
    # cross_validation(mlp, fold_type="KFold")
    #


#
# grid_search_cv = GridSearchCV(
#     cv=10,
#     estimator=mlp,
#     param_grid=[{
#         'alpha': [.00001, .0001, .001, .01, .1],
#         'hidden_layer_sizes': [(10,), (50,), (100,), (200,)],
#         'learning_rate_init': [0.0001, 0.001, 0.01, 0.1, 1]
#     }]
# )
#
# grid_search_cv.fit(X_data, y)


if __name__ == '__main__':
    main()
