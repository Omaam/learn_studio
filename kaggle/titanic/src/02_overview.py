"""
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def do_feature_engineering(data):

    data["Sex"] = data["Sex"].replace({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Embarked"] = data["Embarked"].replace({"S": 0, "C": 1, "Q": 2})
    data["Fare"] = data["Fare"].fillna(np.mean(data["Fare"]))

    age_avg = np.mean(data["Age"])
    age_std = np.std(data["Age"])
    data["Age"] = data["Age"].fillna(
        np.random.randint(age_avg-age_std, age_avg+age_std)
    )

    delete_columns = ["Name", "PassengerId", "SibSp",
                      "Parch", "Ticket", "Cabin"]
    data = data.drop(delete_columns, axis=1)
    return data


def main():

    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    gender_submission = pd.read_csv("../input/gender_submission.csv")

    data = pd.concat([train, test], sort=False)

    data = do_feature_engineering(data)

    train = data[:len(train)]
    test = data[len(train):]

    y_train = train["Survived"]
    X_train = train.drop("Survived", axis=1)
    X_test = test.drop("Survived", axis=1)

    clf = LogisticRegression(penalty="l2", solver="sag", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    sub = gender_submission
    sub["Survived"] = y_pred.astype(int)
    sub.to_csv("../output/submission.csv", index=None)


if __name__ == "__main__":
    main()
