from config import RANDOM_STATE

def split_data(data):
    train = data[data["year"] < 2013]
    test = data[data["year"] == 2013]

    train = train.sample(frac=1, random_state=RANDOM_STATE)

    X_train = train.drop(columns=["WnvPresent","year"])
    y_train = train["WnvPresent"]

    X_test = test.drop(columns=["WnvPresent","year"])
    y_test = test["WnvPresent"]

    return X_train, X_test, y_train, y_test