import argparse
import os.path
import pickle

from sklearn.datasets import make_moons

from decision_tree.classifier import DecisionTreeClassifier
from decision_tree.draw_tree import draw_tree
from decision_tree.plots import plot_2d, plot_roc_curve


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def dump(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


def load(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def main():
    args = get_args()
    if args.prepare_data:
        noise = 0.35
        X_train, y_train = make_moons(1500, noise=noise)
        X_test, y_test = make_moons(200, noise=noise)

        if not os.path.exists("data"):
            os.mkdir("data")
        dump(X_train, "data/train_x")
        dump(y_train, "data/train_y")
        dump(X_test, "data/test_x")
        dump(y_test, "data/test_y")
    elif args.fit:
        tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
        X_train = load("data/train_x")
        y_train = load("data/train_y")
        tree.fit(X_train, y_train)

        if not os.path.exists("model"):
            os.mkdir("model")
        dump(tree, "model/tree")
    elif args.predict:
        tree = load("model/tree")
        X_test = load("data/test_x")
        pred_y = tree.predict_proba(X_test)
        dump(pred_y, "data/pred_y")
    elif args.plots:
        X_train = load("data/train_x")
        y_train = load("data/train_y")
        tree = load("model/tree")
        y_test = load("data/test_y")
        y_pred = load("data/pred_y")

        if not os.path.exists("plots"):
            os.mkdir("plots")
        plot_2d(tree, X_train, y_train, save_path="plots/2d.png")
        plot_roc_curve(y_test, y_pred, save_path="plots/roc_curve.png")
        draw_tree(tree, save_path="plots/tree.png")
    else:
        print(os.getcwd())


if __name__ == "__main__":
    main()
