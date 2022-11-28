from typing import Any, Dict, List, NoReturn, Optional, Union

import numpy as np

from decision_tree.metrics import entropy, gain, gini
from decision_tree.node import DecisionTreeLeaf, DecisionTreeNode


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
    ):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X_data: np.ndarray, y_labels: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X_data : np.ndarray
            Обучающая выборка.
        y_labels : np.ndarray
            Вектор меток классов.
        """
        self.root = self._build(X_data, y_labels, 0)

    def predict_proba(self, X_data: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X_data : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """
        ans = []
        for x_point in X_data:
            cur = self.root
            while not isinstance(cur, DecisionTreeLeaf):
                if x_point[cur.split_dim] < cur.split_value:
                    cur = cur.left
                else:
                    cur = cur.right
            ans.append(cur.m)
        return ans

    def predict(self, X_array: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X_array : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X_array)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]

    def _build(
        self, X_array: np.ndarray, y_labels: np.ndarray, depth: int
    ) -> Union["DecisionTreeNode", DecisionTreeLeaf]:
        if len(np.unique(y_labels)) == 1:
            return DecisionTreeLeaf(y_labels)

        if depth == self.max_depth:
            return DecisionTreeLeaf(y_labels)

        if len(y_labels) < 2 * self.min_samples_leaf:
            return DecisionTreeLeaf(y_labels)

        dim = -1
        threshold = -1
        inf_gain = 0

        for i in range(X_array.shape[1]):
            ind = list(range(X_array.shape[0]))
            ind.sort(key=lambda k: X_array[k][i])
            for j_point in range(
                self.min_samples_leaf, X_array.shape[0] - self.min_samples_leaf
            ):
                cur_gain = gain(
                    y_labels[ind[:j_point]], y_labels[ind[j_point:]], self.criterion
                )
                if cur_gain > inf_gain:
                    dim = i
                    threshold = X_array[ind[j_point]][i]
                    inf_gain = cur_gain

        if dim == -1:
            return DecisionTreeLeaf(y_labels)

        left = [
            i for i, (x, _) in enumerate(zip(X_array, y_labels)) if x[dim] < threshold
        ]
        right = [
            i for i, (x, _) in enumerate(zip(X_array, y_labels)) if x[dim] >= threshold
        ]

        return DecisionTreeNode(
            dim,
            threshold,
            self._build(X_array[left], y_labels[left], depth + 1),
            self._build(X_array[right], y_labels[right], depth + 1),
        )
