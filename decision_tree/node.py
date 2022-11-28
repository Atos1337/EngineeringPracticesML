from collections import defaultdict
from typing import Union


class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, x_features):
        mapp = defaultdict(int)
        for y_feature in x_features:
            mapp[y_feature] += 1
        self.m = {k: v / len(x_features) for k, v in mapp.items()}
        self.y = sorted(mapp.keys(), key=lambda x: -mapp[x])[0]


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(
        self,
        split_dim: int,
        split_value: float,
        left: Union["DecisionTreeNode", DecisionTreeLeaf],
        right: Union["DecisionTreeNode", DecisionTreeLeaf],
    ):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
