from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd


def make_cls(df, cls, classes):
    CLASS_COUNTS = {c: df[df[cls] == c].shape[0] for c in range(len(classes))}
    x, y = make_classification(
        n_samples=max(CLASS_COUNTS.values()) * (len(classes) + 1),
        n_features=random.randint((df.shape[1] - 1) * 2, (df.shape[1] - 1) * 3),
        n_classes=len(classes),
        n_clusters_per_class=1,
        random_state=1,
        n_informative=len(classes) * 2,
    )
    df1 = pd.DataFrame(x)
    df1[cls] = y.astype(int)
    df1 = pd.concat(
        [df1[df1[cls] == c].head(len(df[df[cls] == c])) for c in range(len(classes))]
    )
    df1 = df1.sample(frac=1, random_state=1)
    x, y = df1.values[:, :-1], df1.values[:, -1]
    ss = StandardScaler()
    x = ss.fit_transform(x)
    return x, y
