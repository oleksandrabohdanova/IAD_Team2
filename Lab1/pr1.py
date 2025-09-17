import numpy as np
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Survived']].dropna()
T = df[['Pclass', 'Sex', 'SibSp', 'Parch']]  
y = df['Survived']  
T_str = T.map(str)
y_str = y.astype(str)

def gini_index(subset):
    cardinality = len(subset)
    if cardinality == 0:
        return 0
    counts = subset.value_counts()
    p_j = counts / cardinality
    return 1 - np.sum(p_j**2)

def feature_gscore(data, labels, feature):
    values = data[feature].unique()
    N = len(data)
    G = 0
    for val in values:
        subset_idx = data[feature] == val
        subset_labels = labels[subset_idx]
        G += (len(subset_labels) / N) * gini_index(subset_labels)
    return G

def feature_details(data, labels, feature):
    values = data[feature].dropna().unique()
    details = {}
    for val in values:
        subset_idx = data[feature] == val
        subset_labels = labels[subset_idx]
        details[val] = {
            "count": len(subset_labels),
            "gini": gini_index(subset_labels),
            "class_dist": subset_labels.value_counts().to_dict()
        }
    return details

def best_feature(data, labels):
    scores = {feature: feature_gscore(data, labels, feature) for feature in data.columns}
    best = min(scores, key=scores.get)
    return best, scores

best, scores = best_feature(T_str, y_str)
sorted_scores = sorted(scores.items(), key=lambda x: x[1])

print("G(x_h) scores ranking:")
for feat, sc in sorted_scores:
    print(f"{feat:6s}: {sc:.6f}")

print("\nBest feature: ", best)
print("Details:")
details = feature_details(T_str, y_str, best)
for val, info in details.items():
    print(f"  {best} = {val}: passengers count={info['count']}, gini={info['gini']:.6f}, class distribution={info['class_dist']}")