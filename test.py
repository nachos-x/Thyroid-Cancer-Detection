import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.optimize import root_scalar
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('hypothyroid.csv')

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Encode binary columns f/t to 0/1
binary_cols = [col for col in df.columns if col not in ['age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG', 'referral source', 'binaryClass']]
for col in binary_cols:
    df[col] = df[col].map({'f': 0, 't': 1})

# Encode sex F=0, M=1
df['sex'] = df['sex'].map({'F': 0, 'M': 1})

# Encode referral source with label encoding
df['referral source'] = df['referral source'].astype('category').cat.codes

# Encode binaryClass P=-1, N=1 (P benign, N malignant)
df['binaryClass'] = df['binaryClass'].map({'P': -1, 'N': 1})

# Numerical cols
numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# No imputation, keep NaN for now

# Pre-analyses
# Gender analysis
gender_percent = df.groupby('sex')['binaryClass'].value_counts(normalize=True) * 100
print("Gender-based percentages:\n", gender_percent)

# Age groups
bins = [7, 30, 40, 50, 60, 70, 83]
labels = ['7-30', '31-40', '41-50', '51-60', '61-70', '71-83']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
age_percent = df.groupby('age_group')['binaryClass'].value_counts(normalize=True) * 100
print("Age group percentages:\n", age_percent)

# Missing percentages
missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
print("Missing percentages:\n", missing_percent)

# Drop age_group
df.drop('age_group', axis=1, inplace=True)

# Features and target
features = [col for col in df.columns if col != 'binaryClass']
X = df[features]
y = df['binaryClass']

# Whole-batch condition for each feature
D = len(features)
conditions = {}
for col in features:
    x_col = X[col].dropna()
    if len(x_col) < 2:
        conditions[col] = np.inf
        continue
    b = np.c_[np.ones(len(x_col)), np.abs(x_col)]
    gram = b.T @ b
    eigs = np.linalg.eigvals(gram)
    min_eig = np.min(eigs)
    max_eig = np.max(eigs)
    cond = max_eig / min_eig if min_eig > 0 else np.inf
    conditions[col] = cond
print("Whole-batch conditions:\n", conditions)

# Dimensionality reduction
# Inner-similarity (correlation between features, ignore NaN pairwise)
corr_matrix = X.corr(method='pearson').abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
X_reduced_inner = X.drop(to_drop, axis=1)
print("Features dropped due to high inner similarity (>0.8):", to_drop)

# Target-similarity (absolute difference in means between classes, ignore NaN)
diffs = {}
for col in X_reduced_inner.columns:
    mean_neg = X_reduced_inner[y == -1][col].mean(skipna=True)
    mean_pos = X_reduced_inner[y == 1][col].mean(skipna=True)
    diffs[col] = abs(mean_neg - mean_pos)
sorted_diffs = sorted(diffs.items(), key=lambda item: item[1], reverse=True)
top_features = [f[0] for f in sorted_diffs[:10]]  # Top 10 as in paper
print("Top 10 features by target similarity difference:", top_features)

# Select top features
X_selected = X_reduced_inner[top_features]

# Drop rows with any NaN in selected features to avoid singularity
X_selected = X_selected.dropna()
y_selected = y[X_selected.index]

# Size reduction with hierarchical clustering per class
def reduce_size(group, threshold=1.0):
    if len(group) < 2:
        return group
    scaler = StandardScaler()
    scaled = scaler.fit_transform(group)
    dist = pdist(scaled, metric='euclidean')
    link = linkage(dist, method='weighted')
    clusters = fcluster(link, t=threshold, criterion='distance')
    reduced = group.groupby(clusters).head(1)
    return reduced

benign = X_selected[y_selected == -1]
malignant = X_selected[y_selected == 1]
benign_reduced = reduce_size(benign)
malignant_reduced = reduce_size(malignant)
X_reduced = pd.concat([benign_reduced, malignant_reduced])
y_reduced = y_selected[X_reduced.index]

print("Original size after dropna:", len(X_selected), "Reduced size:", len(X_reduced))

# Train-test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X_reduced.values, y_reduced.values, test_size=0.2, random_state=42)

# Scale data to prevent numerical instability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias
b_train = np.c_[np.ones(len(X_train)), X_train]
b_test = np.c_[np.ones(len(X_test)), X_test]

N, D = b_train.shape

# BLS
w_bls = np.linalg.pinv(b_train.T @ b_train) @ (b_train.T @ y_train)
pred_bls_train = b_train @ w_bls
y_pred_bls_train = np.sign(pred_bls_train)
acc_bls_train = accuracy_score(y_train, y_pred_bls_train)
pred_bls_test = b_test @ w_bls
y_pred_bls_test = np.sign(pred_bls_test)
acc_bls_test = accuracy_score(y_test, y_pred_bls_test)
print("BLS Train Acc:", acc_bls_train, "Test Acc:", acc_bls_test)

# INN
eta = 0.01  # Reduced eta for stability
w_inn = np.zeros(D)
for i in range(N):
    pred = w_inn @ b_train[i]
    e = y_train[i] - pred
    w_inn += eta * e * b_train[i]
pred_inn_train = b_train @ w_inn
y_pred_inn_train = np.sign(pred_inn_train)
acc_inn_train = accuracy_score(y_train, y_pred_inn_train)
pred_inn_test = b_test @ w_inn
y_pred_inn_test = np.sign(pred_inn_test)
acc_inn_test = accuracy_score(y_test, y_pred_inn_test)
print("INN Train Acc:", acc_inn_train, "Test Acc:", acc_inn_test)

# LSLC
eta_c = 0.1
alpha_val = -1.0
p = 1.0
alpha = alpha_val * np.ones(D)
upper_bound = p * np.abs(alpha)
bound = p**2 * np.linalg.norm(alpha)**2
w_lslc = np.zeros(D)
for i in range(N):
    # Project if outside norm bound
    current_norm = np.linalg.norm(w_lslc)
    if current_norm**2 > bound:
        w_lslc = w_lslc * np.sqrt(bound) / (current_norm + 1e-8)
    b = b_train[i]  # (D,)
    G = np.outer(b, b)  # D x D
    h = b * y_train[i]  # D,
    def get_increment(lam):
        mat = G + lam * np.eye(D)
        try:
            inv_mat = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            inv_mat = np.linalg.pinv(mat)
        return inv_mat @ h
    def f(lam):
        if lam < 0:
            return 1e10
        inc = get_increment(lam)
        w_temp = w_lslc + eta_c * inc
        return np.linalg.norm(w_temp)**2 - bound
    # Unconstrained
    inc_un = get_increment(0)
    w_un = w_lslc + eta_c * inc_un
    if np.linalg.norm(w_un)**2 <= bound:
        w_lslc = w_un
    else:
        # Check signs
        f0 = f(0)
        flarge = f(1e6)
        if f0 * flarge > 0:
            # Cannot find root, use large lam
            inc = get_increment(1e6)
            w_lslc = w_lslc + eta_c * inc
        else:
            sol = root_scalar(f, bracket=[0, 1e6], xtol=1e-5, rtol=1e-5)
            if sol.converged:
                lam = sol.root
                inc = get_increment(lam)
                w_lslc = w_lslc + eta_c * inc
            else:
                print("Root finder did not converge for sample", i)
    # Clip to box constraints
    w_lslc = np.clip(w_lslc, alpha, upper_bound)
pred_lslc_train = b_train @ w_lslc
y_pred_lslc_train = np.sign(pred_lslc_train)
acc_lslc_train = accuracy_score(y_train, y_pred_lslc_train)
pred_lslc_test = b_test @ w_lslc
y_pred_lslc_test = np.sign(pred_lslc_test)
acc_lslc_test = accuracy_score(y_test, y_pred_lslc_test)
print("LSLC Train Acc:", acc_lslc_train, "Test Acc:", acc_lslc_test)

# ICA
def ica_optimize(b, y, n_country=50, n_imp=5, max_iter=200, beta=2.0, gamma=0.1):
    D = b.shape[1]
    def cost(w):
        pred = b @ w
        return 0.5 * np.mean((y - pred)**2)
    # Initial countries
    countries = np.random.uniform(-1, 1, (n_country, D))
    costs = np.array([cost(c) for c in countries])
    # Select imperialists
    idx = np.argsort(costs)[:n_imp]
    imperialists = countries[idx]
    imp_costs = costs[idx]
    colonies = np.delete(countries, idx, axis=0)
    col_costs = np.delete(costs, idx)
    n_col = len(colonies)
    # Power
    power = 1.0 / (imp_costs + 1e-8)
    power /= power.sum()
    col_per_emp = np.round(power * n_col).astype(int)
    # Adjust sum
    diff = int(n_col - sum(col_per_emp))
    if diff > 0:
        col_per_emp[np.argsort(power)[-diff:]] += 1
    elif diff < 0:
        col_per_emp[np.argsort(power)[: -diff]] -= 1
    empires = []
    col_idx = 0
    for i in range(n_imp):
        num_col = col_per_emp[i]
        emp_cols = colonies[col_idx : col_idx + num_col] if num_col > 0 else np.array([])
        emp = {'imp': imperialists[i], 'cols': emp_cols, 'imp_cost': imp_costs[i]}
        empires.append(emp)
        col_idx += num_col
    # Main loop
    for _ in range(max_iter):
        for emp in empires:
            cols = emp['cols']
            if len(cols) == 0:
                continue
            # Assimilation
            for j in range(len(cols)):
                direction = emp['imp'] - cols[j]
                move = np.random.uniform(0, beta * np.linalg.norm(direction), D) * (direction / (np.linalg.norm(direction) + 1e-8))
                cols[j] += move
            # Revolution
            for j in range(len(cols)):
                if np.random.rand() < gamma:
                    cols[j] += np.random.normal(0, 0.5, D)
            emp['cols'] = cols
            # Swap if better
            col_costs = [cost(c) for c in cols]
            for j in range(len(cols)):
                if col_costs[j] < emp['imp_cost']:
                    temp = emp['imp'].copy()
                    emp['imp'] = cols[j].copy()
                    cols[j] = temp
                    emp['imp_cost'] = col_costs[j]
            emp['cols'] = cols
        # Competition
        if len(empires) <= 1:
            break
        total_costs = []
        for emp in empires:
            tc = emp['imp_cost']
            if len(emp['cols']) > 0:
                tc += gamma * np.mean([cost(c) for c in emp['cols']])
            total_costs.append(tc)
        total_costs = np.array(total_costs)
        power = 1.0 / (total_costs + 1e-8)
        power /= power.sum()
        # Weakest empire
        weak_idx = np.argmax(total_costs)
        weak_emp = empires[weak_idx]
        if len(weak_emp['cols']) > 0:
            # Weakest colony
            col_costs = [cost(c) for c in weak_emp['cols']]
            weak_col_idx = np.argmax(col_costs)
            col_to_move = weak_emp['cols'][weak_col_idx]
            # Strongest empire
            strong_idx = np.argmax(power)
            empires[strong_idx]['cols'] = np.append(empires[strong_idx]['cols'], [col_to_move], axis=0)
            weak_emp['cols'] = np.delete(weak_emp['cols'], weak_col_idx, axis=0)
        # Eliminate empty
        if len(weak_emp['cols']) == 0 and len(empires) > 1:
            del empires[weak_idx]
    # Return the best
    best_costs = [emp['imp_cost'] for emp in empires]
    best_idx = np.argmin(best_costs)
    w_ica = empires[best_idx]['imp']
    return w_ica

w_ica = ica_optimize(b_train, y_train)
pred_ica_train = b_train @ w_ica
y_pred_ica_train = np.sign(pred_ica_train)
acc_ica_train = accuracy_score(y_train, y_pred_ica_train)
pred_ica_test = b_test @ w_ica
y_pred_ica_test = np.sign(pred_ica_test)
acc_ica_test = accuracy_score(y_test, y_pred_ica_test)
print("ICA Train Acc:", acc_ica_train, "Test Acc:", acc_ica_test)