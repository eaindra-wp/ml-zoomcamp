from curses.ascii import alt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def exercises(df: pd.DataFrame):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )

    y_train = df_train['converted']
    y_val = df_val['converted']

    q1_cols = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']

    highest_auc = 0
    highest_auc_idx = q1_cols[0]
    for a in q1_cols:
        auc = roc_auc_score(y_train,df_train[a])
        if auc < 0.5:
            auc = roc_auc_score(y_train, -df_train[a])
        if auc > highest_auc:
            highest_auc_idx = a

    print(f'1. {highest_auc_idx}')

    #########################################
    train_dicts = df_train.drop(columns=['converted']).to_dict(orient='records')
    val_dicts = df_val.drop(columns=['converted']).to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)

    model.fit(X_train, y_train)
    X_val = dv.transform(val_dicts)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    print(f'2. {round(roc_auc_score(y_val, y_pred_proba), 3)}')

    #############################
    thresholds = np.arange(0, 1, 0.01)
    plots = []
    for t in thresholds:
        y_pred = y_pred_proba >= t
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn)
        p_minus_r = abs(precision - recall)

        f1 = round((2 * precision * recall) / (precision + recall), 3)
        plots.append((t, precision, recall, p_minus_r, f1))

    df_results = pd.DataFrame(plots, columns=['threshold', 'precision', 'recall', 'p_minus_r', 'f1'])
    df_results = df_results[(df_results['precision'] > 0) & (df_results['recall'] > 0)]

    print(f'3. {df_results.loc[df_results['p_minus_r'].idxmin(), 'threshold']}')
    print(f'4. {df_results.loc[df_results['f1'].idxmax(), 'threshold']}')

    ##########################

    print(f'5. {round(np.std(k_fold(df_full_train)), 3)}')

    C_scores = [0.000001, 0.001, 1]
    results = [
        (round(np.mean(s := k_fold(df_full_train, c)), 3),
        -round(np.std(s), 3),
        -c,
        c)
        for c in C_scores
    ]
    print(f'6. {max(results)[3]}')

def k_fold(df_full_train: pd.DataFrame, C=1.0):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):

        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]
    
        y_train_fold = df_train_fold['converted']
        y_val_fold = df_val_fold['converted'] 

        train_dicts = df_train_fold.drop(columns=['converted']).to_dict(orient='records')
        val_dicts = df_val_fold.drop(columns=['converted']).to_dict(orient='records')

        dv = DictVectorizer(sparse=False)
        X_train_fold = dv.fit_transform(train_dicts)
        X_val_fold = dv.transform(val_dicts)
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_prob_fold = model.predict_proba(X_val_fold)[:, 1]

        scores.append(roc_auc_score(y_val_fold, y_pred_prob_fold))

    return scores

def main():
    HERE = Path(__file__).parent.parent
    df = pd.read_csv(HERE / "course_lead_scoring_2026.csv")

    df = df.fillna({
        col: 0 if pd.api.types.is_any_real_numeric_dtype(df[col]) else 'NA'
        for col in df
    })

    exercises(df)

if __name__ == "__main__":
    main()