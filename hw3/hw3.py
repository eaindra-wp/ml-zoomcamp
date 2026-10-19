from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer

def logistic_regression(df_train_org, df_val_org, col_name='converted', c=1.0):
    df_train = df_train_org.copy()
    df_val = df_val_org.copy()

    y_train = df_train[col_name]
    X_train = df_train.drop(columns=[col_name])
    
    y_val = df_val[col_name]
    X_val = df_val.drop(columns=[col_name])
    
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    
    dv = DictVectorizer(sparse=False)
    
    train_dict = X_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    model.fit(X_train, y_train)
    
    X_val = dv.transform(X_val.to_dict(orient='records'))
    y_pred = model.predict(X_val)

    return np.round((y_pred == y_val).mean(), 2)


def exercises(df: pd.DataFrame):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=42
    )

    ### biggest correlation ###
    opt_1 = df_full_train['interaction_count'].corr(df_full_train['lead_score'])
    opt_2 = df_full_train['number_of_courses_viewed'].corr(df_full_train['lead_score'])
    opt_3 = df_full_train['number_of_courses_viewed'].corr(df_full_train['interaction_count'])
    opt_4 = df_full_train['annual_income'].corr(df_full_train['interaction_count'])

    arr = np.array([opt_1, opt_2, opt_3, opt_4])
    print(f'2. {np.argmax(arr)}')

    ### mutual information ### 
    question_3_cols = ['industry', 'location', 'lead_source', 'employment_status']
    mutual_info_scores = []
    for i in question_3_cols:
        mutual_info_scores.append(mutual_info_score(df_train['converted'], df_train[i]))

    mutual_info_scores = np.array(mutual_info_scores)
    print(f'3. {np.argmax(mutual_info_scores)}')

    print(f'4. {logistic_regression(df_train, df_val)}')

    features = ['lead_source', 'number_of_courses_viewed', 'interaction_count']
    features_accuracy = []
    for f in features:
        df_train_cpy = df_train.drop(columns=f)
        df_val_cpy = df_val.drop(columns=f)
        features_accuracy.append(logistic_regression(df_train_cpy, df_val_cpy))
    print(f'5. {np.array(features_accuracy).argmax()}')

    C = [0.000001, 0.00001, 0.0001, 0.001]
    regularized = []
    for c in C:
        regularized.append(logistic_regression(
            df_train_org=df_train,
            df_val_org=df_val,
            c = c))
    print(f'6. {np.array(regularized).argmax()}')

def main():
    HERE = Path(__file__).parent.parent
    df = pd.read_csv(HERE / "course_lead_scoring_2026.csv")

    df = df.fillna({
        col: 0 if pd.api.types.is_any_real_numeric_dtype(df[col]) else 'NA'
        for col in df
    })

    print(f'1. {df['industry'].value_counts().index[0]}') 
    exercises(df)

if __name__ == "__main__":
    main()