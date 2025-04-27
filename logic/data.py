import pandas as pd
from sklearn.model_selection import train_test_split

# load the dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(['Anxiety Level (1-10)'],axis=1)
    y = df['Anxiety Level (1-10)']

    print("feature and target sets are ready")
    return X, y, df

def cat_num(X):
    num = X.select_dtypes(include=["int64" , "float64"]).columns
    cat = X.select_dtypes(include=["object"]).columns

    print("categorical and numerical columns splited successfully")
    return num, cat


def split(X, y):
    print("data splitted into test and train successfully")
    return train_test_split(X, y, test_size=0.2, random_state=42)
