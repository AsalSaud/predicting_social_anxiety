from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(cat_cols):

    cat_transformer = Pipeline([
        ('encoder' , OneHotEncoder(handle_unknown='ignore', sparse_output = False, drop='if_binary'))
    ])

    preprocessor = ColumnTransformer([
        ('catagorial', cat_transformer, cat_cols)
    ])

    print('building pipeline done')
    return preprocessor
