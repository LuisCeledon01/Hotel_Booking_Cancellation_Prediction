from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from model.processing.features import DateFeatureExtractor

def get_pipeline(estimator_model, config):
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, config.numerical_features),
        ('cat', cat_transformer, config.categorical_features)
    ])

    return Pipeline([
        ('date_extractor', DateFeatureExtractor(config.date_features)),
        ('preprocessor', preprocessor),
        ('classifier', estimator_model)
    ])
