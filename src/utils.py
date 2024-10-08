import pandas as pd

def extract_features_from_vector(df, selected_features):
    feature_columns = pd.DataFrame(df['features'].tolist(), columns=selected_features)
    df_with_features = pd.concat([df, feature_columns], axis=1)
    df_with_features = df_with_features.drop(columns=['features'])
    return df_with_features