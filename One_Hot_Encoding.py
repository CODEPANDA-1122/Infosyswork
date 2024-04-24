import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from Reading_Training_data import dataframe

# Select the columns you want to one-hot encode
columns_to_encode = ['gender','residence','location']

# Apply one-hot encoding
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(dataframe[columns_to_encode])

# Convert the encoded data to a DataFrame
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(columns_to_encode))

# Concatenate the original DataFrame with the encoded DataFrame
df_encoded = pd.concat([dataframe.drop(columns=columns_to_encode), encoded_df], axis=1)

# Now df_encoded contains the original data with one-hot encoded parameters
print(df_encoded)
