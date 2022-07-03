import pandas as pd
from sklearn import preprocessing


def load_data(dataset, attributes, sensitive_attribute, normalize=False, standardize=False, sample=False):
    """Loads the data set and prepares the data

    Args:
        dataset (str): path and name of the data set
        attributes (list): name(s) of attributes used in the clustering
        sensitive_attribute (list): name(s) of protected attribute(s)
        normalize (bool): normalize date i.e. values will range between 0 and 1
        standardize (bool): standardize data i.e. values will have a mean of 0 and a standard deviation of 1
        sample (int): randomly sample a group of objects from the DataFrame

    Returns:
        pd.DataFrame: adjusted data set

    """

    # Read dataset
    df = pd.read_csv(dataset)

    # Adjust specific datasets
    if "diabetes" in dataset:
        age_buckets = {'[70-80)': 75, '[60-70)': 65, '[50-60)': 55,
                       '[80-90)': 85, '[40-50)': 45, '[30-40)': 35,
                       '[90-100)': 95, '[20-30)': 25, '[10-20)': 15, '[0-10)': 5}
        df['age'] = df.apply(lambda x: age_buckets[x['age']], axis=1)

    # Group the values 'divorced' and 'single' into a common category
    if "bank" in dataset:
        df['marital'] = df.marital.str.replace("divorced", "single")

    # Store sensitive attributes
    sensitive = df[sensitive_attribute]

    # Store remaining attributes
    df = df[attributes]

    # Factorize sensitive attribute (0/1 values)
    sensitive[sensitive_attribute] = sensitive[sensitive_attribute].apply(lambda x: pd.factorize(x)[0])

    # Normalize data
    if normalize:
        minmax_scaler = preprocessing.MinMaxScaler()
        df = pd.DataFrame(minmax_scaler.fit_transform(df), index=df.index, columns=df.columns)

    # Standardize data:
    if standardize:
        standard_scaler = preprocessing.StandardScaler()
        df = pd.DataFrame(standard_scaler.fit_transform(df), index=df.index, columns=df.columns)

    # Sample data:
    if type(sample) == int:
        df = df.sample(n=sample, replace=False, random_state=24)

    # Append sensitive attribute to DataFrame
    df = pd.concat([df, sensitive], axis=1)

    # Return DataFrame
    return df
