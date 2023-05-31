import numpy as np
import pandas as pd

from typing import Generic, List, Iterator, Tuple, TypeVar
from dataclasses import dataclass

from mollifiers.my_types import DiscreteDataInfo, ContinuousDataInfo, DataInfo, Data

from scipy.io.arff import loadarff
from sklearn.model_selection import KFold
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas, load_preproc_data_adult, load_preproc_data_german

DataInfoT = TypeVar('DataInfoT', bound='DataInfo')

@dataclass
class Dataset(Generic[DataInfoT]):
    data: Data
    info: DataInfoT
    feature_names: List[str]

    def __len__(self):
        return self.data.shape[0]
    
@dataclass
class ContinuousDataset(Dataset[ContinuousDataInfo]):
    
    @property
    def transformed_data(self):
        transformed_x = []
        transformed_y = []
        transformed_s = []

        for m_tuple in self.info.feature_mapping:
            transformed_x.append(m_tuple.map(self.data.x[:,m_tuple.col]))

        for m_tuple in self.info.label_mapping:
            transformed_y.append(m_tuple.map(self.data.s[:,m_tuple.col]))

        for m_tuple in self.info._mapping:
            transformed_s.append(m_tuple.map(self.data.s[:,m_tuple.col]))
        
@dataclass
class DiscreteDataset(Dataset[DiscreteDataInfo]):
    data: np.ndarray
    info: DiscreteDataInfo
    feature_names: List[str]

    def __len__(self):
        return self.data.shape[0]

def generate_domain(cols: np.ndarray) -> np.ndarray:
    # Get unique values
    values = np.unique(cols, axis=0)

    # Turn values into strings for ordering
    str_values = np.apply_along_axis(
        lambda x: ''.join(map(lambda i: str(int(i)), x)), 1, values)

    # Sort for reverse binary order (one hots)
    domain = [tuple(v) for v in values[str_values.argsort()[::-1]]]
    return domain

def aif360_generate_pinfo(dataframe, feature_attributes: List[str], label_attributes: List[str], sensitive_attributes: List[str], positive_label: np.ndarray) -> DiscreteDataInfo:
    np_df = dataframe.to_numpy()
    
    feature_columns = [i for i, c in enumerate(dataframe.columns) if c in feature_attributes]
    label_columns = [i for i, c in enumerate(dataframe.columns) if c in label_attributes]
    sensitive_columns = [i for i, c in enumerate(dataframe.columns) if c in sensitive_attributes]

    feature_domain = generate_domain(np_df[:, feature_columns])
    label_domain = generate_domain(np_df[:, label_columns])
    sensitive_domain = generate_domain(np_df[:, sensitive_columns])

    return DiscreteDataInfo(feature_columns, label_columns, sensitive_columns, feature_domain, label_domain, sensitive_domain, positive_label)

def compas_dataset(sensitive_attributes: List[str]) -> Dataset:
    dataset_orig = load_preproc_data_compas(sensitive_attributes)

    df, info = dataset_orig.convert_to_dataframe()

    # Remove duplicate col
    del df['c_charge_degree=F']
    
    all_cols = list(df.columns)
    label_attributes = info['label_names']
    # sensitive_attributes already defined
    feature_attributes = [c for c in all_cols if c not in label_attributes and c not in sensitive_attributes]

    p_info = aif360_generate_pinfo(df, feature_attributes, label_attributes, sensitive_attributes, np.array([1]))

    print('Dataset column names are:', all_cols)

    return Dataset(df.to_numpy(), p_info, all_cols)

def adult_dataset(sensitive_attributes: List[str]) -> Dataset:
    dataset_orig = load_preproc_data_adult(sensitive_attributes)

    df, info = dataset_orig.convert_to_dataframe()

    all_cols = list(df.columns)
    label_attributes = info['label_names']
    # sensitive_attributes already defined
    feature_attributes = [c for c in all_cols if c not in label_attributes and c not in sensitive_attributes]

    p_info = aif360_generate_pinfo(df, feature_attributes, label_attributes, sensitive_attributes, np.array([1]))

    print('Dataset column names are:', all_cols)

    return Dataset(df.to_numpy(), p_info, all_cols)

def german_dataset(sensitive_attributes: List[str]) -> Dataset:
    dataset_orig = load_preproc_data_german(sensitive_attributes)

    df, info = dataset_orig.convert_to_dataframe()

    all_cols = list(df.columns)
    label_attributes = info['label_names']
    # sensitive_attributes already defined
    feature_attributes = [c for c in all_cols if c not in label_attributes and c not in sensitive_attributes]
    df[label_attributes[0]] = (df[label_attributes[0]] == 1).astype(int)

    p_info = aif360_generate_pinfo(df, feature_attributes, label_attributes, sensitive_attributes, np.array([1]))

    print('Dataset column names are:', all_cols)

    return Dataset(df.to_numpy(), p_info, all_cols)

def dutch_dataset(sensitive_attributes: List[str]) -> Dataset:
    data, meta = loadarff('data/dutch_census_2001.arff')

    # Binarize the domain
    df = pd.DataFrame(data)
    df['sex'] = (df['sex'] == b'1').astype(int)
    df['occupation'] = (df['occupation'] == b'2_1').astype(int)
    df['age'] = (df['age'].astype(int) > 7).astype(int)  # To match other datasets + baselines
    df['cur_eco_activity'] = (df['cur_eco_activity'] != b'200').astype(int)
    df['prev_residence_place'] = (df['prev_residence_place'] == b'1').astype(int)
    df = pd.get_dummies(df, columns=[c for c in df.columns if c not in ['sex', 'occupation', 'age', 'cur_eco_activity', 'prev_residence_place']])
    
    all_cols = list(df.columns)
    label_attributes = ['occupation']

    feature_attributes = [c for c in all_cols if c not in label_attributes and c not in sensitive_attributes]

    p_info = aif360_generate_pinfo(df, feature_attributes, label_attributes, sensitive_attributes, np.array([1]))

    print('Dataset column names are:', all_cols)

    return Dataset(df.to_numpy(), p_info, all_cols)

def dutchlarge_dataset(sensitive_attributes: List[str]) -> Dataset:
    data, meta = loadarff('data/dutch_census_2001.arff')

    # Binarize the domain
    df = pd.DataFrame(data)
    df['sex'] = (df['sex'] == b'1').astype(int)
    df['occupation'] = (df['occupation'] == b'2_1').astype(int)
    df['age'] = (df['age'].astype(int) > 7).astype(int)  # To match other datasets + baselines
    df['prev_residence_place'] = (df['prev_residence_place'] == b'1').astype(int)
    df = pd.get_dummies(df, columns=[c for c in df.columns if c not in ['sex', 'occupation', 'age', 'prev_residence_place']])

    all_cols = list(df.columns)
    label_attributes = ['occupation']

    feature_attributes = [c for c in all_cols if c not in label_attributes and c not in sensitive_attributes]

    p_info = aif360_generate_pinfo(df, feature_attributes, label_attributes, sensitive_attributes, np.array([1]))

    print('Dataset column names are:', all_cols)

    return Dataset(df.to_numpy(), p_info, all_cols)

def minneapolis_dataset() -> Dataset:
    label_name = 'personSearch'
    year_examine = 2017

    data = pd.read_csv('data/Police_Stop_Data.csv', low_memory=False)
    all_cols = ['lat', 'long', 'race', label_name, 'responseDate']
    data = data[all_cols]
    data.dropna(inplace=True)
    data = data[data.race != 'Unknown']

    # Normalization
    data = data[data.long < -13]
    data.lat = (data.lat - data.lat.mean()) / data.lat.std()
    data.long = (data.long - data.long.mean()) / data.long.std()
    data.race = (data.race != 'Black').astype(float)
    data['nocite'] = (data[label_name] == 'NO').astype(float)
    data['year'] = data['responseDate'].apply(lambda x: int(x[0:4]))
    data = data[data.year == year_examine]
    del data[label_name]
    del data['responseDate']
    del data['year']

    feature_columns = [0, 1]
    label_columns = [2]
    sensitive_columns = [3]

    label_domain = [np.array([1]), np.array([0])]
    sensitive_domain = [np.array([1]), np.array([0])]

    pos_label = np.array([1])

    info = ContinuousDataInfo(feature_columns, label_columns, sensitive_columns, label_domain, sensitive_domain, pos_label)
    data = data.to_numpy()

    print('Dataset column names are:', all_cols)

    return Dataset(data, info, all_cols)

def synth_dataset(n_samples = 1_000, mu_0 = 0.0, mu_1 = 0.7, std = 1.0, y_cond_s0 = 0.6, y_cond_s1 = 0.8
                  ) -> Dataset:
    feature_columns = [0, 1]
    label_columns = [2]
    sensitive_columns = [3]

    all_cols = ['x0', 'x1', 'y', 's']

    label_domain = [np.array([1]), np.array([0])]
    sensitive_domain = [np.array([1]), np.array([0])]

    pos_label = np.array([1])

    info = ContinuousDataInfo(feature_columns, label_columns, sensitive_columns, label_domain, sensitive_domain, pos_label)

    x_0 = np.random.multivariate_normal([mu_0, mu_0], std * np.eye(2), n_samples * 2)
    x_1 = np.random.multivariate_normal([mu_1, mu_1], std * np.eye(2), n_samples * 2)
    y_0 = np.random.binomial(1, y_cond_s0, n_samples * 2).reshape(-1, 1)
    y_1 = np.random.binomial(1, y_cond_s1, n_samples * 2).reshape(-1, 1)
    s_0 = np.zeros_like(y_0)
    s_1 = np.ones_like(y_1)

    x = np.concatenate([x_0, x_1])
    y = np.concatenate([y_0, y_1])
    s = np.concatenate([s_0, s_1])
    data = np.hstack([x, y, s])

    print('Dataset column names are:', all_cols)

    return Dataset(data, info, all_cols)

def cross_validate_dataset(dataset: Dataset[DataInfoT], n_splits: int, random_state: int) -> Iterator[Tuple[Dataset[DataInfoT], Dataset[DataInfoT]]]:
    # Find the indices
    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Split via indices
    for train_indices, test_indices in cv.split(range(len(dataset))):
        train_data = dataset.data[train_indices, :]
        test_data = dataset.data[test_indices, :]

        train_dataset = Dataset[DataInfoT](train_data, dataset.info,
                                dataset.feature_names)
        test_dataset = Dataset[DataInfoT](test_data, dataset.info,
                               dataset.feature_names)

        yield train_dataset, test_dataset