from itertools import product
import math
import torch
import numpy as np

from abc import ABC
from typing import Callable, List, NamedTuple, Tuple
from dataclasses import astuple, dataclass

C_CONST = math.log(2)

class Data(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    
DiscreteDomain = List[Tuple[int]]

class DomainMap(NamedTuple):
    col: List[int]
    map: Callable[[np.ndarray], np.ndarray]

@dataclass
class DataInfo(ABC):
    pass

@dataclass
class ContinuousDataInfo(DataInfo):
    feature_columns: List[int]  # X
    label_columns: List[int]  # Y
    sensitive_columns: List[int]  # S

    label_domain: List[np.ndarray]  # Y
    sensitive_domain: List[np.ndarray]  # S
    
    positive_label: np.ndarray  # Y = +1
    
    @property
    def col_size(self) -> int:
        return len(self.feature_columns) + len(self.label_columns) + len(self.sensitive_columns)

    @property
    def feature_col_size(self) -> int:
        return len(self.feature_columns)
    
    @property
    def label_col_size(self) -> int:
        return len(self.label_columns)
    
    @property
    def sensitive_col_size(self) -> int:
        return len(self.sensitive_columns)

    @property
    def label_dom_size(self) -> int:
        return len(self.label_domain)
    
    @property
    def sensitive_dom_size(self) -> int:
        return len(self.sensitive_domain)
    
    @property
    def unlabel_columns(self) -> List[int]:
        return [i for i in range(self.col_size) if i not in self.label_columns]

@dataclass
class DiscreteDataInfo(DataInfo):
    feature_columns: List[int]  # X
    label_columns: List[int]  # Y
    sensitive_columns: List[int]  # S

    feature_domain: List[np.ndarray]  #X
    label_domain: List[np.ndarray]  # Y
    sensitive_domain: List[np.ndarray]  # S
    
    positive_label: np.ndarray  # Y = +1
    
    @property
    def col_size(self) -> int:
        return len(self.feature_columns) + len(self.label_columns) + len(self.sensitive_columns)

    @property
    def feature_col_size(self) -> int:
        return len(self.feature_columns)
    
    @property
    def label_col_size(self) -> int:
        return len(self.label_columns)
    
    @property
    def sensitive_col_size(self) -> int:
        return len(self.sensitive_columns)

    @property
    def label_dom_size(self) -> int:
        return len(self.label_domain)
    
    @property
    def sensitive_dom_size(self) -> int:
        return len(self.sensitive_domain)
    
    @property
    def unlabel_columns(self) -> List[int]:
        return [i for i in range(self.col_size) if i not in self.label_columns]
    
    @property
    def full_domain(self) -> List[np.ndarray]:
        domain = []
        for x, y, s in product(self.feature_domain, self.label_domain, self.sensitive_domain):
            cur_val = np.zeros(self.col_size)
            cur_val[self.feature_columns] = x
            cur_val[self.label_columns] = y
            cur_val[self.sensitive_columns] = s
            
            domain.append(cur_val)
            
        return domain

@dataclass
class ContinuousDataInfoTransformer(DataInfo):
    feature_columns: List[int]  # X
    label_columns: List[int]  # Y
    sensitive_columns: List[int]  # S

    # Provides a named list of functions (per column) which maps the original domain to R
    feature_mapping: List[DomainMap]
    label_mapping: List[DomainMap]
    sensitive_mapping: List[DomainMap]

    # Inverse of above
    feature_inv_mapping: List[DomainMap]
    label_inv_mapping: List[DomainMap]
    sensitive_inv_mapping: List[DomainMap]
    
    def transform_data(self, data: Data) -> Data:
        transformed_x = []
        transformed_y = []
        transformed_s = []

        for m_tuple in self.feature_mapping:
            transformed_x.append(m_tuple.map(data.x[:, m_tuple.col]))

        for m_tuple in self.label_mapping:
            transformed_y.append(m_tuple.map(data.s[:, m_tuple.col]))

        for m_tuple in self.sensitive_mapping:
            transformed_s.append(m_tuple.map(data.s[:, m_tuple.col]))
            
        return Data(np.hstack(transformed_x), np.hstack(transformed_y), np.hstack(transformed_s))

    def inv_transform_data(self, data: Data) -> Data:
        inv_transformed_x = []
        inv_transformed_y = []
        inv_transformed_s = []

        for m_tuple in self.feature_inv_mapping:
            inv_transformed_x.append(m_tuple.map(data.x[:, m_tuple.col]))

        for m_tuple in self.label_inv_mapping:
            inv_transformed_y.append(m_tuple.map(data.s[:, m_tuple.col]))

        for m_tuple in self.sensitive_inv_mapping:
            inv_transformed_s.append(m_tuple.map(data.s[:, m_tuple.col]))
            
        return Data(np.hstack(inv_transformed_x), np.hstack(inv_transformed_y), np.hstack(inv_transformed_s))
