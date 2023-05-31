
import math
import torch
import numpy as np

from typing import List
from itertools import product

import torch.distributions as D

from mollifiers.my_types import ContinuousDataInfo

class InitialHierarchical:
    def __init__(self, data: np.ndarray, info: ContinuousDataInfo, tau: float, sigma: float, equal_rr: bool = False) -> None:
        """
        """
        if len(info.label_domain) != 2:
            raise ValueError('Y assumed to be binary')

        self.tau = tau
        self.sigma = sigma
        
        # Size of encoding for y, s
        self.y_s_enc_size = info.sensitive_dom_size * info.label_dom_size

        self.info = info
        sens_match_list = [list(s) for s in info.sensitive_domain] # for matching

        # Get positive rate given s
        pos_given_s_dict = {}
        s_dict = {}
        for s in info.sensitive_domain:
            pos_match = (data[:, info.label_columns] == info.positive_label)
            sens_match = (data[:, info.sensitive_columns] == s)

            if equal_rr:
                rate = torch.tensor(1 / self.info.sensitive_dom_size)
            else:
                rate = torch.tensor(np.mean(sens_match))
            
            pos_given_s_dict[sens_match_list.index(list(s))] = np.sum(sens_match * pos_match) / np.sum(sens_match)
            s_dict[sens_match_list.index(list(s))] = rate
            
        # Fix positive rate given s, we increase (a decrease is also valid)
        max_pos_given_s = max(pos_given_s_dict.values())
        for s in info.sensitive_domain:
            pos_given_s_dict[sens_match_list.index(list(s))] = max(
                pos_given_s_dict[sens_match_list.index(list(s))], tau * max_pos_given_s)
            
        # Calculate "fixed" joint prob for y, s
        self.y_s_enc_probs = {}
        self.decoding = {}
        self.encoding = {}
        for i, (y, s) in enumerate(product(info.label_domain, info.sensitive_domain)):
            if y == info.positive_label:
                p = pos_given_s_dict[sens_match_list.index(list(s))]
            else:
                p = 1 - pos_given_s_dict[sens_match_list.index(list(s))]
            
            self.decoding[i] = (y, s)  # For decoding
            self.encoding[(tuple(y), tuple(s))] = i  # For encoding
            self.y_s_enc_probs[i] = p * s_dict[sens_match_list.index(list(s))]
            # s_dict values are already tensors
            
        # Fit Gaussians for x given y_s_enc
        self.x_given_y_s_enc_dist = {}
        for i, (y, s) in enumerate(product(info.label_domain, info.sensitive_domain)):
            label_match = (data[:, info.label_columns] == y)
            sens_match = (data[:, info.sensitive_columns] == s)
            
            cur_x = data[(label_match * sens_match).reshape(-1)][:, info.feature_columns]
            cur_x = torch.tensor(cur_x)

            self.x_given_y_s_enc_dist[i] = D.MultivariateNormal(
                torch.mean(cur_x, axis=0), torch.cov(cur_x.T)
                )

    def prob_y_s_enc(self, y_s_enc: torch.Tensor) -> torch.Tensor:
        probs = torch.tensor([ self.y_s_enc_probs[int(enc)] for enc in y_s_enc ])
        return probs

    def log_prob_y_s_enc(self, y_s_enc: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob_y_s_enc(y_s_enc))

    def log_prob_x_given_y_s_enc(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return torch.vstack([
            self.x_given_y_s_enc_dist[int(enc)].log_prob(x[i]) for i, enc in enumerate(y_s_enc)
        ])
        
    def log_prob(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return self.log_prob_y_s_enc(y_s_enc) + self.log_prob_x_given_y_s_enc(x, y_s_enc).reshape(-1)
        
    def reshape_from_encoding(self, x_samples: torch.Tensor, y_s_enc_samples: torch.Tensor) -> torch.Tensor:
        n_samples = len(x_samples)

        y_samples = []
        s_samples = []
        for enc in y_s_enc_samples:
            y_val, s_val = self.decoding[int(enc)]
            y_samples.append(torch.tensor(y_val, dtype=torch.float32))
            s_samples.append(torch.tensor(s_val, dtype=torch.float32))
        y_samples = torch.vstack(y_samples)
        s_samples = torch.vstack(s_samples)
        
        # Empty tensor to ``reshape'' data
        sampled = torch.zeros(n_samples, self.info.col_size)
        sampled[:, self.info.feature_columns] = x_samples
        sampled[:, self.info.label_columns] = y_samples
        sampled[:, self.info.sensitive_columns] = s_samples
        
        return sampled
    
    def fisher_score_x_given_y_s_enc(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        x.requires_grad_()
        u = self.log_prob_x_given_y_s_enc(x, y_s_enc)
        grad = torch.autograd.grad(u, x)[0]
        x.detach()
        return grad


class EmpiricalDensity(D.Distribution):
    arg_constraints = {}

    def __init__(self, domain: List[np.ndarray], probs: List[float], C: float = 0) -> None:

        norm = sum((1-C) * p + C / len(domain) for p in probs)

        self.probs = torch.Tensor([((1-C) * p + C / len(domain)) / norm for p in probs])
        self.domain = domain

        self.domain_hash = {tuple(value): i for i, value in enumerate(domain)}

    def log_prob(self, value: np.ndarray) -> float:
        #idx = self.domain.index(tuple(value))
        idx = self.domain_hash[tuple(np.array(value))]
        p = self.probs[idx]
        
        try:
            lp = math.log(p)
        except ValueError:
            lp = math.log(1e-9)

        return torch.tensor(lp)

    def rsample(self, sample_shape: torch.Size) -> torch.Tensor:
        ps = self.probs.multinomial(sample_shape.numel(), replacement=True)
        return self.domain[ps.reshape(sample_shape)]


class EmpiricalDiscrete:
    def __init__(self, data: np.ndarray, info: ContinuousDataInfo, mix: float = 0) -> None:
        """
        """
        if len(info.label_domain) != 2:
            raise ValueError('Y assumed to be binary')

        # Size of encoding for y, s
        self.y_s_enc_size = info.sensitive_dom_size * info.label_dom_size

        self.info = info
        sens_match_list = [list(s) for s in info.sensitive_domain] # for matching

        # Get positive rate given s
        pos_given_s_dict = {}
        s_dict = {}
        for s in info.sensitive_domain:
            pos_match = (data[:, info.label_columns] == info.positive_label)
            sens_match = (data[:, info.sensitive_columns] == s)

            rate = torch.tensor(np.mean(sens_match))

            pos_given_s_dict[sens_match_list.index(list(s))] = np.sum(sens_match * pos_match) / np.sum(sens_match)
            s_dict[sens_match_list.index(list(s))] = rate

        # Calculate "fixed" joint prob for y, s
        self.y_s_enc_probs = {}
        self.decoding = {}
        self.encoding = {}
        for i, (y, s) in enumerate(product(info.label_domain, info.sensitive_domain)):
            if y == info.positive_label:
                p = pos_given_s_dict[sens_match_list.index(list(s))]
            else:
                p = 1 - pos_given_s_dict[sens_match_list.index(list(s))]
            
            self.decoding[i] = (y, s)  # For decoding
            self.encoding[(tuple(y), tuple(s))] = i  # For encoding
            self.y_s_enc_probs[i] = p * s_dict[sens_match_list.index(list(s))]
            # s_dict values are already tensors

        # Create and fit Empirical Distributions
        self.x_given_y_s_enc_dist = {}
        for i, (y, s) in enumerate(product(info.label_domain, info.sensitive_domain)):
            label_match = (data[:, info.label_columns] == y)
            sens_match = (data[:, info.sensitive_columns] == s)
            data_mask = (label_match * sens_match).reshape(-1)

            cur_feature_data = data[data_mask][:, info.feature_columns]
            
            vals, cnts = np.unique(cur_feature_data, return_counts=True, axis=0)
            cnts = cnts / len(cur_feature_data)
            count_dict = dict(zip(map(tuple, vals), cnts))
            sg_cond_probs = [(count_dict[d] if d in count_dict else 0) for d in info.feature_domain]

            self.x_given_y_s_enc_dist[i] = EmpiricalDensity(info.feature_domain, sg_cond_probs, C=mix)

    def prob_y_s_enc(self, y_s_enc: torch.Tensor) -> torch.Tensor:
        probs = torch.tensor([ self.y_s_enc_probs[int(enc)] for enc in y_s_enc ])
        return probs

    def log_prob_y_s_enc(self, y_s_enc: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob_y_s_enc(y_s_enc))

    def log_prob_x_given_y_s_enc(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return torch.vstack([
            self.x_given_y_s_enc_dist[int(enc)].log_prob(x[i]) for i, enc in enumerate(y_s_enc)
        ])
        
    def log_prob(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return self.log_prob_y_s_enc(y_s_enc) + self.log_prob_x_given_y_s_enc(x, y_s_enc).reshape(-1)
        
    def reshape_from_encoding(self, x_samples: torch.Tensor, y_s_enc_samples: torch.Tensor) -> torch.Tensor:
        n_samples = len(x_samples)

        y_samples = []
        s_samples = []
        for enc in y_s_enc_samples:
            y_val, s_val = self.decoding[int(enc)]
            y_samples.append(torch.tensor(y_val, dtype=torch.float32))
            s_samples.append(torch.tensor(s_val, dtype=torch.float32))
        y_samples = torch.vstack(y_samples)
        s_samples = torch.vstack(s_samples)
        
        # Empty tensor to ``reshape'' data
        sampled = torch.zeros(n_samples, self.info.col_size)
        sampled[:, self.info.feature_columns] = x_samples
        sampled[:, self.info.label_columns] = y_samples
        sampled[:, self.info.sensitive_columns] = s_samples
        
        return sampled

class InitialDiscrete:
    def __init__(self, data: np.ndarray, info: ContinuousDataInfo, tau: float, equal_rr: bool = False, mix: float = 0.0) -> None:
        """
        """
        if len(info.label_domain) != 2:
            raise ValueError('Y assumed to be binary')

        self.tau = tau
       
        # Size of encoding for y, s
        self.y_s_enc_size = info.sensitive_dom_size * info.label_dom_size

        self.info = info
        sens_match_list = [list(s) for s in info.sensitive_domain] # for matching

        # Get positive rate given s
        pos_given_s_dict = {}
        s_dict = {}
        for s in info.sensitive_domain:
            pos_match = (data[:, info.label_columns] == info.positive_label)
            sens_match = (data[:, info.sensitive_columns] == s)

            if equal_rr:
                rate = torch.tensor(1 / self.info.sensitive_dom_size)
            else:
                rate = torch.tensor(np.mean(sens_match))

            pos_given_s_dict[sens_match_list.index(list(s))] = np.sum(sens_match * pos_match) / np.sum(sens_match)
            s_dict[sens_match_list.index(list(s))] = rate

        # Fix positive rate given s, we increase (a decrease is also valid)
        max_pos_given_s = max(pos_given_s_dict.values())
        for s in info.sensitive_domain:
            pos_given_s_dict[sens_match_list.index(list(s))] = max(
                pos_given_s_dict[sens_match_list.index(list(s))], tau * max_pos_given_s)

        # Calculate "fixed" joint prob for y, s
        self.y_s_enc_probs = {}
        self.decoding = {}
        self.encoding = {}
        for i, (y, s) in enumerate(product(info.label_domain, info.sensitive_domain)):
            if y == info.positive_label:
                p = pos_given_s_dict[sens_match_list.index(list(s))]
            else:
                p = 1 - pos_given_s_dict[sens_match_list.index(list(s))]
            
            self.decoding[i] = (y, s)  # For decoding
            self.encoding[(tuple(y), tuple(s))] = i  # For encoding
            self.y_s_enc_probs[i] = p * s_dict[sens_match_list.index(list(s))]
            # s_dict values are already tensors


        # Create and fit Empirical Distributions
        self.x_given_y_s_enc_dist = {}
        for i, (y, s) in enumerate(product(info.label_domain, info.sensitive_domain)):
            label_match = (data[:, info.label_columns] == y)
            sens_match = (data[:, info.sensitive_columns] == s)
            data_mask = (label_match * sens_match).reshape(-1)

            cur_feature_data = data[data_mask][:, info.feature_columns]

            vals, cnts = np.unique(cur_feature_data, return_counts=True, axis=0)
            cnts = cnts / len(cur_feature_data)
            count_dict = dict(zip(map(tuple, vals), cnts))
            sg_cond_probs = [(count_dict[d] if d in count_dict else 0) for d in info.feature_domain]

            self.x_given_y_s_enc_dist[i] = EmpiricalDensity(info.feature_domain, sg_cond_probs, C=mix)

    def prob_y_s_enc(self, y_s_enc: torch.Tensor) -> torch.Tensor:
        probs = torch.tensor([ self.y_s_enc_probs[int(enc)] for enc in y_s_enc ])
        return probs

    def log_prob_y_s_enc(self, y_s_enc: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob_y_s_enc(y_s_enc))

    def log_prob_x_given_y_s_enc(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return torch.vstack([
            self.x_given_y_s_enc_dist[int(enc)].log_prob(x[i]) for i, enc in enumerate(y_s_enc)
        ])
        
    def log_prob(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return self.log_prob_y_s_enc(y_s_enc) + self.log_prob_x_given_y_s_enc(x, y_s_enc).reshape(-1)
        
    def reshape_from_encoding(self, x_samples: torch.Tensor, y_s_enc_samples: torch.Tensor) -> torch.Tensor:
        n_samples = len(x_samples)

        y_samples = []
        s_samples = []
        for enc in y_s_enc_samples:
            y_val, s_val = self.decoding[int(enc)]
            y_samples.append(torch.tensor(y_val, dtype=torch.float32))
            s_samples.append(torch.tensor(s_val, dtype=torch.float32))
        y_samples = torch.vstack(y_samples)
        s_samples = torch.vstack(s_samples)
        
        # Empty tensor to ``reshape'' data
        sampled = torch.zeros(n_samples, self.info.col_size)
        sampled[:, self.info.feature_columns] = x_samples
        sampled[:, self.info.label_columns] = y_samples
        sampled[:, self.info.sensitive_columns] = s_samples
        
        return sampled
