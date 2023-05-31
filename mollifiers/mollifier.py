import math
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from mollifiers.init import InitialDiscrete, InitialHierarchical, EmpiricalDiscrete
from mollifiers.my_types import DataInfo, ContinuousDataInfo, DiscreteDataInfo
from mollifiers.hypothesis import Hypothesis

class FairDensity(ABC):
    arg_constraints = {}
    data_info: DataInfo
    models: List[Hypothesis] = []
    thetas: List[float] = []

    @abstractmethod
    def append(self, model: Hypothesis, theta: float) -> None:
        pass
    

class HierarchicalFairDensity(FairDensity):
    def __init__(self, tau: float, info: ContinuousDataInfo, sr_init: float = 1.0, equal_rr: bool = False) -> None:
        super().__init__()

        self.tau = tau
        self.data_info = info
        self.sr_init = sr_init
        self.equal_rr = equal_rr
        
        self.joint_zs = []
        self.conditional_zs = { enc: [] for enc in range(info.sensitive_dom_size * info.label_dom_size) }

        self.init_distribution = None
        self.initial_enc_samples = { enc: None for enc in \
            range(info.sensitive_dom_size * info.label_dom_size) }
        
        self.normalized = False
        
    def init(self, data: np.ndarray, normalizer_samples_n: int = 100_000) -> None:
        self.models = []
        self.thetas = []
        self.normalized = True

        self.init_distribution = InitialHierarchical(
            data, self.data_info, self.sr_init, sigma = 1.0,
            equal_rr = self.equal_rr)

        for i in self.initial_enc_samples.keys():
            self.initial_enc_samples[i] = self.init_distribution.x_given_y_s_enc_dist[i].sample(
                (normalizer_samples_n,)).to(torch.float32)
        
        
    def append(self, model: Hypothesis, theta: float) -> None:
        self.models.append(model)
        self.thetas.append(theta)
        
        self.normalized = False
        
    def normalize(self):
        Ztot = 0
        total_count = sum(len(v) for v in self.initial_enc_samples.values())
        for i, cur_x in self.initial_enc_samples.items():
            cur_y_s_enc = torch.ones(len(cur_x), 1) * i

            cur_samples = self.init_distribution.reshape_from_encoding(cur_x, cur_y_s_enc)

            acc = torch.zeros(cur_samples.shape[0], 1)
            for theta, m in zip(self.thetas, self.models):
                acc += theta * m(cur_samples)

            Zi = torch.mean(torch.exp(acc))
            self.conditional_zs[i].append(Zi)

            Ztot += torch.sum(torch.exp(acc))
            
        self.joint_zs.append(math.log(Ztot / total_count))
        
        self.normalized = True
        
    def marginal_prob_y_s_enc(self, y_s_enc: torch.Tensor) -> torch.Tensor:
        if not self.normalized:
            raise AttributeError("Not normalized")

        probs = torch.tensor([ self.init_distribution.prob_y_s_enc([int(enc)]) \
            for enc in y_s_enc ])

        if len(self.models) > 0:
            probs *= torch.tensor([ (self.conditional_zs[int(enc)][-1]) for enc in y_s_enc ])

        probs  /= torch.sum(probs)

        return probs
    
    def marginal_prob_enc_vector(self) -> torch.Tensor:
        return self.marginal_prob_y_s_enc(
            torch.tensor(range(self.init_distribution.y_s_enc_size))).reshape(-1)

    def log_prob(self, data: torch.Tensor) -> torch.Tensor:
        x = data[:, self.data_info.feature_columns]
        
        ys = data[:, self.data_info.label_columns]
        ss = data[:, self.data_info.sensitive_columns]

        y_s_enc = torch.tensor([
            self.init_distribution.encoding[(tuple(y.numpy()), tuple(s.numpy()))] \
                for y, s in zip(ys, ss)])

        log_prob = self.init_distribution.log_prob(x, y_s_enc)
        for theta, m in zip(self.thetas, self.models):
            log_prob += theta * m(data.to(torch.float32)).reshape(-1)
            
        if self.joint_zs:
            log_prob -= math.log(self.joint_zs[-1])
        
        return log_prob

        
    def fisher_score_x_given_y_s_enc(self, x: torch.Tensor, y_s_enc: int) -> torch.Tensor:
        tot_grad = self.init_distribution.fisher_score_x_given_y_s_enc(x, torch.tensor([y_s_enc]))
        data = self.init_distribution.reshape_from_encoding(x, torch.tensor([y_s_enc]))

        for theta, m in zip(self.thetas, self.models):
            tot_grad += theta * m.gradient(data)[:, self.data_info.feature_columns]
            
        return tot_grad
    
    def rejoin_samples(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return self.init_distribution.reshape_from_encoding(x, y_s_enc)


class DiscreteFairDensity(FairDensity):
    def __init__(self, tau: float, info: DiscreteDataInfo, sr_init: float = 1.0, equal_rr: bool = False, mix: float = 0.0) -> None:
        super().__init__()

        self.tau = tau
        self.data_info = info
        self.sr_init = sr_init
        self.equal_rr = equal_rr
        self.mix = mix
        
        self.tot_zs = []

        self.init_distribution = None
        
        self.normalized = False
        
    def init(self, data: np.ndarray) -> None:
        self.models = []
        self.thetas = []
        self.tot_zs = []
        self.normalized = True

        self.init_distribution = InitialDiscrete(
            data, self.data_info, self.sr_init, equal_rr = self.equal_rr, mix=self.mix)
        
    def append(self, model: Hypothesis, theta: float) -> None:
        self.models.append(model)
        self.thetas.append(theta)
        
        self.normalized = False
        
    def normalize(self): 
        Ztot = 0
        for i in range(self.data_info.sensitive_dom_size * self.data_info.label_dom_size): 
            z_i = 0
            for xys in self.data_info.full_domain:
                xys = np.array(xys).reshape(1, -1)
                x = xys[:, self.data_info.feature_columns]
                acc = self.init_distribution.log_prob(x, torch.tensor([i])).squeeze()
                for theta, m in zip(self.thetas, self.models):
                    acc += (theta * m(xys)).squeeze()

                z_i += torch.exp(acc)
            Ztot += z_i

        self.tot_zs.append(math.log(Ztot))
        
        self.normalized = True
        
    def log_prob(self, data: torch.Tensor) -> torch.Tensor:
        if type(data) is not torch.Tensor:
            data = torch.tensor(data)

        x = data[:, self.data_info.feature_columns]
        
        ys = data[:, self.data_info.label_columns]
        ss = data[:, self.data_info.sensitive_columns]

        y_s_enc = [self.init_distribution.encoding[(tuple(y.numpy()), tuple(s.numpy()))] for y, s in zip(ys, ss)]
        y_s_enc = torch.tensor(y_s_enc)

        log_prob = self.init_distribution.log_prob(x, y_s_enc)
        for theta, m in zip(self.thetas, self.models):
            log_prob += theta * m(data.to(torch.float32)).reshape(-1)
            
        if self.tot_zs:
            log_prob -= self.tot_zs[-1]
        
        return log_prob

    def rejoin_samples(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
        return self.init_distribution.reshape_from_encoding(x, y_s_enc)


class DiscreteEmpiricalDensity(FairDensity):
    def __init__(self, info: DiscreteDataInfo) -> None:
        super().__init__()

        self.data_info = info
        
        self.tot_zs = []

        self.init_distribution = None
        
    def init(self, data: np.ndarray, mix: float = 0) -> None:
        self.models = []
        self.thetas = []
        self.tot_zs = []
        self.normalized = True

        self.init_distribution = EmpiricalDiscrete(data, self.data_info, mix=mix)
        
    def append(self, model: Hypothesis, theta: float) -> None:
        pass
        
    def normalize(self): 
        pass
        
    def log_prob(self, data: torch.Tensor) -> torch.Tensor:
        if type(data) is not torch.Tensor:
            data = torch.tensor(data)

        x = data[:, self.data_info.feature_columns]
        
        ys = data[:, self.data_info.label_columns]
        ss = data[:, self.data_info.sensitive_columns]

        y_s_enc = [self.init_distribution.encoding[(tuple(y.numpy()), tuple(s.numpy()))] for y, s in zip(ys, ss)]
        y_s_enc = torch.tensor(y_s_enc)

        log_prob = self.init_distribution.log_prob(x, y_s_enc)
        for theta, m in zip(self.thetas, self.models):
            log_prob += theta * m(data.to(torch.float32)).reshape(-1)
            
        if self.tot_zs:
            log_prob -= self.tot_zs[-1]
        
        return log_prob

    def rejoin_samples(self, x: torch.Tensor, y_s_enc: torch.Tensor) -> torch.Tensor:
       return self.init_distribution.reshape_from_encoding(x, y_s_enc)
