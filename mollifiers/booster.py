import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Tuple

from mollifiers.mollifier import DiscreteFairDensity, FairDensity, HierarchicalFairDensity, DiscreteEmpiricalDensity
from mollifiers.my_types import ContinuousDataInfo, DiscreteDataInfo
from mollifiers.leverage import LeverageSchedule
from mollifiers.hypothesis import HypothesisClass
from mollifiers.sampler import DiscreteSampler, HierarchicalULA

def generate_adversarial_labels(true_samples: np.ndarray,
                                fake_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    train_x = np.concatenate([true_samples, fake_samples], axis=0)
    train_y = np.concatenate([np.ones((len(true_samples), 1)),
                              np.zeros((len(fake_samples), 1))], axis=0)

    return torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)

class AbstractBoostedDensityEstimator(ABC):
    log: List[ Dict[str, Any] ]
    q : FairDensity

    @abstractmethod
    def step(self, data: np.ndarray) -> None:
        pass

        
class HierarchicalBoostedDensityEstimator(AbstractBoostedDensityEstimator):
    def __init__(self, data: np.ndarray, info: ContinuousDataInfo, tau: float,
                 hypothesis_class: HypothesisClass, leverage_schedule: LeverageSchedule,
                 sr_init: float = 1.0, burn_in_n: int = 100_000, equal_rr: bool = False) -> None:
        super().__init__()

        # Data + info
        self.data = data
        self.info = info
        
        # Fairness parameters
        self.tau = tau
        self.sr_init = sr_init
        self.equal_rr = equal_rr

        self.cur_iter = None
        
        # Density
        self.density = None # To initialize

        self.hypothesis_class = hypothesis_class
        self.leverage_schedule = leverage_schedule
        
        # Save last training examples (for stats)
        self.burn_in_n = burn_in_n

        self.fake_sampler = None
        
    def init(self, normalizer_samples_n: int = 100_000) -> None:
        self.density = HierarchicalFairDensity(self.tau, self.info, self.sr_init,
            equal_rr = self.equal_rr)
        self.density.init(self.data, normalizer_samples_n)
        self.cur_iter = 1

    def reset_sampler(self):
        self.fake_sampler = None

    def get_cur_samples(self, n_samples: int):
        if self.fake_sampler is None:
            marginal_probs = self.density.marginal_prob_enc_vector()

            self.fake_sampler = HierarchicalULA(
                marginal_probs, self.density.fisher_score_x_given_y_s_enc,
                self.info.feature_col_size, burn_in_n=self.burn_in_n)
            self.fake_sampler.init()

        x_samples, enc_samples = self.fake_sampler.sample(n_samples)
        samples = self.density.rejoin_samples(x_samples, enc_samples)
        return samples
        
    def step(self, n_fake_samples: Optional[int] = None) -> None:
        if n_fake_samples is None:
            n_fake_samples = len(self.data)
        
        # generate fake vs real labels 
        fake_samples = self.get_cur_samples(n_fake_samples)
        training_x, training_y = generate_adversarial_labels(self.data, fake_samples)

        # Generate hypothesis
        cur_model = self.hypothesis_class.generate_hypothesis(training_x, training_y)
        cur_leverage = self.leverage_schedule.leverage(self.cur_iter, self.tau, self.sr_init)

        self.density.append(cur_model, cur_leverage)
        self.density.normalize()

        # Reset sampler
        self.reset_sampler()

        # Iterate number here so incomplete computation doesn't break loop
        self.cur_iter += 1
        

class DiscreteBoostedDensityEstimator(AbstractBoostedDensityEstimator):
    def __init__(self, data: np.ndarray, info: DiscreteDataInfo, tau: float,
                 hypothesis_class: HypothesisClass, leverage_schedule: LeverageSchedule,
                 sr_init: float = 1.0, equal_rr: bool = False, mix: float = 0.0) -> None:
        super().__init__()

        # Data + info
        self.data = data
        self.info = info
        
        # Fairness parameters
        self.tau = tau
        self.sr_init = sr_init
        self.equal_rr = equal_rr
        self.mix = mix

        self.cur_iter = None
        
        # Density
        self.density = None # To initialize

        self.hypothesis_class = hypothesis_class
        self.leverage_schedule = leverage_schedule

        self.fake_sampler = None
        
    def init(self) -> None:
        self.density = DiscreteFairDensity(self.tau, self.info, self.sr_init, equal_rr = self.equal_rr, mix=self.mix)
        self.density.init(self.data)
        self.cur_iter = 1

    def reset_sampler(self):
        self.fake_sampler = None

    def get_cur_samples(self, n_samples: int):
        if self.fake_sampler is None:
            self.fake_sampler = DiscreteSampler(self.density)
            self.fake_sampler.init()

        samples = self.fake_sampler.sample(n_samples)
        return samples
        
    def step(self, n_fake_samples: Optional[int] = None) -> None:
        if n_fake_samples is None:
            n_fake_samples = len(self.data)
        
        # generate fake vs real labels 
        fake_samples = self.get_cur_samples(n_fake_samples)
        training_x, training_y = generate_adversarial_labels(self.data, fake_samples)
        
        # Generate hypothesis
        cur_model = self.hypothesis_class.generate_hypothesis(training_x, training_y)
        cur_leverage = self.leverage_schedule.leverage(self.cur_iter, self.tau, self.sr_init)

        self.density.append(cur_model, cur_leverage)
        self.density.normalize()
        
        # Reset sampler
        self.reset_sampler()
        
        # Iterate number here so incomplete computation doesn't break loop
        self.cur_iter += 1
        

class DiscreteEmpiricalDensityEstimator(AbstractBoostedDensityEstimator):
    def __init__(self, data: np.ndarray, info: DiscreteDataInfo) -> None:
        super().__init__()

        # Data + info
        self.data = data
        self.info = info

        self.cur_iter = None
        
        # Density
        self.density = None # To initialize

        self.fake_sampler = None
        
    def init(self) -> None:
        self.density = DiscreteEmpiricalDensity(self.info)
        self.density.init(self.data)
        self.cur_iter = 1

    def reset_sampler(self):
        self.fake_sampler = None

    def get_cur_samples(self, n_samples: int):
        return self.data
        
    def step(self, n_fake_samples: Optional[int] = None) -> None:
        pass
