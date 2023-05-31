import torch
import torch.distributions as D
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, Tuple
from functools import partial
from tqdm import tqdm
#from tqdm.autonotebook import tqdm

from mollifiers.mollifier import DiscreteFairDensity

class NotBurntIn(Exception):
    pass

class Sampler(ABC):
    @abstractmethod
    def sample(self, n_samples: int):
        pass

class ULA(Sampler):
    """ Unadjusted Langevin Algorithm
    """
    
    def __init__(self, score_function: Callable[[torch.Tensor], torch.Tensor],
                 x_size: int, step: float = 0.01, burn_in_n: int = 10_000) -> None:
        super().__init__()

        self.score_function = score_function
        self.x_size = x_size
        self.step = step
        self.burn_in_n = burn_in_n
        
        self.is_burnt_in = False
        self.xi = torch.randn(1, self.x_size)
        
        # Stat stuff
        self.burn_in_samps = None
        
    def reset(self) -> None:
        self.xi = torch.randn(1, self.x_size)

        self.is_burnt_in = False
        
    def burn_in(self) -> None:
        self.burn_in_samps = self._sample(self.burn_in_n)
        self.is_burnt_in = True

    def sample(self, n_samples: int) -> torch.Tensor:
        if not self.is_burnt_in:
            raise NotBurntIn
        return self._sample(n_samples)

    def _sample(self, n_samples: int) -> torch.Tensor:
        """ Based on "https://github.com/abdulfatir/langevin-monte-carlo"
        """
        xi_samps = []
        for _ in tqdm(range(n_samples)):
            x_grad = self.score_function(self.xi)
            self.xi = self.xi + self.step * x_grad + np.sqrt(2 * self.step) * torch.randn(1, self.x_size)
            
            xi_samps.append(self.xi)

        return torch.cat(xi_samps, axis=0).detach()
    

class HierarchicalULA(Sampler):

    def __init__(self, marginal_probs: torch.Tensor,
                 cond_score_function: Callable[[torch.Tensor, int], torch.Tensor],
                 x_size: int, step: float = 0.01, burn_in_n: int = 10_000) -> None:
        """ Assume that `cond_score_function` takes features x as first argument,
            and conditional encoding as second argument.
            
            The function will sample from the `marginal_probs` first to create encoding values,
            then utilize ULA to sample each of the conditional score functions.
            
        """
        super().__init__()

        self.marginal_probs = marginal_probs
        self.cond_score_function = cond_score_function

        self.x_size = x_size
        self.step = step
        self.burn_in_n = burn_in_n
        
        self.lmc_dict: Callable[[int], ULA] = {}
        for enc in range(len(self.marginal_probs)):
            cur_score_function = partial(self.cond_score_function, y_s_enc=enc)
            
            cur_sampler = ULA(cur_score_function, x_size, step, burn_in_n)

            self.lmc_dict[enc] = cur_sampler
        self.marginal_sampler = D.Categorical(self.marginal_probs)
        
    def init(self) -> None:
        for s in self.lmc_dict.values():
            s.burn_in()
            
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_samples = self.marginal_sampler.sample((n_samples,))
        enc_counts = torch.bincount(enc_samples)
        
        marg_samples = []
        cond_samples = []
        for enc, enc_sample_n in enumerate(enc_counts):
            cur_cond_samples = self.lmc_dict[enc].sample(enc_sample_n)
            
            marg_samples.append(torch.tensor([[enc]] * enc_sample_n))
            cond_samples.append(cur_cond_samples)
            
        return torch.cat(cond_samples, dim=0), torch.cat(marg_samples, dim=0)
    

class DiscreteSampler(Sampler):

    def __init__(self, density: DiscreteFairDensity) -> None:
        super().__init__()

        self.density = density
        self.probs = []
        self.input_hash = {}

        self.sampler = None
        
    def init(self) -> None:
        idx = 0
        for i in range(self.density.data_info.sensitive_dom_size * self.density.data_info.label_dom_size): 
            for x in self.density.data_info.feature_domain:
                cur_input = self.density.rejoin_samples(torch.tensor([x], dtype=torch.float32), torch.tensor([[i]], dtype=torch.float32))
                cur_prob = np.exp(self.density.log_prob(cur_input))

                self.probs.append(cur_prob)
                self.input_hash[idx] = cur_input

                idx += 1
                
        self.sampler = D.Categorical(probs = torch.tensor(self.probs))
        
    def sample(self, n_samples: int) -> torch.Tensor:
        idx_samples = self.sampler.sample((n_samples,))
        return torch.vstack([self.input_hash[int(i)] for i in idx_samples])
