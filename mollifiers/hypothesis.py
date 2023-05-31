import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV


from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
#from tqdm.autonotebook import tqdm
from tqdm import tqdm
from mollifiers.my_types import ContinuousDataInfo, DataInfo, DiscreteDataInfo
from mollifiers.mlp import ClippedModule, OneHiddenLayerMLP, TwoHiddenLayerMLP

class Hypothesis(ABC):
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def original_object(self) -> Any:
        pass
    
    @abstractmethod
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)
    
class HypothesisClass(ABC):
    info: DataInfo

    @abstractmethod
    def generate_hypothesis(self, x: np.ndarray, y: np.ndarray) -> Hypothesis:
        pass
    
class TorchHypothesis(Hypothesis):

    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(self, torch_obj: nn.Module, wl_acc: Optional[float] = None, wl_loss: Optional[float] = None) -> None:
        super().__init__()
        self.model = torch_obj
        self.wl_acc = wl_acc
        self.wl_loss = wl_loss
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @property
    def original_object(self) -> nn.Module:
        return self.model
    
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x.detach()
        x.requires_grad_()
        output = self.model(x)
        grads = torch.autograd.grad(output, x)[0]
        x.detach()
        return grads
    
class ClippedOneHiddenLayerHypothesisClass(HypothesisClass):
    def __init__(self, info: ContinuousDataInfo, hidden_size: int, clip: float,
                 batch_size: int = 64, device: str = 'cpu',
                 epochs: int = 32, lr: float = 1.0,
                 gamma: float = 0.7, log_interval: int = 10) -> None:
        super().__init__()
        
        # Data info
        self.info: ContinuousDataInfo = info
        
        # MLP info
        self.hidden_size = hidden_size

        self.clip = clip
        
        # Loss
        self.loss = F.binary_cross_entropy_with_logits
        
        # Training params
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.log_interval = log_interval
        
        self.device = torch.device(device)

    def generate_hypothesis(self, x: np.ndarray, y: np.ndarray) -> Hypothesis:
        # Parameters for the loader
        train_kwargs = {'batch_size': self.batch_size,
                        'shuffle': True}
        if self.device.type == 'cuda':
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True}
            train_kwargs.update(cuda_kwargs)

        # For data
        dataloader = DataLoader(list(zip(x, y)), **train_kwargs)
        
        # Define model and optimizer
        _model = OneHiddenLayerMLP(self.info.col_size, self.hidden_size).to(self.device)
        optimizer = optim.Adadelta(_model.parameters(), lr=self.lr)

        # Training loop
        _model.train()
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        pbar = tqdm(range(1, self.epochs + 1))
        for epoch in pbar:
        #for epoch in (pbar := tqdm(range(1, self.epochs + 1))):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()

                output = _model(data)
                loss = self.loss(output, target)

                loss.backward()
                optimizer.step()

            scheduler.step()
        _model.eval()
        model = ClippedModule(_model, self.clip)
        model.eval()
        

        correct = (model(x) > 0) == y
        wl_acc = float(correct.float().mean())
        wl_loss = float(self.loss(model(x), y))
        return TorchHypothesis(model, wl_acc=wl_acc, wl_loss=wl_loss)

class ClippedTwoHiddenLayerHypothesisClass(HypothesisClass):
    def __init__(self, info: ContinuousDataInfo, hidden_size: int, clip: float,
                 batch_size: int = 64, device: str = 'cpu',
                 epochs: int = 32, lr: float = 1.0,
                 gamma: float = 0.7, log_interval: int = 10) -> None:
        super().__init__()
        
        # Data info
        self.info: ContinuousDataInfo = info
        
        # MLP info
        self.hidden_size = hidden_size
        self.clip = clip
        
        # Loss
        self.loss = F.binary_cross_entropy_with_logits
        
        # Training params
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.log_interval = log_interval
        
        self.device = torch.device(device)

    def generate_hypothesis(self, x: np.ndarray, y: np.ndarray) -> Hypothesis:
        # Parameters for the loader
        train_kwargs = {'batch_size': self.batch_size,
                        'shuffle': True}
        if self.device.type == 'cuda':
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True}
            train_kwargs.update(cuda_kwargs)

        # For data
        dataloader = DataLoader(list(zip(x, y)), **train_kwargs)
        
        # Define model and optimizer
        _model = TwoHiddenLayerMLP(self.info.col_size, self.hidden_size).to(self.device)
        model = ClippedModule(_model, self.clip)
        optimizer = optim.Adadelta(model.parameters(), lr=self.lr)

        # Training loop
        model.train()
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        pbar = tqdm(range(1, self.epochs + 1))
        for epoch in pbar:
        #for epoch in (pbar := tqdm(range(1, self.epochs + 1))):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()

                output = model(data)
                loss = self.loss(output, target)

                loss.backward()
                optimizer.step()

                # Report
                if batch_idx % self.log_interval == 0:
                    pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader), loss.item()))
            scheduler.step()
        model.eval()
        
        correct = (model(x) > 0) == y
        wl_acc = float(correct.float().mean())
        wl_loss = float(self.loss(model(x), y))
        return TorchHypothesis(model, wl_acc=wl_acc, wl_loss=wl_loss)
    

class ProbHypothesis(Hypothesis):
    def __init__(self, prob_func: Callable[[np.ndarray], np.ndarray], clip: float, prob_object: Optional[Any] = None, wl_acc: Optional[float] = None) -> None:
        self.prob_func = prob_func
        self.prob_object = prob_object
        self.clip = clip
        self.wl_acc = wl_acc

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Rescale a probability prediction to [-C, C]
        if len(x.shape) > 1:
            ret_val = (self.prob_func(x) * 2 - 1) * self.clip
        else:
            ret_val = (self.prob_func(x) * 2 - 1) * self.clip
            ret_val = ret_val[0]

        return ret_val

    @property
    def original_object(self) -> Any:
        return self.prob_object

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return None


class SKLearnDTHypothesisClass(HypothesisClass):
    def __init__(self, clip: float, info: DiscreteDataInfo, max_depth: int = 4) -> None:
        super().__init__()
        
        self.clip = clip
        self.info = info
        self.max_depth = max_depth
        
    def generate_hypothesis(self, x: np.ndarray, y: np.ndarray) -> ProbHypothesis:
        dt_orig = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=0.2, criterion='entropy')
        dt_orig.fit(x, y.ravel())
        dt = CalibratedClassifierCV(base_estimator=dt_orig, cv='prefit')
        dt.fit(x, y.ravel())

        return ProbHypothesis(lambda i: dt.predict_proba(i)[:, [1]],
                    self.clip, prob_object=dt, wl_acc=dt.score(x, y.ravel()))


class SKLearnDTUncaliHypothesisClass(HypothesisClass):
    def __init__(self, clip: float, info: DiscreteDataInfo, max_depth: int = 4) -> None:
        super().__init__()
        
        self.clip = clip
        self.info = info
        self.max_depth = max_depth
        
    def generate_hypothesis(self, x: np.ndarray, y: np.ndarray) -> ProbHypothesis:
        dt = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=0.2, criterion='entropy')
        dt.fit(x, y.ravel())

        return ProbHypothesis(lambda i: dt.predict_proba(i)[:, [1]],
                    self.clip, prob_object=dt, wl_acc=dt.score(x, y.ravel()))


class OnlyXSKLearnDTHypothesisClass(HypothesisClass):
    def __init__(self, clip: float, info: DiscreteDataInfo, max_depth: int = 4) -> None:
        super().__init__()
        
        self.clip = clip
        self.info = info
        self.max_depth = max_depth
        
    def generate_hypothesis(self, x: np.ndarray, y: np.ndarray) -> ProbHypothesis:
        dt_orig = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=0.2, criterion='entropy')
        dt_orig.fit(x[:, self.info.feature_columns], y.ravel())
        dt = CalibratedClassifierCV(base_estimator=dt_orig, cv='prefit')
        dt.fit(x[:, self.info.feature_columns], y.ravel())

        return ProbHypothesis(lambda i: dt.predict_proba(i[:, self.info.feature_columns])[:, [1]],
                    self.clip, prob_object=dt, wl_acc=dt.score(x, y.ravel()))


class OnlyXSKLearnDTUncaliHypothesisClass(HypothesisClass):
    def __init__(self, clip: float, info: DiscreteDataInfo, max_depth: int = 4) -> None:
        super().__init__()
        
        self.clip = clip
        self.info = info
        self.max_depth = max_depth
        
    def generate_hypothesis(self, x: np.ndarray, y: np.ndarray) -> ProbHypothesis:
        dt = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=0.2, criterion='entropy')
        dt.fit(x[:, self.info.feature_columns], y.ravel())

        return ProbHypothesis(lambda i: dt.predict_proba(i[:, self.info.feature_columns])[:, [1]],
                    self.clip, prob_object=dt, wl_acc=dt.score(x, y.ravel()))
