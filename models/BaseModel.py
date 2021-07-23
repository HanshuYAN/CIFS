from abc import ABC, abstractclassmethod

class BaseModelDNN(ABC):
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def eval_mode(self) -> None:
        pass


    def train_mode(self) -> None:
        pass
    
    @abstractclassmethod
    def predict(self) -> None:
        pass
    

    def fit(self) -> None:
        pass
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
