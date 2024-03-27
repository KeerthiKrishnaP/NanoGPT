import torch
from torch import Tensor

def weighted_average(B,T,C, string) ->Tensor:
    bowl_of_words = torch.zeros()
    for b in range(B):
        for t in range(T):
            previous_words = string[b,:t+1]
            bowl_of_words = torch.mean(previous_words, 0)
    
    return bowl_of_words