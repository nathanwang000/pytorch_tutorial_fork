#!/bin/bash
python main.py -o 'torch.optim.Adam'
python main.py -o 'torch.optim.SGD'
python main.py -o 'optimizers.Diff'
python main.py -o 'optimizers.DiffUnbiased'
python main.py -o 'optimizers.DiffUnbiasedBounded'
python main.py -o 'torch.optim.Adadelta'
