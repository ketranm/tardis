# TARDIS

Tardis stands for Time And Relative Dimension In Space.
Yes, it is the [TARDIS](https://en.wikipedia.org/wiki/TARDIS) from Dr. Who.

Did I mention the TARDIS looks like a BLEU telephone box. Oops, I meant blue telephone box.

We have switched to use [cudnn](https://github.com/soumith/cudnn.torch), updated code will be made available soon.
## Features

- Standard Seq2seq model (we won't update this for now)
- Seq2seq with attention
- Fast LSTM/GRU
- Bilinear/Dot attention mechanism
- Hard Attention (using REINFORCE update rule)

## Code Organization
The code is organized as follow

- `src/core` Contains core modules, such as fast LSTM, GRU, MemoryBlock, ... These are building units of more complicated models,
- `src/seq2seq` Neural machine translation modules,
- `src/data` Data processing scripts, to create/load bitext or monolingual data,
- `src/misc` some ultility functions
- `src/tardis` search engine, probably should rename to `search`
- `src/test` unit tests

## Planing
- Recurrent Memory Networks (in progress)
- Character LSMT models
- REINFORCE training NMT (in progress)
- Efficient padding/bucketting sentences to speed up training

## Contributors

- Ke Tran
- Arianna Bisazza

## Acknowledgments
Our implementation utilizes code from the following:
- https://github.com/jcjohnson/torch-rnn
- https://github.com/wojzaremba/lstm
