require 'torch'
require 'nn'
require 'seq2seq.NMTA'

local config = {srcVocabSize = 100,
            trgVocabSize = 100,
            embeddingSize = 10, 
            hiddenSize = 10, pad_idx = 1,
            numLayers = 4, dropout = 0}

local model = nn.NMT(config)
local batchSize, tx, ty = 5, 7, 11
local x = torch.range(1, batchSize * tx):reshape(batchSize, tx)
local y = torch.range(1, batchSize * ty):reshape(batchSize, ty)
local next_y = y:clone()
next_y:sub(1,-1,1,-2):copy(y:sub(1,-1,2,-1))
next_y:sub(1,-1,-1,-1):fill(100)

local nll = 0
for i = 1, 100 do
    nll = model:forward({x,y}, next_y)
    model:backward({x,y}, next_y)
    model:update(1)
    print(i, nll)
end
