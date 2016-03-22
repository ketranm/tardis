require 'nn'
local Seq2seq = require 'model.Seq2seq'

local kwargs = {source_vocab_size = 10000,
            target_vocab_size = 100,
            embedding_size = 10, 
            hidden_size = 10,
            num_layers = 4, dropout = 0,
            batch_norm = false}

local model = Seq2seq.create(kwargs)
local batch_size, tx, ty = 5, 7, 11
local x = torch.range(1, batch_size * tx):reshape(batch_size, tx)
local y = torch.range(1, batch_size * ty):reshape(batch_size, ty)
local next_y = y:clone()
next_y:sub(1,-1,1,-2):copy(y:sub(1,-1,2,-1))
next_y:sub(1,-1,-1,-1):fill(100)

local nll = 0
for i = 1, 1000 do
    nll = model:forward({x,y}, next_y:view(-1))
    model:backward({x,y}, next_y:view(-1))
    model:update(1)
    print(i, nll)
end
