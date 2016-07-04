--[[ Transducer: Stacking multiple RNNs
By default, we use a layer for word embeddings in Transducer
--]]
require 'torch'
require 'nn'
require 'core.LSTM'
require 'core.GRU'

local Transducer, parent = torch.class('nn.Transducer', 'nn.Module')


function Transducer:__init(config)
    self.vocabSize = config.vocabSize
    self.embeddingSize = config.embeddingSize
    self.hiddenSize = config.hiddenSize
    self.numLayers = config.numLayers
    self.dropout = config.dropout or 0
    self.rnn = config.rnn or 'lstm'
    -- build transducer
    self.transducer = nn.Sequential()
    self.transducer:add(nn.LookupTable(self.vocabSize, self.embeddingSize))
    self._rnns = {}
    self.output = torch.Tensor()

    for i = 1, self.numLayers do
        local prevSize = self.hiddenSize
        if i == 1 then prevSize = self.embeddingSize end
        local rnn
        if self.rnn == 'lstm' then
            rnn = nn.LSTM(prevSize, self.hiddenSize)
        elseif self.rnn == 'gru' then
            rnn = nn.GRU(prevSize, self.hiddenSize)
        else
            error("only support LSTM or GRU!")
        end
        self.transducer:add(rnn)
        table.insert(self._rnns, rnn)
        if self.dropout > 0 then
            self.transducer:add(nn.Dropout(self.dropout))
        end
    end
end

function Transducer:updateOutput(input)
    return self.transducer:forward(input)
end

function Transducer:backward(input, gradOutput, scale)
    return self.transducer:backward(input, gradOutput, scale)
end

function Transducer:parameters()
    return self.transducer:parameters()
end

function Transducer:lastStates()
    local state = {}
    for _, rnn in ipairs(self._rnns) do
        table.insert(state, rnn:lastStates())
    end
    return state
end

function Transducer:training()
    self.transducer:training()
    parent.training(self)
end

function Transducer:evaluate()
    self.transducer:evaluate()
    parent.evaluate(self)
end

function Transducer:setStates(state)
    for i, s in ipairs(state) do
        self._rnns[i]:setStates(s)
    end
end

function Transducer:getGrad()
    local gradState = {}
    for _, rnn in ipairs(self._rnns) do
        table.insert(gradState, rnn:getGrad())
    end
    return gradState
end

function Transducer:setGrad(gradState)
    for i, grad in ipairs(gradState) do
        self._rnns[i]:setGrad(grad)
    end
end

function Transducer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function Transducer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function Transducer:clearState()
    for _, rnn in ipairs(self._rnns) do
        rnn:clearState()
    end
end
