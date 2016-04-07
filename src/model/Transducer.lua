--[[
Transducer: Stacking multiple RNNs
--]]
require 'torch'
require 'nn'
require 'model.LSTM'
require 'model.GRU'

local Transducer, parent = torch.class('nn.Transducer', 'nn.Module')


function Transducer:__init(kwargs)
    self.vocabSize = kwargs.vocabSize
    self.embeddingSize = kwargs.embeddingSize
    self.hiddenSize = kwargs.hiddenSize
    self.numLayer = kwargs.numLayer
    self.dropout = kwargs.dropout or 0
    self.batchNorm = kwargs.batchNorm
    self.rnn = kwargs.rnn or 'lstm'
    -- build transducer
    self.transducer = nn.Sequential()
    self.transducer:add(nn.LookupTable(self.vocabSize, self.embeddingSize))
    self._rnns = {}
    -- for batch normalization
    self.bn_view_in = {}
    self.bn_view_out = {}
    self.output = torch.Tensor()

    for i = 1, self.numLayer do
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
        if self.batchNorm then
            local view_in = nn.View(1, 1, -1):setNumInputDims(3)
            table.insert(self.bn_view_in, view_in)
            self.transducer:add(view_in)

            local view_out = nn.View(1, -1):setNumInputDims(2)
            table.insert(self.bn_view_out, view_out)
            self.transducer:add(view_out)
        end
        if self.dropout > 0 then
            self.transducer:add(nn.Dropout(self.dropout))
        end
    end
end


function Transducer:updateOutput(input)
    local batchSize, length = input:size(1), input:size(2)

    for _,view_in in ipairs(self.bn_view_in) do
        view_in:resetSize(batchSize * length, -1)
    end
    for _,view_out in ipairs(self.bn_view_out) do
        view_out:resetSize(batchSize, length, -1)
    end
    return self.transducer:forward(input)
end


function Transducer:backward(input, gradOutput, scale)
    return self.transducer:backward(input, gradOutput, scale)
end


function Transducer:parameters()
    return self.transducer:parameters()
end


function Transducer:lastState()
    local state = {}
    for _,rnn in ipairs(self._rnns) do
        table.insert(state, rnn:lastState())
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


function Transducer:initState(state)
    for i, s in ipairs(state) do
        self._rnns[i]:initState(s)
    end
end


function Transducer:getGradState()
    local gradState = {}
    for _, rnn in ipairs(self._rnns) do
        table.insert(gradState, rnn:getGradState())
    end
    return gradState
end


function Transducer:setGradState(gradState)
    for i, grad in ipairs(gradState) do
        self._rnns[i]:setGradState(grad)
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
