require 'core.Transducer'
require 'core.ReverseTensor'

local BiRNN, parent = torch.class('nn.BiRNN', 'nn.Module')

function BiRNN:__init(config)
    local vocabSize = config.vocabSize
    local embeddingSize = config.embeddingSize
    local hiddenSize = config.hiddenSize
    self.numLayers = config.numLayers
    self.dropout = config.dropout or 0
    local rnn_type = config.rnn or 'lstm'

    local RNN = nil
    if rnn_type == 'lstm' then
        RNN = nn.LSTM
    elseif rnn_type == 'gru' then
        RNN = nn.GRU
    else
        error('only support lstm or gru!')
    end

    -- build forward rnn
    local fwrnn = nn.Sequential()
    self._fwrnns = {}
    self._bwrnns = {}
    local bwrnn = nn.Sequential()
    bwrnn:add(nn.ReverseTensor(2, true)) -- reverse the input
    local prevSize = nil
    local lrnn = nil
    for i = 1, self.numLayers do
        if i == 1 then
            prevSize = embeddingSize
        else
            prevSize = hiddenSize
        end
        -- forward rnn
        lrnn = RNN(prevSize, hiddenSize)
        fwrnn:add(lrnn)
        table.insert(self._fwrnns, lrnn)
        -- backward rnn
        lrnn = RNN(prevSize, hiddenSize)
        bwrnn:add(lrnn)
        table.insert(self._bwrnns, lrnn)
        if self.dropout > 0 then
            fwrnn:add(nn.Dropout(self.dropout))
            bwrnn:add(nn.Dropout(self.dropout))
        end
    end

    self.view = nn.View(-1)
    local concat = nn.ConcatTable()
    concat:add(fwrnn)
    concat:add(bwrnn)
    -- put everything together
    local brnn = nn.Sequential()
    brnn:add(nn.LookupTable(vocabSize, embeddingSize))
    brnn:add(concat)
    brnn:add(nn.JoinTable(3)) -- concatenate the top layer of fw and bw rnns
    -- also do a linear transformation
    brnn:add(nn.View(-1, 2 * hiddenSize))
    brnn:add(nn.Linear(2 * hiddenSize, hiddenSize, false))
    brnn:add(self.view)

    self.brnn = brnn
end

function BiRNN:updateOutput(input)
    local N, T = input:size(1), input:size(2)
    self.view:resetSize(N, T, -1)
    return self.brnn:forward(input)
end

function BiRNN:backward(input, gradOutput)
    return self.brnn:backward(input, gradOutput)
end

function BiRNN:parameters()
    return self.brnn:parameters()
end

function BiRNN:lastStates()
    -- this is because we already reverse the sentence
    local state = {}
    for _, rnn in ipairs(self._fwrnns) do
        table.insert(state, rnn:lastStates())
    end
    return state
end


function BiRNN:getGrad(grad)
    local gradState = {}
    for _, rnn in ipairs(self._fwrnns) do
        table.insert(gradState, rnn:getGrad())
    end
    return gradState
end

function BiRNN:setGrad(gradState)
    for i, grad in ipairs(gradState) do
        self._fwrnns[i]:setGrad(grad)
    end
end

function BiRNN:training()
    self.brnn:training()
    parent.training(self)
end

function BiRNN:evaluate()
    self.brnn:evaluate()
    parent.evaluate(self)
end

function BiRNN:clearState()
    for i = 1, self.numLayers do
        self._fwrnns[i]:clearState()
        self._bwrnns[i]:clearState()
    end
end
