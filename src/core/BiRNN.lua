require 'core.Transducer'
require 'core.ReverseTensor'

local BiRNN, parent = torch.class('nn.BiRNN', 'nn.Module')

function BiRNN:__init(config)
    local fwrnn = nn.Transducer(config)
    local bwrnn = nn.Sequential()
    bwrnn:add(nn.ReverseTensor(2, false)) -- we don't need gradInput
    bwrnn:add(nn.Transducer(config))
    bwrnn:add(nn.ReverseTensor(2, true)) -- reverse output

    self.view = nn.View(-1)

    local concat = nn.ConcatTable()
    concat:add(fwrnn)
    concat:add(bwrnn)
    self.btransducer = nn.Sequential()
    self.btransducer:add(concat)
    self.btransducer:add(nn.JoinTable(3))
    self.btransducer:add(nn.View(-1, 2 * config.hiddenSize))
    self.btransducer:add(nn.Linear(2 * config.hiddenSize, config.hiddenSize, false))
    self.btransducer:add(self.view)
    self.fwrnn = fwrnn
    self.bwrnn = bwrnn
end

function BiRNN:updateOutput(input)
    local N, T = input:size(1), input:size(2)
    self.view:resetSize(N, T, -1)
    return self.btransducer:forward(input)
end

function BiRNN:backward(input, gradOutput)
    return self.btransducer:backward(input, gradOutput)
end

function BiRNN:parameters()
    return self.btransducer:parameters()
end

function BiRNN:lastStates()
    -- this is because we already reverse the sentence
    return self.fwrnn:lastStates()
end

function BiRNN:setGrad(grad)
    self.fwrnn:setGrad(grad)
end

function BiRNN:training()
    self.fwrnn:training()
    self.bwrnn:training()
end

function BiRNN:evaluate()
    self.fwrnn:evaluate()
    self.bwrnn:evaluate()
end

function BiRNN:clearState()
    self.fwrnn:clearState()
    self.bwrnn:clearState()
end
