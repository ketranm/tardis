require 'core.Transducer'
require 'core.ReverseTensor'

local BiRNN, parent = torch.class('nn.BiRNN', 'nn.Module')

function BiRNN:__init(config)
    local fwrnn = nn.Transducer(config)
    local bwrnn = nn.Sequential()
    bwrnn:add(nn.ReverseTensor(2, false)) -- we don't need gradInput
    bwrnn:add(nn.Transducer(config))
    bwrnn:add(nn.ReverseTensor(2)) -- reverse output

    local concat = nn.ConcatTable()
    concat:add(fwrnn)
    concat:add(bwrnn)
    self.btransducer = nn.Sequential()
    self.btransducer:add(concat)
    self.btransducer:add(nn.JoinTable(3))

    self.fwrnn = fwrnn
    self.bwrnn = bwrnn
end

function BiRNN:updateOutput(input)
    return self.btransducer:forward(input)
end

function BiRNN:backward(input, gradOutput)
    return self.btransducer:backward(input, gradOutput)
end

function BiRNN:parameters()
    return self.btransducer:parameters()
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
