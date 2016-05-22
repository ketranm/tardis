-- Recurrent Neural Language Model
-- Author: Ke Tran <m.k.tran@uva.nl>

require 'model.Transducer'
require 'util.utils'

local RNLM, parent = torch.class('nn.RLM', 'nn.Module')

function RNLM:__init(config)
    local hiddenSize = config.hiddenSize
    local vocabSize = config.vocabSize

    self.rnn = nn.Transducer(config)
    self.layer = nn.Sequential()
    self.layer:add(nn.View(-1, config.hiddenSize))
    self.layer:add(nn.Linear(config.hiddenSize, config.vocabSize))
    self.layer:add(nn.LogSoftMax())

    self.params, self.gradParams = model_utils.combine_all_parameters(self.rnn, self.layer)
    self.criterion = nn.ClassNLLCriterion()
end

function RNLM:step(x)
    local hid = self.rnn:updateOutput(x)
    self.prevState = self.rnn:lastState()
    return hid
end

function RNLM:forward(input, target)
    local hid = self:step(input)
    local logProb = self.layer:forward(hid)
    local loss = self.criterion:forward(logProb, target:view(-1))
    self.buffer = {hid, logProb}
    return loss
end

function RNLM:backward(input, target)
    self.gradParams:zero()
    local hid, logProb = unpack(self.buffer)
    local gradLoss = self.criterion:backward(self.logProb, target)
    local gradLayer = self.layer:backward(hid, gradLoss)
    self.rnn:backward(input, gradLayer)
end

function RNLM:update(learningRate, maxNorm)
    utils.scale_clip(self.gradParams, maxNorm)
    self.params:add(-learningRate, self.gradParams)
end

function RNLM:training()
    self.rnn:training()
end

function RNLM:evaluate()
    self.rnn:evaluate()
end

function RNLM:clearState()
    self.rnn:clearState()
    self.layer:clearState()
end
