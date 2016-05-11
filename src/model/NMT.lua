--[[
Sequence to Sequence model.
It has an encoder and a decoder. No attention mechanism.
--]]

require 'model.Transducer'
local model_utils = require 'model.model_utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')

function NMT:__init(config)
    local config = config;

    -- over write option
    config.vocabSize = config.srcVocabSize
    self.encoder = nn.Transducer(config)

    -- over write option
    config.vocabSize = config.trgVocabSize
    self.decoder = nn.Transducer(config)

    self.layer = nn.Sequential()
    self.layer:add(nn.View(-1, config.hiddenSize))
    self.layer:add(nn.Linear(config.hiddenSize, config.trgVocabSize))
    self.layer:add(nn.LogSoftMax())

    local weights = torch.ones(config.trgVocabSize)
    weights[config.padidx] = 0
    self.padidx = config.padidx

    self.criterion = nn.ClassNLLCriterion(weights, false)
    self.tot = torch.Tensor()
    self.numSamples = 0

    self.gradEncoder = torch.Tensor()  -- always zeros

    -- get parameters and gradients for optimization
    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.layer)
    self.maxNorm = config.maxNorm or 5
    self.buffers = {}
end


function NMT:forward(input, target)
    --[[ Forward pass of NMT

    Parameters:
    - `input` : table of source and target tensor
    - `target` : a tensor of next words

    Return:
    - `logProb` : negative log-likelihood of the mini-batch
    --]]

    self:stepEncoder(input[1])
    local logProb = self:stepDecoder(input[2])
    self.tot:resizeAs(target)
    self.tot:ne(target, self.padidx)
    self.numSamples = self.tot:sum()
    local nll = self.criterion:forward(logProb, target)
    return nll/ self.numSamples

end


function NMT:backward(input, target)
    -- zero grad manually here
    self.gradParams:zero()

    -- make it more explicit

    local buffers = self.buffers
    local outputEncoder = buffers.outputEncoder
    local outputDecoder = buffers.outputDecoder
    local logProb = buffers.logProb

    -- all good. Ready to backprop

    local gradLoss = self.criterion:backward(logProb, target)
    gradLoss:div(self.numSamples)

    local gradDecoder = self.layer:backward(outputDecoder, gradLoss)

    self.decoder:backward(input[2], gradDecoder)
    -- init gradient from decoder
    self.encoder:setGradState(self.decoder:getGradState())
    -- backward to encoder
    local gradEncoder = self.gradEncoder
    gradEncoder:resizeAs(outputEncoder):zero()
    self.encoder:backward(input[1], gradEncoder)
end

function NMT:update(learningRate)
    local gradNorm = self.gradParams:norm()
    local scale = learningRate
    if gradNorm > self.maxNorm then
        scale = scale*self.maxNorm /gradNorm
    end
    self.params:add(self.gradParams:mul(-scale)) -- do it in-place
end

function NMT:stepEncoder(x)
    --[[ Encode the source sequence
    All the information produced by the encoder is stored in buffers
    Parameters:
    - `x` : source tensor, can be a matrix (batch)
    --]]

    local outputEncoder = self.encoder:updateOutput(x)
    local prevState = self.encoder:lastState()
    self.buffers = {outputEncoder = outputEncoder, prevState = prevState}
end


function NMT:stepDecoder(x)
    --[[ Run the decoder
    If it is called for the first time, the decoder will be initialized
    from the last state of the encoder. Otherwise, it will continue from
    its last state. This is useful for beam search or reinforce training
    Parameters:
    - `x` : target sequence, can be a matrix (batch)
    Return:
    - `logProb` : cross entropy loss of the sequence
    --]]

    -- get out necessary information from the buffers
    local buffers = self.buffers
    local outputEncoder, prevState = buffers.outputEncoder, buffers.prevState

    self.decoder:initState(prevState)
    local outputDecoder = self.decoder:updateOutput(x)
    local logProb = self.layer:forward(outputDecoder)

    -- update buffer, adding information needed for backward pass
    buffers.outputDecoder = outputDecoder
    buffers.prevState = self.decoder:lastState()
    buffers.logProb = logProb

    return logProb
end

function NMT:parameters()
    return self.params
end

function NMT:training()
    self.encoder:training()
    self.decoder:training()
end

function NMT:evaluate()
    self.encoder:evaluate()
    self.decoder:evaluate()
end

function NMT:load(filename)
    local params = torch.load(filename)
    self.params:copy(params)
end


function NMT:save(filename)
    torch.save(filename, self.params)
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end
