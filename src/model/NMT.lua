--[[
Sequence to Sequence model.
It has an encoder and a decoder. No attention mechanism.
--]]

require 'model.Transducer'
local model_utils = require 'model.model_utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')

function NMT:__init(kwargs)
    local kwargs = kwargs;

    -- over write option
    kwargs.vocabSize = kwargs.srcVocabSize
    self.encoder = nn.Transducer(kwargs)

    -- over write option
    kwargs.vocabSize = kwargs.trgVocabSize
    self.decoder = nn.Transducer(kwargs)

    self.layer = nn.Sequential()
    self.layer:add(nn.View(-1, kwargs.hiddenSize))
    self.layer:add(nn.Linear(kwargs.hiddenSize, kwargs.trgVocabSize))
    self.layer:add(nn.LogSoftMax())

    self.criterion = nn.ClassNLLCriterion()
    self.gradEncoder = torch.Tensor()  -- always zeros

    -- get parameters and gradients for optimization
    self.params, self.gradParams = model_utils.combine_all_parameters(self.encoder, self.decoder, self.layer)
    self.maxNorm = kwargs.maxNorm or 5
    self.buffers = {}
end


function NMT:forward(input, target)
    --[[ Forward pass of NMT
    Parameters:
    - `input` : table of source and target tensor
    - `target` : a tensor of next words
    Return:
    - `logProb` : negative log-likelihood of the minibatch
    --]]

    self:stepEncoder(input[1])
    local logProb = self:stepDecoder(input[2])
    return self.criterion:forward(logProb, target)
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
