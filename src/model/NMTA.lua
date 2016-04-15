--[[
Sequence to Sequence model with Attention
This implement a simple attention mechanism described in
Effective Approaches to Attention-based Neural Machine Translation
url: http://www.aclweb.org/anthology/D15-1166
--]]

require 'model.Transducer'
require 'model.GlimpseDot'
local model_utils = require 'model.model_utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')


function NMT:__init(kwargs)
    -- over write option
    kwargs.vocabSize = kwargs.srcVocabSize
    self.encoder = nn.Transducer(kwargs)

    -- over write option
    kwargs.vocabSize = kwargs.trgVocabSize
    self.decoder = nn.Transducer(kwargs)

    self.glimpse = nn.GlimpseDot(kwargs.hiddenSize)
    --
    self.layer = nn.Sequential()
    -- joining inputs, can be coded more efficient
    local pt = nn.ParallelTable()
    pt:add(nn.Identity())
    pt:add(nn.Identity())

    self.layer:add(pt)
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 2 * kwargs.hiddenSize))
    self.layer:add(nn.Linear(2 * kwargs.hiddenSize, kwargs.hiddenSize, false))
    self.layer:add(nn.ELU(1, true))
    self.layer:add(nn.Linear(kwargs.hiddenSize, kwargs.trgVocabSize, true))
    self.layer:add(nn.LogSoftMax())

    self.criterion = nn.ClassNLLCriterion()

    self.params, self.gradParams = model_utils.combine_all_parameters(self.encoder, self.decoder, self.glimpse, self.layer)
    self.maxNorm = kwargs.maxNorm or 5

    -- use buffer to store all the information needed for forward/backward
    self.buffers = {}
end


function NMT:forward(input, target)
    -- input: a table of tensors

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
    local context = buffers.context
    local logProb = buffers.logProb

    -- all good. Ready to backprop

    local gradLoss = self.criterion:backward(logProb, target)
    local gradLayer = self.layer:backward({context, outputDecoder}, gradLoss)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse = self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accummulate gradient in-place 

    self.decoder:backward(input[2], gradDecoder)
    -- init gradient from decoder
    self.encoder:setGradState(self.decoder:getGradState())
    -- backward to encoder
    local gradEncoder = gradGlimpse[1]
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


function NMT:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
end

function NMT:save(fileName)
    torch.save(fileName, self.params)
end

-- useful interface for beam search

function NMT:stepEncoder(x)
    local outputEncoder = self.encoder:updateOutput(x)
    local prevState = self.encoder:lastState()
    self.buffers = {outputEncoder = outputEncoder, prevState = prevState}
end

function NMT:stepDecoder(x)
    -- get out necessary information from the buffers
    local buffers = self.buffers
    local outputEncoder, prevState = buffers.outputEncoder, buffers.prevState

    self.decoder:initState(prevState)
    local outputDecoder = self.decoder:updateOutput(x)
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})

    -- update buffer, adding information needed for backward pass
    buffers.outputDecoder = outputDecoder
    buffers.prevState = self.decoder:lastState()
    buffers.context = context
    buffers.logProb = logProb

    return logProb
end

function NMT:indexDecoderState(index)
    --[[
    similar to torch.index function, return a new state of kept index
    this function is particularly useful for generating translation
    --]]
    local currState = self.decoder:lastState()
    local newState = {}
    for _, state in ipairs(currState) do
        local sk = {}
        for _, s in ipairs(state) do
            table.insert(sk, s:index(1, index))
        end
        table.insert(newState, sk)
    end

    -- here, it make sense to update the buffer as well
    local buffers = self.buffers
    buffers.prevState = newState
    buffers.outputEncoder = buffers.outputEncoder:index(1, index)

    return newState
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end
