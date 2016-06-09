--[[ Sequence to Sequence model with Attention
This implement a simple attention mechanism described in
Effective Approaches to Attention-based Neural Machine Translation
url: http://www.aclweb.org/anthology/D15-1166
--]]

require 'core.Transducer'
require 'core.GlimpseDot'
local model_utils = require 'core.model_utils'

local utils = require 'misc.utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')


function NMT:__init(config)
    -- over write option
    config.vocabSize = config.srcVocabSize
    self.encoder = nn.Transducer(config)

    -- over write option
    config.vocabSize = config.trgVocabSize
    self.decoder = nn.Transducer(config)

    self.glimpse = nn.GlimpseDot(config.hiddenSize)

    self.layer = nn.Sequential()
    -- joining inputs, can be coded more efficient
    local pt = nn.ParallelTable()
    pt:add(nn.Identity())
    pt:add(nn.Identity())

    self.layer:add(pt)
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 2 * config.hiddenSize))
    self.layer:add(nn.Linear(2 * config.hiddenSize, config.hiddenSize, false))
    self.layer:add(nn.ELU(1, true))
    self.layer:add(nn.Linear(config.hiddenSize, config.trgVocabSize, true))
    self.layer:add(nn.LogSoftMax())

    local weights = torch.ones(config.trgVocabSize)
    weights[config.pad_idx] = 0

    self.sizeAverage = false
    self.pad_idx = config.pad_idx
    self.criterion = nn.ClassNLLCriterion(weights, false)
    self.tot = torch.Tensor() -- count non padding symbols
    self.numSamples = 0

    self.params, self.gradParams =
        model_utils.combine_all_parameters(self.encoder,
                                           self.decoder,
                                           self.glimpse,
                                           self.layer)
    self.maxNorm = config.maxNorm or 5

    -- use buffer to store all the information needed for forward/backward
    self.buffers = {}
    self.output = torch.LongTensor()
end

function NMT:selectState()
    -- get states that might be helpful for predicting reward
    return self.layer:get(5).output
end

function NMT:forward(input, target)
    --[[ Forward pass of NMT

    Parameters:
    - `input` : table of source and target tensor
    - `target` : a tensor of next words

    Return:
    - `logProb` : negative log-likelihood of the mini-batch
    --]]
    local target = target:view(-1)
    self:stepEncoder(input[1])
    local logProb = self:stepDecoder(input[2])
    self.tot:resizeAs(target)
    self.tot:ne(target, self.pad_idx)
    self.numSamples = self.tot:sum()
    local nll = self.criterion:forward(logProb, target)
    return nll / self.numSamples

end

function NMT:backward(input, target, gradOutput)
    -- zero grad manually here
    self.gradParams:zero()

    -- make it more explicit

    local buffers = self.buffers
    local outputEncoder = buffers.outputEncoder
    local outputDecoder = buffers.outputDecoder
    local context = buffers.context
    local logProb = buffers.logProb

    -- all good. Ready to back-prop
    local gradLoss = gradOutput
    -- by default, we use Cross-Entropy loss
    if not gradLoss then
        gradLoss = self.criterion:backward(logProb, target:view(-1))
        local norm_coeff = 1/ (self.sizeAverage and self.numSamples or 1)
        gradLoss:mul(norm_coeff)
    end

    local gradLayer = self.layer:backward({context, outputDecoder}, gradLoss)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accumulate gradient in-place

    self.decoder:backward(input[2], gradDecoder)
    -- initialize gradient from decoder
    self.encoder:setGrad(self.decoder:getGrad())
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
    --[[ Encode the source sequence
    All the information produced by the encoder is stored in buffers
    Parameters:
    - `x` : source tensor, can be a matrix (batch)
    --]]

    local outputEncoder = self.encoder:updateOutput(x)
    local prevState = self.encoder:lastStates()
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

    self.decoder:setStates(prevState)
    local outputDecoder = self.decoder:updateOutput(x)
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})

    -- update buffer, adding information needed for backward pass
    buffers.outputDecoder = outputDecoder
    buffers.prevState = self.decoder:lastStates()
    buffers.context = context
    buffers.logProb = logProb

    return logProb
end

-- REINFORCE training Neural Machine Translation
function NMT:sample(nsteps, k)
    --[[ Sample `nsteps` from the model
    Assume that we already run the encoder and started reading in <s> symbol,
    so the buffers must contain log probability of the next words

    Parameters:
    - `nsteps` : integer, number of time steps
    - `k` : sample from top k
    Returns:
    - `output` : 2D tensor of sampled words
    --]]

    if not self.prob then
        self.prob = torch.Tensor():typeAs(self.params)
    end

    local buffers = self.buffers
    local outputEncoder, prevState = buffers.outputDecoder, buffers.prevState
    self.decoder:setStates(prevState)

    local logProb = buffers.logProb  -- from the previous prediction
    assert(logProb ~= nil)
    local batchSize = outputEncoder:size(1)
    self.output:resize(batchSize, nsteps)

    for i = 1, nsteps do
        self.prob:resizeAs(logProb)
        self.prob:copy(logProb)
        self.prob:exp()
        if k then
            local prob_k, idx = self.prob:topk(k, true)
            prob_k:cdiv(prob_k:sum(2):repeatTensor(1, k)) -- renormalized
            local sample = torch.multinomial(prob_k, 1)
            self.output[{{}, {i}}] = idx:gather(2, sample)
        else
            self.prob.multinomial(self.output[{{}, {i}}], self.prob, 1)
        end
        logProb = self:stepDecoder(self.output[{{},{i}}])
    end

    return self.output
end

function NMT:indexDecoderState(index)
    --[[ This method is useful for beam search.
    It is similar to torch.index function, return a new state of kept index

    Parameters:
    - `index` : torch.LongTensor object

    Return:
    - `state` : new hidden state of the decoder, indexed by the argument
    --]]

    local currState = self.decoder:lastStates()
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
