--[[ Sequence to Sequence model with Attention
This implement a simple attention mechanism described in
Effective Approaches to Attention-based Neural Machine Translation
url: http://www.aclweb.org/anthology/D15-1166
--]]

require 'model.Transducer'
require 'model.GlimpseDot'
local model_utils = require 'model.model_utils'
require 'mixer.RewardFactory'
require 'mixer.RFCriterion'
require 'mixer.ClassNLLCriterionWeighted'
require 'optim'

local utils = require 'util.utils'

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
    self.prob = torch.Tensor()
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

    self:stepEncoder(input[1])
    local logProb = self:stepDecoder(input[2])
    self.tot:resizeAs(target)
    self.tot:ne(target, self.pad_idx)
    self.numSamples = self.tot:sum()
    local nll = self.criterion:forward(logProb, target)
    return nll / self.numSamples

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

    -- all good. Ready to back-prop

    local gradLoss = self.criterion:backward(logProb, target)
    -- self.sum already count in the forward pass
    gradLoss:div(self.numSamples)
    local gradLayer = self.layer:backward({context, outputDecoder}, gradLoss)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accummulate gradient in-place 

    self.decoder:backward(input[2], gradDecoder)
    -- initialize gradient from decoder
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
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})

    -- update buffer, adding information needed for backward pass
    buffers.outputDecoder = outputDecoder
    buffers.prevState = self.decoder:lastState()
    buffers.context = context
    buffers.logProb = logProb

    return logProb
end

-- REINFORCE training Neural Machine Translation
function NMT:sample(nsteps)
    --[[ Sample `nsteps` from the model
    Assume that we already run the encoder and started reading in <s> symbol,
    so the buffers must contain log probability of the next words

    Parameters:
    - `nsteps` : integer, number of time steps

    Returns:
    - `output` : 2D tensor of sampled words
    --]]
    local buffers = self.buffers
    local outputEncoder, prevState = buffers.outputDecoder, buffers.prevState
    self.decoder:initState(prevState)

    local logProb = buffers.logProb  -- from the previous prediction
    assert(logProb ~= nil)
    local batchSize = outputEncoder:size(1)
    self.output:resize(batchSize, nsteps)

    for i = 1, nsteps do
        self.prob:resizeAs(logProb)
        self.prob:copy(logProb)
        self.prob:exp()
        self.prob.multinomial(self.output[{{}, {i}}], self.prob, 1)
        logProb = self:stepDecoder(self.output[{{},{i}}])
    end
    return self.output
end

function NMT:initMixer(config)
    print('init mixer')

    self.eos_idx = config.eos_idx
    self.pad_idx = config.pad_idx
    self.unk_idx = config.unk_idx

    -- setup xent criterion
    self.wxent = 0.5  -- hard-coded for now
    local weights = torch.ones(config.trgVocabSize)
    weights[config.pad_idx] = 0
    self.criterion_xe = nn.ClassNLLCriterionWeighted(
        self.wxent, weights, false)
    -- set up reinforce criterion
    local reward_func =
        RewardFactory(config.trgVocabSize,
                      self.eos_idx,
                      self.unk_idx,
                      self.pad_idx)
    self.criterion_rf = RFCriterion(reward_func,
                        self.eos_idx, self.pad_idx, 1)
    self.criterion_rf:setWeight(1 - self.wxent)
    self.criterion_rf:training_mode()

    -- should we setup the cumulative reward predictor here?
    -- lets do it first, think later
    -- this is different from the original MIXER code
    -- we create only one instance of the regressor
    -- and we use it though different time steps
    -- I think it does not make much different
    local crp = nn.Linear(config.hiddenSize, 1)
    crp.bias:fill(0.01)
    crp.weight:fill(0)
    self.crp = crp
    self.param_crp, self.grad_param_crp = crp:getParameters()
    -- TODO: we can train crp using Adadelta,
    -- it is more stable with adaptive learning rate method
    -- buffers
    self.drf = torch.Tensor()
    self.vocab_size = config.trgVocabSize
end

function NMT:overwrite_prediction(pred)
    -- when previous token was eos or pad
    -- pad is produced deterministically
    local curr = pred:clone()
    -- TODO
end

function NMT:trainMixer(input, target, skips, learningRate)
    --[[ train MIXER on one minibatch
    Parameters:
    - `input` : table of (src, target_history)
    - `target` : next_target
    - `skips` : integer, length of reference policy, ie. xent loss

    --]]
    -- probably we should not put this code here
    -- TODO: avoid passing data twice
    local src, trg_inpt = unpack(input)
    local length = trg_inpt:size(2)
    local mbsz = src:size(1)
    local length_xe = skips - 1
    local length_rf = length - skips + 1

    -- be careful with the indices
    self:stepEncoder(input[1])
    if length_xe > 0 then
        local x = trg_inpt:narrow(2, 1, length_xe) --:contiguous()
        self:stepDecoder(x)
    end
    -- now we start with one step
    local x = trg_inpt[{{}, {length_xe + 1}}] --:contiguous()
    self:stepDecoder(x)

    --[[ runing example
    skips = 4, length = 6
    length_rf = 6 - 4 + 1 = 3
    inp : - - - x  s1 s2
    out : - - - s1 s2 s3
    --]]

    local sampled_w = self:sample(length_rf)
    -- TODO: if eos is found, should overwrite the next word to pad

    x = trg_inpt:clone() -- do we need to clone?
    x[{{}, {length_xe + 2, -1}}] = sampled_w[{{}, {1, -2}}]

    local y = target:clone()
    y[{{}, {length_xe + 1, -1}}] =  sampled_w

    -- roll in the RNNs
    self.buffers.prevState = self.encoder:lastState()
    local logProb = self:stepDecoder(x)

    -- compute reinforce loss and gradient
    self.criterion_rf:setSkips(skips)

    -- using a baseline to reduce variance
    local state = self:selectState()
    local pred_crw = self.crp:forward(state)
    local baseline = pred_crw:viewAs(x)

    local reward = self.criterion_rf:forward({y, baseline}, target)    
    local grad_rf  = self.criterion_rf:backward({y, baseline}, target)
    local grad_crp = grad_rf[2]
    -- error of the baseline
    local crp_err = grad_crp:norm()

    self.grad_param_crp:zero()
    self.crp:backward(state, grad_crp:view(-1, 1))
    utils.scale_clip(self.grad_param_crp, 5)
    local lr = learningRate * 0.001
    self.param_crp:add(-lr, self.grad_param_crp)

    -- Overwrite target as we do not need it anymore
    -- use padding to ignore sampled words
    target[{{}, {length_xe + 1, -1}}]:fill(self.pad_idx) 
    target = target:view(-1)
    self.tot:ne(target, self.pad_idx)
    self.numSamples = self.tot:sum()

    local nll = self.criterion_xe:forward(logProb, target)
    nll = nll / self.numSamples

    -- Compute the gradient of MIXER
    -- (1) take gradient of XENT
    local gradLoss = self.criterion_xe:backward(logProb, target)
    -- (2) normalize it
    gradLoss:div(self.numSamples)
    -- (3) take gradient of REINFORCE
    self.drf:resize(mbsz, length, self.vocab_size):zero()
    self.drf:scatter(3, y:view(mbsz, -1, 1):long(), grad_rf[1]:view(mbsz, -1, 1))
    -- (4) add it to gradient of XENT
    gradLoss:add(self.drf:view(-1, self.vocab_size))


    --------------   ok! ready to bprop  ------------------
    self.gradParams:zero()
    local buffers = self.buffers
    local outputEncoder = buffers.outputEncoder
    local outputDecoder = buffers.outputDecoder
    local context = buffers.context
    local logProb = buffers.logProb

    local gradLayer = self.layer:backward({context, outputDecoder}, gradLoss)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradGlimpse =
        self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accummulate gradient in-place 

    self.decoder:backward(input[2], gradDecoder)
    -- initialize gradient from decoder
    self.encoder:setGradState(self.decoder:getGradState())
    -- backward to encoder
    local gradEncoder = gradGlimpse[1]
    self.encoder:backward(input[1], gradEncoder)
    self:update(learningRate)

    -- reward is negative (we are minimizing)
    return {nll, -reward, crp_err}
end

function NMT:indexDecoderState(index)
    --[[ This method is useful for beam search.
    It is similar to torch.index function, return a new state of kept index

    Parameters:
    - `index` : torch.LongTensor object

    Return:
    - `state` : new hidden state of the decoder, indexed by the argument
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
