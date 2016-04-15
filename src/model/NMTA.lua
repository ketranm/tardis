--[[
Sequence to Sequence model with Attention
This implement a simple attention mechanism described in
Effective Approaches to Attention-based Neural Machine Translation
url: http://www.aclweb.org/anthology/D15-1166
--]]

require 'model.Transducer'
require 'model.GlimpseDot'
local model_utils = require 'model.model_utils'
local utils = require 'util.utils'

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
    -- TODO: set sizeAverage = false
    -- this is because the batch size varies during training
    self.criterion.sizeAverage = false

    self.params, self.gradParams = model_utils.combine_all_parameters(self.encoder, self.decoder, self.glimpse, self.layer)
    self.maxNorm = kwargs.maxNorm or 5
    self.buffers = {}
end


function NMT:forward(input, target)
    --[[Forward pass
    Args:
        input: a table of {x, y} where x = (batchSize, srcLength) Tensor and
        y = (batchSize, trgLength) Tensor
        target: a tensor of (batchSize, trgLength)
    --]]
    -- encode pass
    local outputEncoder = self.encoder:updateOutput(input[1])
    -- we then initialize the decoder with the last state of the encoder
    self.decoder:initState(self.encoder:lastState())
    local outputDecoder = self.decoder:updateOutput(input[2])
    -- compute the context vector
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})

    -- store in buffers
    self.buffers = {outputEncoder, outputDecoder, context, logProb}

    return self.criterion:forward(logProb, target)
end


function NMT:backward(input, target)
    self.gradParams:zero()
    local outputEncoder, outputDecoder, context, logProb = unpack(self.buffers)

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

-- Translation functions ---------------------

function NMT:use_vocab(vocab)
    self.srcVocab = vocab[1]
    self.trgVocab = vocab[2]
    self.id2word = {}
    for w, id in pairs(self.trgVocab) do
        self.id2word[id] = w
    end
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
    self.buffers = {outputEncoder, prevState}
end

function NMT:stepDecoder(x)
    -- run the decode one step
    local outputEncoder, prevState = unpack(self.buffers)
    self.decoder:initState(prevState)
    local outputDecoder = self.decoder:updateOutput(x)
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})
    -- update prevState
    self.buffers[2] = self.decoder:lastState()
    return logProb
end

function NMT:indexDecoderState(index)
    -- similar to torch.index function
    -- return a new state of kept index
    local currState = self.decoder:lastState()
    local newState = {}
    for _, state in ipairs(currState) do
        local sk = {}
        for _, s in ipairs(state) do
            table.insert(sk, s:index(1, index))
        end
        table.insert(newState, sk)
    end

    return newState
end

function NMT:translate(x, beamSize, numTopWords, maxLength)
    --[[Translate input sentence with beam search
    Args:
        x: source sentence
        beamSize: size of the beam search
        we use self.buffers to keep track of outputEncoder and prevState
    --]]

    local srcVocab, trgVocab = self.srcVocab, self.trgVocab
    local id2word = self.id2word
    local idx_GO = trgVocab['<s>']
    local idx_EOS = trgVocab['</s>']

    -- use this to generate clean sentences from ids
    local _ignore = {[idx_GO] = true, [idx_EOS] = true}

    -- number of hypotheses kept in the beam
    local K = beamSize or 10
    -- number of top words selected from the vocabulary distribution output by the model
    local Nw = numTopWords or K

    x = utils.encodeString(x, srcVocab, true)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)

    x = x:expand(K, srcLength):typeAs(self.params)

    self:stepEncoder(x)

    local scores = torch.Tensor():typeAs(x):resize(K, 1):zero()
    local hypothesis = torch.Tensor():typeAs(x):resize(T, K):zero():fill(idx_GO)
    local completeHyps = {}
    local aliveK = K

    -- avoid using goto
    -- handle the first prediction
    local curIdx = hypothesis[1]:view(-1, 1)
    local logProb = self:stepDecoder(curIdx)
    local maxScores, indices = logProb:topk(K, true)  -- should be K not Nw for the first prediction only

    hypothesis[2] = indices[1]
    scores = maxScores[1]:view(-1, 1)

    for i = 2, T-1 do
        local curIdx = hypothesis[i]:view(-1, 1)
        local logProb = self:stepDecoder(curIdx)
        local maxScores, indices = logProb:topk(Nw, true)

        -- previous scores
        local curScores = scores:repeatTensor(1, Nw)
        -- add them to current ones
        maxScores:add(curScores)
        local flat = maxScores:view(maxScores:size(1) * maxScores:size(2))

        local nextIndex = {}
        local expand_k = {}
        local nextScores = {}

        local nextAliveK = 0
        for k = 1, aliveK do
            local logp, index = flat:max(1)
            local prev_k, yi = utils.flat_to_rc(maxScores, indices, index[1])
            -- make it -INF so we will not select it next time
            flat[index[1]] = -math.huge

            if yi == idx_EOS then
                -- complete hypothesis
                local cand = utils.decodeString(hypothesis[{{}, prev_k}], id2word, _ignore)
                completeHyps[cand] = scores[prev_k][1]/i -- normalize by sentence length
            else
                table.insert(nextIndex,  yi)
                table.insert(expand_k, prev_k)
                table.insert(nextScores, logp[1])
                nextAliveK = nextAliveK + 1
            end
        end

        -- nothing left in the beam to expand
        aliveK = nextAliveK
        if aliveK == 0 then break end


        expand_k = torch.Tensor(expand_k):long()
        local nextHypothesis = hypothesis:index(2, expand_k)  -- remember to convert to cuda
        -- note: at this point nextHypothesis may contain aliveK hypotheses
        nextHypothesis[i+1]:copy(torch.Tensor(nextIndex))
        hypothesis = nextHypothesis
        scores = torch.Tensor(nextScores):typeAs(x):view(-1, 1)

        outputEncoder = self.buffers[1]:sub(1, aliveK)
        self.buffers[1] = outputEncoder
        local nextState = self:indexDecoderState(expand_k)
        self.buffers[2] = nextState
    end


    for k = 1, aliveK do
        local cand = utils.decodeString(hypothesis[{{}, k}], id2word, _ignore)
        completeHyps[cand] = scores[k][1] / (T-1)
    end

    local nBest = {}
    for cand in pairs(completeHyps) do nBest[#nBest + 1] = cand end
    -- sort the result and pick the best one
    table.sort(nBest, function(c1, c2)
        return completeHyps[c1] > completeHyps[c2] or completeHyps[c1] > completeHyps[c2] and c1 > c2
    end)

    -- prepare n-best list for printing
    local nBestList = {}
    for rank, hypo in ipairs(nBest) do
        -- stringx.count is fast
        local length = stringx.count(hypo, ' ') + 1
        table.insert(nBestList, string.format('n=%d s=%.4f l=%d\t%s',  rank, completeHyps[hypo], length, hypo))
    end

    return nBest[1], nBestList
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end
