--[[
Sequence to Sequence model with Attention
This implement a simple attention mechanism described in
Effective Approaches to Attention-based Neural Machine Translation
url: http://www.aclweb.org/anthology/D15-1166


TODO: reinforce
--]]

require 'model.Transducer'
require 'model.GlimpseDot'
local bleu = require 'util.BLEU'
local model_utils = require 'model.model_utils'
local _ = require 'moses'
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

    self.criterion = nn.ClassNLLCriterion(torch.ones(kwargs.trgVocabSize))
    self.params, self.gradParams = model_utils.combine_all_parameters(self.encoder, self.decoder, self.glimpse, self.layer)
    self.maxNorm = kwargs.maxNorm or 5
    self.buffers = {}

    -- not cool but it's aright
    self.regressor = nn.Sequential()
    self.regressor:add(nn.Linear(kwargs.hiddenSize, 1))
    -- we need a number between 0 and 1
    self.regressor:add(nn.Sigmoid())

    self.mse = nn.MSECriterion()
    self.reward = torch.Tensor(kwargs.trgVocabSize)
    self.buffer_r = torch.Tensor(#self.reward)
    self.buffer_b = torch.Tensor()
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


-- Translation functions

function NMT:use_vocab(vocab)
    self.srcVocab = vocab[1]
    self.trgVocab = vocab[2]
    self.id2word = {}
    for w, id in pairs(self.trgVocab) do
        self.id2word[id] = w
    end
end


function NMT:load(filename)
    local params = torch.load(filename)
    self.params:copy(params)
end


function NMT:save(filename)
    torch.save(filename, self.params)
end


function NMT:translate(x, beamSize, maxLength)
    --[[Translate input sentence with beam search
    Args:
        x: source sentence
        beamSize: size of the beam search
    --]]

    local srcVocab, trgVocab = self.srcVocab, self.trgVocab
    local id2word = self.id2word
    local idx_GO = trgVocab['<s>']
    local idx_EOS = trgVocab['</s>']

    -- use this to generate clean sentences from ids
    local _ignore = {[idx_GO] = true, [idx_EOS] = true}

    local K = beamSize or 10

    x = utils.encodeString(x, srcVocab, true)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)

    x = x:expand(K, srcLength):typeAs(self.params)

    -- keep output of encoder for future use
    -- for example, reinforce
    local _outputEncoder = self.encoder:updateOutput(x)
    -- maybe put it to buffer
    self.buffers = {_outputEncoder}
    
    local outputEncoder = _outputEncoder

    local prevState = self.encoder:lastState()

    local scores = torch.Tensor():typeAs(x):resize(K, 1):zero()
    local hypothesis = torch.Tensor():typeAs(x):resize(T, K):zero():fill(idx_GO)
    local completeHyps = {}

    -- avoid using goto
    -- handle the first prediction
    self.decoder:initState(prevState)
    local curIdx = hypothesis[1]:view(-1, 1)
    local outputDecoder = self.decoder:forward(curIdx)
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})
    local maxScores, indices = logProb:topk(K, true)
    hypothesis[2] = indices[1]
    scores = maxScores[1]:view(-1, 1)

    prevState = self.decoder:lastState()

    for i = 2, T-1 do
        self.decoder:initState(prevState)
        local curIdx = hypothesis[i]:view(-1, 1)
        local outputDecoder = self.decoder:forward(curIdx)
        local context = self.glimpse:forward({outputEncoder, outputDecoder})        
        local logProb = self.layer:forward({context, outputDecoder})

        local maxScores, indices = logProb:topk(K, true)
        -- previous scores
        local curScores = scores:repeatTensor(1, K)
        -- add them to current ones
        maxScores:add(curScores)

        local flat = maxScores:view(maxScores:size(1) * maxScores:size(2))
        local nextIndex = {}
        local expand_k = {}
        local nextScores = {}
        local k = 1
        while k <= K do
            local logp, index = flat:max(1)
            local prev_k, yi = utils.flat_to_rc(maxScores, indices, index[1])

            -- make it -INF so we will not select it next time 
            flat[index[1]] = -math.huge

            if yi == idx_EOS then
                -- complete hypothesis
                local cand = utils.decodeString(hypothesis[{{}, prev_k}], id2word, _ignore)
                completeHyps[cand] = scores[prev_k][1]/i -- normalize by sentence length
                -- reduce beam size
                K = K - 1
            else
                table.insert(nextIndex,  yi)
                table.insert(expand_k, prev_k)
                table.insert(nextScores, logp[1])
                k = k + 1
            end
        end

        if K < 1 then
            break
        end

        expand_k = torch.Tensor(expand_k):long()
        local nextHypothesis = hypothesis:index(2, expand_k)
        nextHypothesis[i+1]:copy(torch.Tensor(nextIndex))
        hypothesis = nextHypothesis
        scores = torch.Tensor(nextScores):typeAs(x):resize(K,1)
        
        -- need to resize outputEncoder
        outputEncoder = _outputEncoder:sub(1, K)
        -- carry over the state of selected k
        local currState = self.decoder:lastState()
        local nextState = {}
        for _, state in ipairs(currState) do
            local sk = {}
            for _, s in ipairs(state) do
                table.insert(sk, s:index(1, expand_k))
            end
            table.insert(nextState, sk)
        end
        prevState = nextState
    end

    -- force the produce some output
    if next(completeHyps) == nil then
        for k = 1, K do
            local cand = utils.decodeString(hypothesis[{{}, k}], id2word, _ignore)
            completeHyps[cand] = scores[k][1] / (T-1)
        end
    end

    local nBest = {}
    for cand in pairs(completeHyps) do nBest[#nBest + 1] = cand end
    -- sort the result and pick the best one
    table.sort(nBest, function(c1, c2)
        return completeHyps[c1] > completeHyps[c2] or completeHyps[c1] > completeHyps[c2] and c1 > c2
    end)

    -- prepare n-best list for printing
    local nBestList = {}
    for rank, hypo in pairs(nBest) do
        local score = string.format('%.4f', completeHyps[hypo])
        local length = 0
        for w in hypo:gmatch("%S+") do length = length + 1 end
        table.insert(nBestList, 'n=' .. rank .. ' s=' .. score .. ' l=' .. length .. '\t' .. hypo)
    end
  
    return nBest[1], nBestList
end


function NMT:_sample(input, rollinStep)
    local outputEncoder = self.encoder:updateOutput(input[1])
    -- we then initialize the decoder with the last state of the encoder
    self.decoder:initState(self.encoder:lastState())
    -- this code is not efficient at the moment
    -- TODO: hackin LSTM code to allow preallocation of time step
    -- i don't have time to write it at the moment
    local refPrefixInput = input[2]:narrow(2, 1, rollinStep)
    local outputDecoder = self.decoder:updateOutput(refPrefixInput)
    local lastOuputDecoder = outputDecoder[{{},{-1},{}}]
    local context = self.glimpse:forward({outputEncoder, lastOuputDecoder})
    local logProb = self.layer:forward({context, lastOuputDecoder})
    local prob = logProb:exp()

    local trg = input[2]:clone()
    -- sample this shit
    local sample = torch.multinomial(prob, 1)
    -- start rolling out
    local rolloutStep = input[2]:size(2) - rollinStep

    
    for i = 1, rolloutStep do
        trg[{{}, rollinStep + i}] = sample
        lastOuputDecoder =  self.decoder:updateOutput(sample)
        context = self.glimpse:forward({outputEncoder, lastOuputDecoder})
        logProb = self.layer:forward({context, lastOuputDecoder})
        prob = logProb:exp()
        sample = torch.multinomial(prob, 1)
    end
    self.buffers = {outputEncoder}
    return trg
end

function NMT:reinforce(input, ref, rollinStep)
    local sample = self:_sample(input, rollinStep)
    local outputEncoder = self.buffers[1]

    self.decoder:initState(self.encoder:lastState())
    local outputDecoder = self.decoder:updateOutput(sample)
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder})

    local nextTarget = sample:clone()
    nextTarget:sub(1,-1,1,-2):copy(sample:sub(1,-1,2,-1))
    nextTarget:sub(1,-1,-1,-1):fill(self.trgVocab["</s>"])
    
    ref = ref:view(#nextTarget)
    local batchSize = ref:size(1)

    local reward = torch.Tensor(#ref):typeAs(self.params)
    for i = 1, ref:size(1) do
        local r = bleu.scoreT(nextTarget[i], ref[i])
        reward[i]:fill(r)
    end

    -- train the regressor
    local hiddenSize = context:size(3)
    local baseline = self.regressor:forward(context:view(-1, hiddenSize))
    self.mse:forward(baseline, reward)
    local grad = self.mse:backward(baseline, reward)
    self.regressor:backward(context:view(-1, hiddenSize), grad)
    self.regressor:updateParameters(0.1)

    -- rescale with the reward
    --self.buffer_r:resizeAs(#logProb)
    -- self.criterion.weight = baseline:view(-1)
    local relReward = reward:view(-1):add(-1, baseline:view(-1))
    -- should average reward of the same word

    self.reward:fill(0):indexCopy(1, nextTarget:view(-1):long(), reward:view(-1))
    -- this trick is used to normalized the reward
    --self.buffer_r:fill(1e-10):indexAdd(1, nextTarget:view(-1):long(), torch.ones(nextTarget:numel()))
    --self.reward:cdiv(self.buffer_r)

    self.criterion.weights = self.reward
    local rl_loss = self.criterion:forward(logProb, nextTarget:view(-1))
    self.reward:fill(0):indexCopy(1, nextTarget:view(-1):long(), relReward)
    self.criterion.weights = self.reward
    local xentGrad = self.criterion:backward(logProb, nextTarget:view(-1))

    local gradLayer = self.layer:backward({context, outputDecoder}, xentGrad)
    local gradGlimpse = self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])
    local gradDecoder = gradLayer[2]
    gradDecoder:add(gradGlimpse[2])

    self.decoder:backward(sample, gradDecoder)
    self.encoder:setGradState(self.decoder:getGradState())
    local gradEncoder = gradGlimpse[1]
    self.encoder:backward(input[1], gradEncoder)
    return rl_loss
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end
