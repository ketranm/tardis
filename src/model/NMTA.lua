--[[
Sequence to Sequence model with Attention
This implement a simple attention mechanism described in
Effective Approaches to Attention-based Neural Machine Translation
url: http://www.aclweb.org/anthology/D15-1166
--]]

require 'model.Transducer'
require 'model.Glimpse'
local model_utils = require 'model.model_utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')


function NMT:__init(kwargs)
    -- over write option
    kwargs.vocabSize = kwargs.srcVocabSize
    self.encoder = nn.Transducer(kwargs)

    -- over write option
    kwargs.vocabSize = kwargs.trgVocabSize
    self.decoder = nn.Transducer(kwargs)

    self.glimpse = nn.Glimpse(kwargs.hiddenSize)
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
    self.gradEncoder = torch.Tensor()  -- buffer

    self.params, self.gradParams = model_utils.combine_all_parameters(self.encoder, self.decoder, self.glimpse, self.layer)
    self.maxNorm = kwargs.maxNorm or 5
    self.buffers = {}
end


function NMT:forward(input, target)
    --[[Forward pass
    Args:
        input: a table of {x, y} where x = (batchSize, len_xs) Tensor and
        y = (batchSize, len_ys) Tensor
        target: a tensor of (batchSize, len_ys)
    --]]
    -- encode pass
    local outputEncoder = self.encoder:updateOutput(input[1])
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
    local gradDecoder = gradLayer[2]
    local gradGimpse = self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGimpse[2]) -- accummulate gradient in-place 

    self.decoder:backward(input[2], gradDecoder)
    -- init gradient from decoder
    self.encoder:setGradState(self.decoder:getGradState())
    -- backward to encoder
    self.gradEncoder:resizeAs(outputEncoder):zero()
    self.encoder:backward(input[1], self.gradEncoder)
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

--[[
Translation functions
]]

function flat_to_rc(v, indices, flat_index)
    local row = math.floor((flat_index - 1)/v:size(2)) + 1
    return row, indices[row][(flat_index - 1) % v:size(2) + 1]
end


function NMT:_decodeString(x)
    local ws = {}
    local vocab = self.trgVocab
    for i = 1, x:nElement() do
        local idx = x[i]
        if idx ~= vocab['<s>'] and idx ~= vocab['</s>'] then
            ws[#ws + 1] = self.id2word[idx]
        end
    end
    return table.concat(ws, ' ')
end


function NMT:_encodeString(x)
    -- encode source sentence
    local xs = stringx.split(x)
    local xids = {}
    for i = #xs, 1, -1 do
        local w = xs[i]
        local idx = self.srcVocab[w] or self.srcVocab['<unk>']
        table.insert(xids, idx)
    end
    return torch.Tensor(xids):view(1, -1)
end


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
        x: source sentence, (1, T) Tensor
        beamSize: size of the beam search
    --]]
    local srcVocab, trgVocab = self.srcVocab, self.trgVocab
    local idx_GO = trgVocab['<s>']
    local idx_EOS = trgVocab['</s>']

    local K = beamSize or 10
    local T = maxLength or 50

    x = self:_encodeString(x)
    x = x:expand(K, x:size(2)):typeAs(self.params)

    local outputEncoder = self.encoder:updateOutput(x)
    local prevState = self.encoder:lastState()

    local scores = torch.Tensor():typeAs(x):resize(K, 1):zero()
    local hypothesis = torch.Tensor():typeAs(x):resize(T, K):zero():fill(idx_GO)
    local completeHyps = {}
    for i = 1, T-1 do
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
        local k = 1
        while k <= K do
            local score, index = flat:max(1)
            local prev_k, yi = flat_to_rc(maxScores, indices, index[1])
            -- make it -INF so we will not select it next time 
            flat[index[1]] = -math.huge
            if yi == idx_EOS then
                -- complete hypothesis
                local hypo = self:_decodeString(hypothesis[{{}, prev_k}])
                completeHyps[hypo] = scores[prev_k][1]/i -- normalize by sentence length
            else
                table.insert(nextIndex,  yi)
                table.insert(expand_k, prev_k)
                scores[k] = score[1]
                k = k + 1
            end
        end
        expand_k = torch.Tensor(expand_k):long()
        local nextHypothesis = hypothesis:index(2, expand_k)  -- remember to convert to cuda
        nextHypothesis[i+1]:copy(torch.Tensor(nextIndex))
        hypothesis = nextHypothesis
        -- carry over the state of selected k
        local currState = self.decoder:lastState()
        local nextState = {}
        for _, state in ipairs(currState) do
            local my_state = {}
            for _, s in ipairs(state) do
                table.insert(my_state, s:index(1, expand_k))
            end
            table.insert(nextState, my_state)
        end
        prevState = nextState
    end
    for k = 1, K do
        local hypo = self:_decodeString(hypothesis[{{}, k}])
        completeHyps[hypo] = scores[k][1] / (T-1)
    end
    local n_best = {}
    for hypo in pairs(completeHyps) do n_best[#n_best + 1] = hypo end
    -- sort the result and pick the best one
    table.sort(n_best, function(s1, s2)
        return completeHyps[s1] > completeHyps[s2] or completeHyps[s1] > completeHyps[s2] and s1 > s2
    end)
    return n_best[1]
end


function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end