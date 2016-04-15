-- Beam search
local utils = require 'util.utils'
local BLEU = require 'util.BLEU'
local BeamSearch = torch.class('BeamSearch')


function BeamSearch:__init(kwargs)
    self.srcVocab = kwargs.srcVocab
    self.trgVocab = kwargs.trgVocab

    self.id2word = {}
    for w, id in pairs(self.trgVocab) do
        self.id2word[id] = w
    end

    self.idx_GO = self.trgVocab['<s>']
    self.idx_EOS = self.trgVocab['</s>']
    self._ignore = {[self.idx_GO] = true, [self.idx_EOS] = true}

    self.K = kwargs.beamSize or 10
    self.Nw = kwargs.numTopWords or self.K
end

function BeamSearch:use(model)
    self.model = model
    self.model:evaluate() -- no dropout during testing
    self.dtype = model.params:type()
end

function BeamSearch:search(x, maxLength, ref)
    --[[
    Beam search:
    - x: source sentence
    - maxLength: maximum length of the translation
    - ref: opition, if it is provided, report sentence BLEU score   
    ]]

    -- use this to generate clean sentences from ids
    local _ignore = self._ignore
    local id2word = self.id2word
    local K, Nw = self.K, self.Nw

    x = utils.encodeString(x, self.srcVocab, true)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)

    x = x:expand(K, srcLength):type(self.dtype)

    self.model:stepEncoder(x)

    local scores = torch.zeros(K, 1):type(self.dtype)
    local hypothesis = torch.zeros(T, K):fill(self.idx_GO):type(self.dtype)
    local completeHyps = {}
    local aliveK = K

    -- avoid using goto
    -- handle the first prediction
    local curIdx = hypothesis[1]:view(-1, 1)
    local logProb = self.model:stepDecoder(curIdx)
    local maxScores, indices = logProb:topk(K, true)  -- should be K not Nw for the first prediction only

    hypothesis[2] = indices[1]
    scores = maxScores[1]:view(-1, 1)

    for i = 2, T-1 do
        local curIdx = hypothesis[i]:view(-1, 1)
        local logProb = self.model:stepDecoder(curIdx)
        local maxScores, indices = logProb:topk(Nw, true)

        -- previous scores
        local curScores = scores:repeatTensor(1, Nw)
        -- add them to current ones
        maxScores:add(curScores)
        local flat = maxScores:view(-1)

        local nextIndex = {}
        local expand_k = {}
        local nextScores = {}

        local nextAliveK = 0
        for k = 1, aliveK do
            local logp, index = flat:max(1)
            local prev_k, yi = utils.flat_to_rc(maxScores, indices, index[1])
            -- make it -INF so we will not select it next time
            flat[index[1]] = -math.huge

            if yi == self.idx_EOS then
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

        self.model:indexDecoderState(expand_k)
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
    local msg
    for rank, hypo in ipairs(nBest) do
        -- stringx.count is fast
        local length = stringx.count(hypo, ' ') + 1
        if ref then
            local reward = BLEU.score(hypo, ref) * 100  -- rescale for readability
            msg = string.format('n=%d s=%.4f l=%d b=%.4f\t%s',  rank, completeHyps[hypo], length, reward, hypo)
        else
            msg = string.format('n=%d s=%.4f l=%d\t%s',  rank, completeHyps[hypo], length, hypo)
        end
        table.insert(nBestList, msg)
    end

    return nBest[1], nBestList
end
