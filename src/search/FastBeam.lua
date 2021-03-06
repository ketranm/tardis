-- Beam search
local utils = require 'misc.utils'
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
    --[[ Beam search
    Parameters:
    - `x` : source sentence
    - `maxLength` : maximum length of the translation
    - `ref` : opition, if it is provided, report sentence BLEU score   
    --]]

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


        local score_k, prev_k, idx_k = utils.topk(aliveK, maxScores, indices)
        local _eos = torch.Tensor(#idx_k):typeAs(idx_k):fill(self.idx_EOS)

        local mask = idx_k:eq(_eos) -- if EOS is generated
        if mask:sum() > 0 then
            local completed_k  = prev_k:maskedSelect(mask)
            for i = 1, completed_k:numel() do
                local _k = completed_k[i]
                local cand = utils.decodeString(hypothesis[{{}, _k}], id2word, _ignore)
                completeHyps[cand] = scores[prev_k][1]/i -- normalize by sentence length
            end
        end

        mask = idx_k:ne(_eos) -- expand
        aliveK = mask:sum()
        if aliveK == 0 then break end

        local expand_k = prev_k:maskedSelect(mask)
        scores = score_k:maskedSelect(mask):view(-1, 1)
        local next_idx = idx_k:maskedSelect(mask)

        local nextHypothesis = hypothesis:index(2, expand_k)
        -- note: at this point nextHypothesis may contain aliveK hypotheses
        nextHypothesis[i+1]:copy(next_idx)
        hypothesis = nextHypothesis

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
