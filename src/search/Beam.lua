-- Beam search
-- Ariana wrote major of this code
local utils = require 'misc.utils'
local BeamSearch = torch.class('BeamSearch')


function BeamSearch:__init(config)
    self.srcVocab = config.srcVocab
    self.trgVocab = config.trgVocab

    self.id2word = {}
    for w, id in pairs(self.trgVocab) do
        self.id2word[id] = w
    end

    self.idx_GO = self.trgVocab['<s>']
    self.idx_EOS = self.trgVocab['</s>']
    self._ignore = {[self.idx_GO] = true, [self.idx_EOS] = true}

    self.K = config.beamSize or 10
    self.Nw = config.numTopWords or self.K
    self.pTh = config.pruneThreshold  -- example value: -6 (math.log(0.002))

    self.reverseInput = config.reverseInput or true
end

function BeamSearch:use(model)
    self.model = model
    self.model:evaluate() -- no dropout during testing
    self.dtype = model.params:type()
end

function BeamSearch:printAttention(_att, numTopA, asMatrix)
    -- att is a trgLength x srcLength soft attention matrix
    -- topA is the number of top aligned source indices to print for each target index

    local att = _att
    if self.reverseInput then
        att = utils.reverse(att, 2)
    end

    -- create a matrix of binary values than can be easily visualized
    local top,idTop = att:topk(numTopA,true)
    local topAtt = torch.zeros(#att):typeAs(att):scatter(2, idTop, 1)

    if asMatrix then return topAtt end

    local str = ''
    for i = 1, topAtt:size(1) do
    local nj = 0
        for j = 1, topAtt:size(2) do
            if topAtt[i][j] ~= 0 then
                str = str .. j
                nj = nj + 1
                if nj < numTopA then str = str .. '|' end
            end
        end
        str = str .. '-' .. i .. ' '
    end
    return str
end

function BeamSearch:search(x, maxLength, ref)
    --[[
    Beam search:
    - x: source sentence
    - maxLength: maximum length of the translation
    - ref: opition, if it is provided, report sentence BLEU score
    ]]
    local K, Nw = self.K, self.Nw

    x = utils.encodeString(x, self.srcVocab, self.reverseInput)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)

    x = x:expand(K, srcLength):type(self.dtype)

    self.model:stepEncoder(x)

    local scores = torch.zeros(K, 1):type(self.dtype)
    local hypotheses = torch.zeros(T, K):fill(self.idx_GO):type(self.dtype)
    -- for each alive hypo, store attention distribution a each time step:
    local attentions = torch.zeros(K, T, srcLength):type(self.dtype)
    -- for each alive hypo, store "source coverage" vector:
    local completeHyps = {}
    local completeHypsAtt = {}
    local numCompleteHyps = 0

    -- avoid using goto
    -- handle the first prediction
    local curIdx = hypotheses[1]:view(-1, 1)
    local logProb = self.model:stepDecoder(curIdx)
    local maxScores, indices = logProb:topk(K, true)  -- should be K not Nw for the first prediction only
    local att = self.model.glimpse:getAttention()
    local _, topAttIds = att:squeeze():max(2)

    hypotheses[2] = indices[1]
    attentions[{ {},1 }] = att
    scores = maxScores[1]:view(-1, 1)

    for i = 2, T-1 do
        local curIdx = hypotheses[i]:view(-1, 1)
        local logProb = self.model:stepDecoder(curIdx)
        local att = self.model.glimpse:getAttention()
        attentions[{ {},i }] = att
        local _, topAttIds = att:squeeze(2):max(2)

        -- local pruning (i.e. keep only Nw best continuations of same prefix) happens here:
        local maxScores, indices = logProb:topk(Nw, true)
        -- apply threshold pruning to each word set
        if self.pTh then
            local thresholds = maxScores:max(2) + self.pTh
            maxScores[maxScores:lt(thresholds:expand(#maxScores))] = -math.huge
        end

        -- previous scores
        local curScores = scores:repeatTensor(1, Nw)
        -- add them to current ones
        maxScores:add(curScores)

        local nextIndex = {}
        local expand_k = {}
        local nextScores = {}
        local completeHypoIds = {}

        self:pruneHypos(maxScores, indices, K, nextIndex, expand_k, nextScores, completeHypoIds)

        for j = 1, #completeHypoIds do
            local id = completeHypoIds[j]
            local cand = utils.decodeString(hypotheses[{{}, id}], self.id2word, self._ignore)
            completeHyps[cand] = scores[id][1] / i -- normalize by sentence length
            completeHypsAtt[cand] = attentions[id]:sub(1,(i-1)):clone()
            numCompleteHyps = numCompleteHyps + 1
            -- for each EOS that is selected, reduce beam-width by one [Cho & al.2014]
            K = K - 1
        end

        -- nothing left in the beam to expand
        if #nextIndex < 1 or K < 1 then break end

        expand_k = torch.Tensor(expand_k):long()
        local nextHypotheses = hypotheses:index(2, expand_k)  -- remember to convert to CUDA
        -- note: at this point nextHypotheses may contain K hypotheses
        nextHypotheses[i+1]:copy(torch.Tensor(nextIndex))
        hypotheses = nextHypotheses

        attentions = attentions:index(1, expand_k)
        scores = torch.Tensor(nextScores):typeAs(x):view(-1, 1)
        self.model:indexDecoderState(expand_k)
    end

    local numLastHypos = math.min(K, hypotheses:size(2))
    -- add alive but uncompleted hypotheses to the last beam
    if numLastHypos > 0 then
        local _, bestLastHypoIds = scores:topk(numLastHypos, 1, true)
        for l = 1, numLastHypos do
            local id = bestLastHypoIds[l][1]
            local cand = utils.decodeString(hypotheses[{{}, id}], self.id2word, self._ignore)
            completeHyps[cand] = scores[id][1] / (T-1)
            completeHypsAtt[cand] = attentions[id]:sub(1,(T-2)):clone()
            numCompleteHyps = numCompleteHyps + 1
        end
    end

    assert(numCompleteHyps > 0, "No hypothesis was completed")

    local nBest = {}
    for cand in pairs(completeHyps) do nBest[#nBest + 1] = cand end
    -- sort the result and pick the best one
    table.sort(nBest, function(c1, c2)
        return completeHyps[c1] > completeHyps[c2] or completeHyps[c1] > completeHyps[c2] and c1 > c2
    end)

    return nBest[1], self:prepareNbestList(nBest, completeHyps, completeHypsAtt, ref)
end


function BeamSearch:pruneHypos(expandScores, expandIndices, K, nextIndex, expand_k, nextScores, completeHypoIds)
    local flat = expandScores:view(-1)
    if K > flat:size(1) then K = flat:size(1) end
    -- beam pruning (i.e. keep only aliveK hypotheses of length i based on global score):
    local maxScoresB, indicesB = flat:topk(K, true)
    local thresholdB
    if self.pTh then thresholdB = flat:max() + self.pTh end

    for k = 1, K do
        -- apply threshold pruning to the set of length-i hypotheses
        if self.pTh and maxScoresB[k] < thresholdB then goto continue end

        local prev_k, yi = utils.flat_to_rc(expandScores, expandIndices, indicesB[k])

        if yi == self.idx_EOS then
            -- complete hypothesis
            table.insert(completeHypoIds,prev_k)
        else
            table.insert(nextIndex,  yi)
            table.insert(expand_k, prev_k)
            table.insert(nextScores, maxScoresB[k])
        end
        ::continue::
    end
end


function BeamSearch:prepareNbestList(nBest, completeHyps, completeHypsAtt, ref)
    local nBestList = {}
    local info
    for rank, hypo in ipairs(nBest) do
        -- stringx.count is fast
        local length = stringx.count(hypo, ' ') + 1
        local align = self:printAttention(completeHypsAtt[hypo], 1, false)
        --[[
        if ref then
            local reward = BLEU.score(hypo, ref) * 100  -- rescale for readability
            info = string.format('n=%d s=%.4f l=%d b=%.4f\t%s\t%s',
                                rank, completeHyps[hypo], length, reward, hypo, align)
        else
            info = string.format('n=%d s=%.4f l=%d\t%s\t%s',
                                rank, completeHyps[hypo], length, hypo, align)
        end
        --]]
        info = string.format('n=%d s=%.4f l=%d\t%s\t%s',
                                rank, completeHyps[hypo], length, hypo, align)
        table.insert(nBestList, info)
    end
    return nBestList
end
