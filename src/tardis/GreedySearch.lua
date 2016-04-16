-- Beam search
local utils = require 'util.utils'
local BLEU = require 'util.BLEU'
local GreedySearch = torch.class('GreedySearch')


function GreedySearch:__init(kwargs)
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

function GreedySearch:use(model)
    self.model = model
    self.model:evaluate() -- no dropout during testing
    self.dtype = model.params:type()
end

function GreedySearch:search(x, maxLength, ref)

    local _ignore = self._ignore
    local id2word = self.id2word

    x = utils.encodeString(x, self.srcVocab, true)
    local srcLength = x:size(2)
    local T = maxLength or utils.round(srcLength * 1.4)

    x = x:type(self.dtype)

    self.model:stepEncoder(x)

    local scores = 0
    local hypothesis = torch.zeros(T, 1):fill(self.idx_GO):type(self.dtype)

    -- handle the first prediction
    local curIdx = hypothesis[1]:view(-1, 1)
    local logProb = self.model:stepDecoder(curIdx)
    local _, idx = logProb:view(-1):max(1)

    hypothesis[2] = idx
    for i = 2, T-1 do
        local curIdx = hypothesis[i]:view(-1, 1)
        local logProb = self.model:stepDecoder(curIdx)
        local _, idx = logProb:view(-1):max(1)
        if idx[1] == self.idx_EOS then
            local cand = utils.decodeString(hypothesis:view(-1), id2word, _ignore)
            return cand
        end
        hypothesis[i+1] = idx
    end


    local cand = utils.decodeString(hypothesis:view(-1), id2word, _ignore)
    return cand
end
