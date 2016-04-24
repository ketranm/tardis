-- Beam search
local utils = require 'util.utils'
local BLEU = require 'util.BLEU'
local PolicySearch = torch.class('PolicySearch')


function PolicySearch:__init(kwargs)
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

function PolicySearch:use(model)
    self.model = model
    self.model:evaluate() -- no dropout during testing
    self.dtype = model.params:type()
end


local function copyState(state)
    -- copy LSTM's state
    local copied = {}
    for _, state_layer in ipairs(state) do
        local s = {}
        for _, v in ipairs(state_layer) do
            table.insert(s, v:clone())
        end
        table.insert(copied, s)
    end

    return copied
end


local function checksum(state)
    -- for debugging
    local sum = 0
    for _, state_layer in ipairs(state) do
        for _, v in ipairs(state_layer) do
            sum = sum + v:sum()
        end
    end

    return sum
end


function PolicySearch:greedyRollin(input, length)
    -- roll-in with learned policy
    local model = self.model
    local trajectory = input:repeatTensor(length, 1)
    local x, logp = input, nil
    local cost = nil

    local is_done = false
    for t = 1, length do
        logp = model:stepDecoder(x)
        cost, x = logp:max(2) -- greedy
        trajectory[t] = x
    end
    return trajectory
end


function PolicySearch:search(x, maxLength, ref)

    local _ignore = self._ignore
    local id2word = self.id2word

    x = utils.encodeString(x, self.srcVocab, true)
    local y = utils.encodeString(ref, self.trgVocab)
    y = y:type(self.dtype)

    local trgLength = y:size(2)
    local y_inp = torch.Tensor(1, trgLength+1):fill(self.idx_GO):type(self.dtype)
    local y_out = torch.Tensor(1, trgLength+1):fill(self.idx_EOS):type(self.dtype)

    y_inp[{{}, {2,-1}}] = y
    y_out[{{}, {1,-2}}] = y

    local srcLength = x:size(2)

    local T = trgLength + 1
    local refReward = BLEU.rewardT(y, y)
    x = x:type(self.dtype)

    self.model:stepEncoder(x)

    K = 5  -- hand-code for now
    local score = 0
    local hypothesis = torch.zeros(T, 1):fill(self.idx_GO):type(self.dtype)

    -- handle the first prediction
    local curIdx = hypothesis[1]:view(-1, 1)
    local goIdx = curIdx:clone()

    local buffer = {}
    for i = 1, T - 1 do
        local curIdx = hypothesis[i]:view(-1, 1)
        local logProb = self.model:stepDecoder(curIdx)

        local s = self.model:selectState()
        _score, idx = logProb:view(-1):max(1)

        -- this is cheap
        -- take out next possible actions according to the model
        local _, possible_idx = logProb:view(-1):topk(K)
        local state_action = {s:clone(), idx, possible_idx}
        table.insert(buffer, state_action)
        score = score + _score[1]
        hypothesis[i+1] = idx
    end

    local stateQ = {}
    local r = BLEU.rewardT(hypothesis:view(-1)[{{2,T}}], y)
    local prev_r = 0
    for i = 1, #buffer-1 do
        -- ignore the first prediction, do add it if shit works
        local s , a = buffer[i][1], buffer[i][2]
        local s2, a2 = buffer[i + 1][1], buffer[i + 1][3]
        local ri = r[i] - prev_r
        prev_r = r[i]
        table.insert(stateQ, {s, a, s2, a2, ri})
    end

    return stateQ
end
