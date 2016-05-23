-- Beam search
local utils = require 'misc.utils'
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


function PolicySearch:greedyRollin(input, length, prevState)
    --[[ roll-in with learned policy
    Parameters:
    - `input` : starting symbol
    - `length`: roll-in step
    - `prevState` : previous state before roll in

    Return:
    - `trajectory` : the greedy trajectory
    --]]
    local model = self.model
    model.buffers.prevState = prevState
    local trajectory = input:repeatTensor(length, 1)
    local x, logp = input, nil
    local cost = nil

    for t = 1, length do
        logp = model:stepDecoder(x)
        cost, x = logp:max(2) -- greedy
        trajectory[t] = x
    end

    local prevState = copyState(model.buffers.prevState)
    local logProb = model.buffers.logProb
    return trajectory, prevState, logProb
end

-- TODO: roll-out with reference policy
function PolicySearch:greedyRollout(input, length, prevState)
    --[[ roll-out with learned policy
    We collect sample for experience replay
    Parameters:
    - `input` : starting symbol
    - `length`: roll-in step
    - `prevState` : previous state before roll in

    Return:
    - `trajectory` : the greedy trajectory
    --]]

    local examples = {}

    local model = self.model
    model.buffers.prevState = prevState

    local trajectory = input:repeatTensor(length, 1)
    local x, logp = input, nil
    local cost = nil

    local s = nil -- state before taking an action
    local s2 = nil -- state after taking an action


    for t = 1, length do
        local D = {}
        s = model:selectState()

        logp = model:stepDecoder(x)
        s2 = model:selectState()
        local a = x:clone()
        cost, x = logp:max(2) -- greedy
        local _, a2 = logp:view(-1):topk(5, true)
        table.insert(examples, {s, a, s2, a2})
        trajectory[t] = x
    end

    local prevState = copyState(model.buffers.prevState)
    local logProb = model.buffers.logProb
    return trajectory, examples
end


function PolicySearch:search(x, maxLength, ref)

    -- refer to the model
    local model = self.model

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

    model:stepEncoder(x)
    local encodeState = copyState(model.buffers.prevState)

    K = 5  -- hand-code for now
    local score = 0
    local hypothesis = torch.zeros(T, 1):fill(self.idx_GO):type(self.dtype)

    local goIdx = hypothesis[1]:view(1, 1):clone()

    local stateQ = {}

    for t = 1, T  do
        for myT = 2, T - 1 do
            model.buffers.prevState = encodeState
            if t == myT then
                -- roll-in with learned policy
                local trajectory, prevState, logProb = self:greedyRollin(goIdx, myT, encodeState)
                -- rolling out
                local _, idx = logProb:topk(K, true)
                idx = idx:view(-1, 1)
                hypothesis[{{1, myT}, {}}] = trajectory
                for k = 1, K do
                    --print(myT, k, checksum(prevState))
                    hypothesis[myT] = idx[k]
                    local trj, D = self:greedyRollout(idx[k]:view(1, 1), T-myT, prevState)
                    hypothesis[{{myT +1, T}, {}}] = trj
                    --print(#D, T-myT)
                    local r = BLEU.rewardT(hypothesis:view(-1), y)
                    for i = myT, T-1 do
                        -- we do not take the last action
                        local Dx = D[i - myT + 1]
                        table.insert(Dx, r[i])
                        table.insert(stateQ, Dx)
                    end
                end
                -- now we should do some optimization to speed up this roll-in, roll-out
            end
        end
    end

    return stateQ
end
