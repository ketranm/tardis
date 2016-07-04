local THNN = require 'nn.THNN'
-- Pointer Network trained with REINFORCE
-- experimental setting
local PrtNet, parent = torch.class('nn.PrtNet', 'nn.Module')
require 'core.SqueezeDiag'

function PrtNet:__init(lambda, nsamples)
    self.lsm = nn.LogSoftMax()
    self.xt = nil
    self.scores = torch.Tensor()
    self.lsm = nn.LogSoftMax()
    self._sample = torch.LongTensor()
    self.sample = torch.LongTensor()
    self.nsamples = nsamples or 3 -- use 5 samples by default
    self.lambda = lambda or 0.01 -- reinforce weight
    self.sqd = nn.SqueezeDiag(1000)
    self.b = 0 -- baseline
    self.xent = nn.ClassNLLCriterion()
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

local function check_dims(x, dims)
    assert(x:dim() == #dims)
    for i, d in ipairs(dims) do
        assert(x:size(i) == d)
    end
end

local function get_dims(x)
    local dims = {}
    for i = 1, x:dim() do
        dims[i] = x:size(i)
    end
    return dims
end


function PrtNet:forward(input)
    local x, y = input[1], input[2]
    -- checking
    assert(x:size(1) == y:size(1), 'mini-batch size is inconsistent')
    assert(x:size(3) == y:size(3), 'embedding size is inconsistent')
    local N, D = x:size(1), x:size(3)
    local Tx, Ty = x:size(2), y:size(2)
    -- in this particular case Tx == Ty
    assert(Tx == Ty)
    -- make a copy for backprop
    self.N, self.D, self.T = N, D, Tx
    self.xt = x:transpose(2,3) -- (N, D, Tx)
    self.scores:resize(N, Ty, Tx):bmm(y, self.xt)
    self.buffer = self.sqd:forward(self.scores)
    local my_scores = self.buffer:view(N * Ty, Tx-1)

    -- compute attention
    self.logp = self.lsm(my_scores)
    self.prob = torch.exp(self.logp)
    -- sample this shit
    self.prob.multinomial(self._sample, self.prob, self.nsamples, true)
    -- handling nsamples

    self.logp = self.logp:repeatTensor(self.nsamples, 1) --  create nsamples copy
    self._sample = self._sample:t():contiguous()
    -- compute reward, we need to scale xent loss with the reward later
    self.reward = self.logp:gather(2, self._sample:view(-1, 1))
    self.output = self.xent:forward(self.logp, self._sample:view(-1))
    return self.output
end

function PrtNet:getSample()
    local sample = self._sample:view(self.N, -1)
    -- note that because we sample on squeezed matrix
    -- we need to restore the correct position
    for n = 1, self.N do
        for t = 1, self.T do
            if sample[n][t] >= t then
                sample[n][t] = sample[n][t] + 1
            end
        end
    end
    return sample
end

function PrtNet:backward(input, gradOutput)
    -- we never use gradOutput
    local x, y = input[1], input[2]
    if not self.b then
        self.b = self.output
    else
        self.b = 0.9 * self.b + 0.1 * self.output
    end
    -- cross entropy loss
    local dxent = self.xent:backward(self.logp, self._sample:view(-1))
    -- scale by the reward
    self.reward:add(-1 * self.b):mul(self.lambda)
    local _reward = self.reward:repeatTensor(1, self.T - 1)
    dxent:cmul(_reward)
    -- collapse gradient
    local df = dxent:split(self.N * self.T)
    assert(#df == self.nsamples)
    local dfx = df[1]
    for i = 2, self.nsamples do
        dfx:add(df[i])
    end
    -- done collapsing gradient of samples
    local dsqd = self.lsm:backward(self.buffer, dfx)
    local dbuff = self.sqd:backward(self.scores, dsqd:view(self.N, self.T, -1))

    local dx, dy = self.gradInput[1], self.gradInput[2]
    dx:resizeAs(x):bmm(dbuff:transpose(2,3), y)
    dy:resizeAs(y):bmm(dbuff, x)
    return self.gradInput
end

function PrtNet:clearState()
    self.scores:set()
end
