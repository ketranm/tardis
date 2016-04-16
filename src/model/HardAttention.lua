local THNN = require 'nn.THNN'

--[[
Hard attention mechanism
This is an implementation of Bilinear Attention http://www.aclweb.org/anthology/D15-1166
There is a little difference in the performance of various attention mechanims, 
the choice of Bilinear attention form is due to it simplicity.

After the softmax is computed, instead of weighting average by the softmax output, 
we choose one location to attend by drawing a sample from multinomial distribution 
paramatrized by the softmax. 
This is a form of REINFORCE. For details, see the following paper
http://arxiv.org/pdf/1502.03044v2.pdf

--]]
local Glimpse, parent = torch.class('nn.Glimpse', 'nn.Module')


function Glimpse:__init(input_size)
    self.input_size = input_size
    self.weight = torch.Tensor(input_size, input_size)
    self.gradWeight = torch.Tensor(input_size, input_size)

    self.gradInput = {torch.Tensor(), torch.Tensor()}
    -- buffer
    self.mul_buffer = torch.Tensor()
    self.att_buffer = torch.Tensor()
    self.deriv_buffer = torch.Tensor() -- buffer derivative
    -- helper
    self.lsm = nn.LogSoftMax()
    self.output = torch.Tensor()
    self:reset()

    -- shit needed for hard attention
    self.shouldScaleGradByFreq = true
    self._count = self._count or torch.IntTensor()

    self.sample = torch.LongTensor()
    self.beta = 0
    self.criterion = nn.ClassNLLCriterion()
    self._sample = torch.LongTensor()
    self.tau = 0.1
end

function Glimpse:reset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1./math.sqrt(self.weight:size(2))
    end
    self.weight:uniform(-stdv, stdv)
    return self
end

function Glimpse:updateOutput(input)
    --[[
    Args
        input: a table {x, y} of two tensors x: (N, Tx, D) Tensor y: (N, Ty, D) Tensor
        output: context tensor of (N, Ty, D)
    ]]
    local x, y = input[1], input[2]
    local D = self.input_size
    assert(x:size(1) == y:size(1))

    local N, Tx, Ty = x:size(1), x:size(2), y:size(2)
    local y2 = y:view(N * Ty, D)
    -- transform
    self.mul_buffer:resize(N * Ty, D):mm(y2, self.weight)
    -- reshape back (N, Ty, D)
    self.mul_buffer = self.mul_buffer:view(N, Ty, D)
    -- xt: (N, D, Tx)
    self.xt = x:transpose(2,3)
    -- buffer_a: (N, Ty, Tx)
    self.att_buffer:resize(N, Ty, Tx):bmm(self.mul_buffer, self.xt)

    -- 2D view
    local buffer_att = self.att_buffer:view(N * Ty, Tx)
    self.log_att = self.lsm(buffer_att)

    self.att = torch.exp(self.log_att)
    -- sample attention
    self.att.multinomial(self._sample, self.att, 1) -- (N * Ty) tensor

    -- shifting index
    self._sample = self._sample:view(-1)
    local offset = torch.range(0, N-1):view(-1, 1):mul(Tx):repeatTensor(1, Ty):typeAs(self._sample)
    self.sample:resizeAs(self._sample)
    self.sample:add(self._sample + offset:view(-1))

    self.output = x:view(-1, D):index(1, self.sample:view(-1)):view(N, Ty, D)
    return self.output

end

function Glimpse:backward(input, gradOutput, reward, scale)
    scale = scale or 1.0

    local x, y = input[1], input[2]
    local N, Tx, Ty = x:size(1), x:size(2), y:size(2)
    local D = self.input_size  -- for readability

    local dx, dy, dz = self.gradInput[1], self.gradInput[2], self.deriv_buffer
    -- dx: (N, Tx, D)
    dx:resize(N * Tx, D):zero()

    -- use lookup table for scaling. Neat idea
    dx.THNN.LookupTable_accGradParameters(
        self.sample:cdata(),
        gradOutput:view(-1, D):cdata(),
        dx:cdata(),
        self._count:cdata(),
        THNN.optionalTensor(self._sorted),
        THNN.optionalTensor(self._indices),
        self.shouldScaleGradByFreq or false,
        self.paddingValue or 0,
        scale or 1
    )

    dx:resize(N, Tx, D)
    -- update moving average baseline
    -- handcoded now, better to move it to option
    self.beta = 0.9 * self.beta + 0.1 * reward:mean()

    -- the reward is the log probability of predicting the next word
    -- we need to reduce the variance by using the baseline
    reward:add(-1 * self.beta) -- reducing variance


    local _reward = reward:view(-1, 1):repeatTensor(1, Tx)
    local grad_xent = self.criterion:backward(self.log_att, self._sample)
    
    grad_xent:cmul(_reward):mul(self.tau)

    -- backprop normally
    local buffer_ax = self.att_buffer
    local deriv_a = self.lsm:backward(buffer_ax, grad_xent)
    deriv_a = deriv_a:view(N, Ty, Tx)

    dx:baddbmm(deriv_a:transpose(2,3), self.mul_buffer)

    dz:resizeAs(self.mul_buffer):bmm(deriv_a, x)  -- deriv of self.mul_buffer
    dz = dz:view(N * Ty, D)
    self.gradWeight:addmm(scale, y:view(N * Ty, D):t(), dz)
    dy:resizeAs(dz):addmm(0, 1, dz, self.weight:t())
    dy = dy:view(N, Ty, -1)

    return self.gradInput
end

function Glimpse:accGradParameters(input, gradOutput, scale)
    scale = scale or 1.0
    return Glimpse:backward(input, gradOutput, scale)

end

Glimpse.sharedAccUpdateGradParameters = Glimpse.accUpdateGradParameters

function Glimpse:getAttention()
    -- return attention distribution
    return self.att
end

function Glimpse:clearState()
    self.mul_buffer.set()
    self.att_buffer.set()
    self.deriv_buffer.set()
end


function Glimpse:__tostring__()
    return torch.type(self) .. string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
