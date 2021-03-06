local THNN = require 'nn.THNN'

--[[
Hard attention mechanism
This is an implementation of Bilinear Attention for computing the softmax.

Detail of Bilinear Attention can be found in:

Effective Approaches to Attention-based Neural Machine Translation. EMNLP-15
http://www.aclweb.org/anthology/D15-1166

There is a little difference in the performance of various attention mechanisms, 
the choice of Bilinear form is due to its simplicity.

After the softmax is computed, instead of weighting average by the softmax output, 
we choose one location to attend by drawing a sample from multinomial distribution 
paramatrized by the softmax. By sampling from multinomial, we can't take derivative anymore.
So the update rule has to be slightly different. It has a form of REINFORCE. 

For details, see the following paper:

Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
International Conference for Machine Learning (2015)
http://arxiv.org/abs/1502.03044

In this implementation, we do not add Entropy regularization.

IMPORTANT NOTE:

Hard attention exhibits high variance (as it uses REINFORCE update rule).
When training hard attention, it's better to start a few epochs with soft-attention first.
This will help stabilizing training process.

Author: Ke Tran <m.k.tran@uva.nl>
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

    self.sample = torch.LongTensor()
    self.beta = nil
    self.criterion = nn.ClassNLLCriterion()
    self._sample = torch.LongTensor()
    self.lambda = 0.001 -- reinforce weight
end

-- need to double check this
function Glimpse:type(type, tensorCache)
    parent.type(self, type, tensorCache)

    if type == 'torch.CudaTensor' then
        -- CUDA uses _sorted and _indices temporary tensors
        self._sorted = self.weight.new()
        self._indices = self.weight.new()
        self._count = self.weight.new()
        self._input = self.weight.new()
    else
        -- self._count and self._input should only be converted if using Cuda
        self._count = torch.IntTensor()
        self._input = torch.LongTensor()
    end

    return self
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
    --[[ Stochastic attention update
    Parameters
    - `input` : a table {x, y} of two tensors x: (N, Tx, D) Tensor y: (N, Ty, D) Tensor
    - `output` : context tensor of (N, Ty, D)
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
    self.sample:add(self._sample, offset:view(-1))

    self.output = x:view(-1, D):index(1, self.sample:view(-1)):view(N, Ty, D)
    return self.output

end

function Glimpse:backward(input, gradOutput, scale)
    scale = scale or 1.0
    local x, y, reward = input[1], input[2], input[3]
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
    -- hand-coded now, better to move it to option
    if not self.beta then
        self.beta = reward:mean() -- initialize the first baseline
    end

    self.beta = 0.9 * self.beta + 0.1 * reward:mean()

    -- the reward is the log probability of predicting the next word
    -- we need to reduce the variance by using the baseline
    reward:add(-1 * self.beta) -- reducing variance


    local _reward = reward:view(-1, 1):repeatTensor(1, Tx)
    local grad_xent = self.criterion:backward(self.log_att, self._sample)

    grad_xent:cmul(_reward):mul(self.lambda)

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
