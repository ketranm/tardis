local layer, parent = torch.class('nn.GRU', 'nn.Module')

--[[ A implementation of Gated Recurrent Unit
Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
Author: Ke Tran <m.k.tran@uva.nl>
Version 1.0
--]]


function layer:__init(input_dim, hidden_dim)
    parent.__init(self)
    local D, H = input_dim, hidden_dim
    self.input_dim, self.hidden_dim = D, H

    self.weight = torch.Tensor(D + H, 3 * H)
    self.gradWeight = torch.Tensor(D + H, 3 * H)
    self.bias = torch.Tensor(3 * H)
    self.gradBias = torch.Tensor(3 * H)

    self:reset()

    self.gates = torch.Tensor()
    -- buffer
    self.buffer = torch.Tensor() -- (N, T, H) Tensor
    self.grad_next_h = torch.Tensor()
    self.buffer_b = torch.Tensor()
    self.grad_a_buffer = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.h0 = torch.Tensor()
    self.remember_states = false
    self.grad_h0 = torch.Tensor()
end


function layer:reset(std)
    if not std then
        std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
    end
    self.bias:zero()
    self.bias[{{self.hidden_dim+1, 2*self.hidden_dim}}]:fill(1) -- bias of the forget gate
    self.weight:normal(0,std)
    return self
end


function layer:resetStates()
    self.h0 = self.h0.new()
end


function layer:lastStates()
    local prev_T = self.output:size(2)
    return {self.output[{{}, prev_T}]}
end


function layer:setStates(state)
    local h0 = state[1]
    self.h0:resizeAs(h0):copy(h0)
end


local function check_dims(x, dims)
    assert(x:dim() == #dims)
    for i, d in ipairs(dims) do
        assert(x:size(i) == d)
    end
end


function layer:_unpack_input(input)
    local h0, x = nil, nil
    if torch.type(input) == 'table' and #input == 2 then
        h0, x = unpack(input)
    elseif torch.isTensor(input) then
        x = input
    else
        assert(false, 'invalid input')
    end
    return h0, x
end


function layer:_get_sizes(input, gradOutput)
    local h0, x = self:_unpack_input(input)
    local N, T = x:size(1), x:size(2)
    local H, D = self.hidden_dim, self.input_dim
    check_dims(x, {N, T, D})
    if h0 then
        check_dims(h0, {N, H})
    end
    if gradOutput then
        check_dims(gradOutput, {N, T, H})
    end
  return N, T, D, H
end


function layer:setGrad(grad0)
    self._set_grad = true
    self.grad_h0 = grad0[1]
end


function layer:getGrad()
    return {self.grad_h0}
end

--[[
Input:
- h0: Initial hidden state, (N, H)
- x: Input sequence, (N, T, D)

Output:
- h: Sequence of hidden states, (N, T, H)
--]]
function layer:updateOutput(input)
    local h0, x = self:_unpack_input(input)
    local N, T, D, H = self:_get_sizes(input)

    self._return_grad_h0 = (h0 ~= nil)

    if not h0 then
        h0 = self.h0
        if h0:nElement() == 0 or not self.remember_states then
            h0:resize(N, H):zero()
        elseif self.remember_states then
            local prev_N, prev_T = self.output:size(1), self.output:size(2)
            assert(prev_N == N, 'batch sizes must be the same to remember states')
            h0:copy(self.output[{{}, prev_T}])
        end
    end

    -- expand the bias to the batch_size
    local bias_expand = self.bias:view(1, 3 * H):expand(N, 3 * H)
    local Wx = self.weight[{{1, D}}]
    local Wh = self.weight[{{D + 1, D + H}}]
    local Wrz = Wh[{{}, {1, 2 * H}}] -- for update and reset gate
    local Whc = Wh[{{}, {2 * H + 1, 3 * H}}]  -- for candidate hidden

    local h = self.output
    h:resize(N, T, H):fill(1)
    local prev_h = h0
    self.gates:resize(N, T, 3 * H):zero()
    self.buffer:resize(N, T, H):zero()
    for t = 1, T do
        local cur_x = x[{{}, t}]
        local cur_buffer = self.buffer[{{}, t}]
        local cur_gates = self.gates[{{}, t}]
        cur_gates:addmm(bias_expand, cur_x, Wx)
        -- compute update gate and reset gate
        cur_gates[{{}, {1, 2 * H}}]:addmm(prev_h, Wrz):sigmoid()
        local z = cur_gates[{{}, {1, H}}]
        local r = cur_gates[{{}, {H + 1, 2 * H}}]
        cur_buffer:cmul(prev_h, r)
        local hc = cur_gates[{{}, {2 * H + 1, 3 * H}}]:addmm(cur_buffer, Whc):tanh()
        local next_h = h[{{}, t}]
        next_h:csub(z):cmul(prev_h) -- (1-z)*prev_h + z*hc
        next_h:addcmul(z, hc)
        prev_h = next_h
    end
    return self.output
end

function layer:backward(input, gradOutput, scale)
    scale = scale or 1.0
    local h0, x = self:_unpack_input(input)
    if not h0 then h0 = self.h0 end

    local grad_h0, grad_x = self.grad_h0, self.gradInput
    local h = self.output
    local grad_h = gradOutput

    local N, T, D, H = self:_get_sizes(input, gradOutput)

    local Wx = self.weight[{{1, D}}]
    local Wh = self.weight[{{D + 1, D + H}}]
    local Wrz = Wh[{{}, {1, 2 * H}}]
    local Whc = Wh[{{}, {2 * H + 1, 3 * H}}]

    local grad_Wx = self.gradWeight[{{1, D}}]
    local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
    local grad_Wrz = grad_Wh[{{}, {1, 2 * H}}]
    local grad_Whc = grad_Wh[{{}, {2 * H + 1, 3 * H}}]
    local grad_b = self.gradBias

    if not self._set_grad then
        grad_h0:resizeAs(h0):zero()
    end

    grad_x:resizeAs(x):zero()
    local grad_next_h = self.grad_next_h:resizeAs(h0):copy(grad_h0)
    for t = T, 1, -1 do
        local next_h = h[{{}, t}]
        local cur_buffer = self.buffer[{{}, t}]
        local prev_h = nil
        if t == 1 then
            prev_h = h0
        else
            prev_h = h[{{}, t - 1}]
        end

        grad_next_h:add(grad_h[{{}, t}])

        local z = self.gates[{{}, t, {1, H}}]
        local r = self.gates[{{}, t, {H + 1, 2 * H}}]
        local hc = self.gates[{{}, t, {2 * H + 1, 3 * H}}]
        -- fill with 1 for convenience
        local grad_a = self.grad_a_buffer:resize(N, 3 * H):fill(1)
        local grad_az = grad_a[{{}, {1, H}}]
        local grad_ar = grad_a[{{}, {H + 1, 2 * H}}]
        local grad_ah = grad_a[{{}, {2 * H + 1, 3 * H}}]

        --[[Use grad_ar (reset gate) to store intermediate tensor.
        1. Store gradient that will come to update gate.
        2. Store tanh^2 to compute gradient that comes to candidate gate cand_h
        3. Store gradient come to the buffer (dot(gate_r, prev_h))
        --]]
        grad_ar:add(hc, -1, prev_h):cmul(grad_next_h) -- (hc - prev_h) * grad_next_h
        grad_az:csub(z):cmul(z):cmul(grad_ar)

        local tanh2 = grad_ar:cmul(hc, hc)  -- tanh square
        grad_ah:csub(tanh2):cmul(z):cmul(grad_next_h)
        -- this will be (N, H) x (H, H) = N, H)
        grad_ar:mm(grad_ah, Whc:t()) -- grad to buffer
        grad_Whc:addmm(scale, cur_buffer:t(), grad_ah)

        -- we do not need cur_buffer anymore, so use it to store temporal values
        -- compute gradient comes to grad_next_h
        cur_buffer:fill(1):csub(z):cmul(grad_next_h)
        -- now we do not need grad_next_h anymore, overwrite it
        -- grad_ar now is a gradient that comes to buffer
        grad_next_h:cmul(grad_ar, r):add(cur_buffer)
        -- reset cur_buffer as we do not need it
        cur_buffer:cmul(prev_h, grad_ar) -- gradient to reset gate

        grad_ar:fill(1):csub(r):cmul(r):cmul(cur_buffer)
        grad_Wx:addmm(scale, x[{{}, t}]:t(), grad_a)
        grad_Wrz:addmm(scale, prev_h:t(), grad_a[{{}, {1, 2 * H}}])

        local grad_a_sum = self.buffer_b:resize(3 * H):sum(grad_a, 1)
        grad_b:add(scale, grad_a_sum)

        grad_x[{{}, t}]:mm(grad_a, Wx:t())
        grad_next_h:addmm(grad_a[{{}, {1, 2 * H}}], Wrz:t()) -- accumulate from reset and update gate
    end

    grad_h0:copy(grad_next_h)

    return self.gradInput
end

function layer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function layer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function layer:clearState()
    nn.utils.clear(self, {
        'output',
        'gates',
        'buffer_b',
        'grad_h0',
        'grad_next_h',
        'grad_a_buffer',
        'gradInput'
    })
end
