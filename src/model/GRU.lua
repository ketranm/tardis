local layer, parent = torch.class('nn.GRU', 'nn.Module')

--[[
A implementation of Gated Recurrent Unit
Author: Ke Tran <m.k.tran@uva.nl>
Version 0.1
--]]


function layer:__init(input_size, hidden_size)
    parent.__init(self)
    local D, H = input_size, hidden_size
    self.input_size, self.hidden_size = D, H

    self.weight = torch.Tensor(D + H, 3 * H)
    self.gradWeight = torch.Tensor(D + H, 3 * H)
    self.bias = torch.Tensor(3 * H)
    self.gradBias = torch.Tensor(3 * H)

    self:reset()
    -- buffer
    self.buffer = torch.Tensor() -- (N, T, H) Tensor
    self.grad_next_h = torch.Tensor()
    self.buffer_b = torch.Tensor()
    self.grad_a_buffer = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.h0 = torch.Tensor()
    self.grad_h0 = torch.Tensor()
    self.gates = torch.Tensor()
    self.output = torch.Tensor()
end

function layer:reset(std)
    if not std then
        std = 1.0 / math.sqrt(self.hidden_size + self.input_size)
    end
    self.bias:zero()
    self.bias[{{self.hidden_size+1, 2*self.hidden_size}}]:fill(1) -- bias of the forget gate
    self.weight:normal(0,std)
    return self
end

function layer:reset_state()
    self.h0 = self.h0.new()
end

function layer:set_grad_state(grad_state)
    local grad_h0 = grad_state[1]
    self.grad_h0:resizeAs(grad_h0):copy(grad_h0)
    self._init_grad_state = true
end

function layer:get_grad_state()
    return {self.grad_h0}
end

function layer:init_state(state)
    local h0 = state[1]
    self.h0:resizeAs(h0):copy(h0)
    self._copied_state = true
end

function layer:last_state()
    local T = self.output:size(2)
    local last_h = self.output[{{}, T}]
    return {last_h}
end

--[[
Args:
    input: A Tensor of (batch_size, seq_length, input_size)
    output: Tensor of (batch_size, seq_length, hidden_size)
--]]
function layer:updateOutput(input)
    local N, T = input:size(1), input:size(2)
    local D, H = self.input_size, self.hidden_size

    local h0 = self.h0
    if h0:nElement() == 0 or not self._copied then
        h0:resize(N, H):zero()
    end

    -- expand the bias to the batch_size
    local bias_expand = self.bias:view(1, 3 * H):expand(N, 3 * H)
    local Wx = self.weight[{{1, D}}]
    local Wh = self.weight[{{D + 1, D + H}}]
    local Wh1 = Wh[{{}, {1, 2 * H}}] -- for update and reset gate
    local Wh2 = Wh[{{}, {2 * H + 1, 3 * H}}]  -- for candidate hidden

    local h = self.output
    h:resize(N, T, H):fill(1)
    local prev_h = h0
    self.gates:resize(N, T, 3 * H):zero()
    self.buffer:resize(N, T, H)
    for t = 1, T do
        local xt = input[{{}, t}]
        local buffer_t = self.buffer[{{}, t}]
        local cur_gates = self.gates[{{}, t}]
        -- update x computation
        cur_gates:addmm(bias_expand, xt, Wx)
        cur_gates[{{}, {1, 2 * H}}]:addmm(prev_h, Wh1):sigmoid()  -- for update gate and reset gate
        local gate_z = cur_gates[{{}, {1, H}}]
        local gate_r = cur_gates[{{}, {H + 1, 2 * H}}]

        buffer_t:cmul(prev_h, gate_r)
        local cand_h = cur_gates[{{}, {2 * H + 1, 3 * H}}]:addmm(buffer_t, Wh2):tanh()

        local next_h = h[{{}, t}]
        next_h:add(-1, gate_z):cmul(prev_h)
        next_h:addcmul(gate_z, cand_h)
        prev_h = next_h
    end
    return self.output
end

function layer:backward(input, gradOutput, scale)
    scale = scale or 1.0
    local  x = input
    --local h0, x = input[1], input[2]
    local h0 = nil
    if not h0 then h0 = self.h0 end

    local grad_h0, grad_x = self.grad_h0, self.gradInput
    local h = self.output
    local grad_h = gradOutput

    local N, T = x:size(1), x:size(2)
    local H, D = self.hidden_size, self.input_size

    local Wx = self.weight[{{1, D}}]
    local Wh = self.weight[{{D + 1, D + H}}]
    local Wh1 = Wh[{{}, {1, 2 * H}}]
    local Wh2 = Wh[{{}, {2 * H + 1, 3 * H}}]

    local grad_Wx = self.gradWeight[{{1, D}}]
    local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
    local grad_Wh1 = grad_Wh[{{}, {1, 2 * H}}]
    local grad_Wh2 = grad_Wh[{{}, {2 * H + 1, 3 * H}}]
    local grad_b = self.gradBias
    -- TODO: carry on state

    grad_h0:resizeAs(h0):zero()
    grad_x:resizeAs(x):zero()
    local grad_next_h = self.grad_next_h:resizeAs(h0):zero()
    for t = T, 1, -1 do
        local next_h = h[{{}, t}]
        local buffer_t = self.buffer[{{}, t}]
        local prev_h = nil
        if t == 1 then
            prev_h = h0
        else
            prev_h = h[{{}, t - 1}]
        end

        grad_next_h:add(grad_h[{{}, t}])

        -- refer to gate
        local gate_z = self.gates[{{}, t, {1, H}}]
        local gate_r = self.gates[{{}, t, {H + 1, 2 * H}}]
        local cand_h = self.gates[{{}, t, {2 * H + 1, 3 * H}}]

        local grad_a = self.grad_a_buffer:resize(N, 3 * H):zero()
        local grad_az = grad_a[{{}, {1, H}}]
        local grad_ar = grad_a[{{}, {H + 1, 2 * H}}]
        local grad_ah = grad_a[{{}, {2 * H + 1, 3 * H}}]

        --[[Use grad_ar (reset gate) to store intermediate tensor.
        1. Store gradient that will come to update gate.
        2. Store tanh^2 to compute gradient that comes to candidate gate cand_h
        3. Store gradient come to the buffer (dot(gate_r, prev_h))
        --]]
        grad_ar:add(cand_h, -1, prev_h):cmul(grad_next_h)
        grad_az:fill(1):add(-1, gate_z):cmul(gate_z):cmul(grad_ar)

        local tanh2 = grad_ar:cmul(cand_h, cand_h)  -- tanh square
        grad_ah:fill(1):add(-1, tanh2):cmul(gate_z):cmul(grad_next_h)
        -- this will be (N, H) x (H, H) = N, H)
        grad_ar:mm(grad_ah, Wh2:t()) -- grad to buffer
        --grad_next_h:cmul(grad_ar, gate_r)
        grad_Wh2:addmm(scale, buffer_t:t(), grad_ah) -- TODO: check with the math

        -- we do not need buffer_t anymore, so use it to store temporal values
        -- compute gradient comes to grad_next_h
        buffer_t:fill(1):add(-1, gate_z):cmul(grad_next_h)
        -- now we do not need grad_next_h anymore, overwrite it
        -- grad_ar now is a gradient that comes to buffer
        grad_next_h:cmul(grad_ar, gate_r):add(buffer_t)
        -- reset buffer_t as we do not need it
        buffer_t:cmul(prev_h, grad_ar) -- gradient to reset gate

        grad_ar:fill(1):add(-1, gate_r):cmul(gate_r):cmul(buffer_t)
        grad_Wx:addmm(scale, x[{{}, t}]:t(), grad_a)
        grad_Wh1:addmm(scale, prev_h:t(), grad_a[{{}, {1, 2 * H}}])

        local grad_a_sum = self.buffer_b:resize(H):sum(grad_a, 1)
        grad_b:add(scale, grad_a_sum)

        grad_x[{{}, t}]:mm(grad_a, Wx:t())
        grad_next_h:addmm(grad_a[{{}, {1, 2 * H}}], Wh1:t()) -- accumulate from reset and update gate
    end
    grad_h0:copy(grad_next_h)
    return self.gradInput
end


--- this follows torch convention ---
function layer:clearState()
    self.gates:set()
    self.buffer_b:set()
    self.grad_next_h:set()
    self.grad_a_buffer:set()
    self.grad_h0:set()
    self.gradInput:set()
    self.output:set()
end


function layer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function layer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end