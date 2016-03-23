local layer, parent = torch.class('nn.LSTM', 'nn.Module')

--[[This implementation is based on Justin Johnson's torch-rnn
--]]

function layer:__init(input_size, hidden_size)
    parent.__init(self)

    local D, H = input_size, hidden_size
    self.input_size, self.hidden_size = D, H

    self.weight = torch.Tensor(D + H, 4 * H)  -- for fast computing
    self.gradWeight = torch.Tensor(D + H, 4 *H):zero()
    self.bias = torch.Tensor(4 * H)
    self.gradBias = torch.Tensor(4 * H):zero()

    self:reset() -- reset parameters

    self.cell = torch.Tensor()    -- This will be (N,T,H)
    self.gates = torch.Tensor()   -- This will be (N,T,H)
    self.buffer_h = torch.Tensor() -- This will be (N,H)
    self.buffer_c = torch.Tensor() -- This will be (N,H)
    self.buffer_b = torch.Tensor() -- This will be (H,)
    self.grad_a_buffer = torch.Tensor()

    self.h0 = torch.Tensor() -- initial hidden state
    self.c0 = torch.Tensor() -- initial cell state

    self.grad_h0 = torch.Tensor()
    self.grad_c0 = torch.Tensor()

    self.gradInput =  torch.Tensor()
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
    self.c0 = self.c0.new()
end

-- helper function
local function check_dims(x, dims)
    assert(x:dim() == #dims)
    for i, d in ipairs(dims) do
        assert(x:size(i) == d)
    end
end

function layer:_get_sizes(input)
    -- return batch size and sequence length
    return input:size(1), input:size(2)
end

function layer:set_grad_state(grad_state)
    --[[This function is helpful for seq2seq model
    It set the initial gradient that comes to state of the encoder
    ]]
    local grad_c0, grad_h0 = unpack(grad_state)
    self.grad_c0:resizeAs(grad_c0):copy(grad_c0)
    self.grad_h0:resizeAs(grad_h0):copy(grad_h0)
    -- use flag to guide backward pass
    self._init_grad_state = true
end


function layer:get_grad_state()
    --[[This is useful for seq2seq model
    We use it to get gradient that comes out of state of the decoder
    --]]
    return {self.grad_c0, self.grad_h0}
end

function layer:init_state(state)
    --[[This is useful for Seq2seq where we initialize the decoder from encoder's state
    When the RNN starts from a given state, we might want to get the gradient comes of of it
    so we set the copied flag to indicate so
    --]]
    local c0, h0 = unpack(state)
    self.c0:resizeAs(c0):copy(c0)
    self.h0:resizeAs(h0):copy(h0)
    self._copied_state = true
end

function layer:last_state()
    -- useful in Seq2seq model
    local N, T = self.cell:size(1), self.cell:size(2)
    local last_c = self.cell[{{}, T}]
    local last_h = self.output[{{}, T}]
    return {last_c, last_h}
end

--[[
Args:
    input: Input sequence (N,T,D)
    Output: sequence of hidden state (N,T,H)
--]]

function layer:updateOutput(input)
    local N, T = self:_get_sizes(input)
    local D, H = self.input_size, self.hidden_size
    local c0, h0 = self.c0, self.h0

    if c0:nElement() == 0 or not self._copied_state then
        c0:resize(N,H):zero()
    end


    if h0:nElement() == 0 or not self._copied_state then
        h0:resize(N,H):zero()
    end

    -- compute the output
    -- expand bias to N, the batch size
    local bias_expand = self.bias:view(1, 4*H):expand(N, 4*H)
    local Wx = self.weight[{{1, D}}]
    local Wh = self.weight[{{D + 1, D + H}}]

    local h, c = self.output, self.cell
    h:resize(N,T,H):zero()
    c:resize(N,T,H):zero()
    -- previous hidden and cell
    local prev_h, prev_c = h0, c0
    self.gates:resize(N,T,4*H):zero()
    for t = 1,T do
        local x_t = input[{{}, t}]
        local next_h = h[{{}, t}]
        local next_c = c[{{}, t}]
        local cur_gates = self.gates[{{}, t}]
        cur_gates:addmm(bias_expand, x_t, Wx)
        cur_gates:addmm(prev_h, Wh)
        cur_gates[{{}, {1, 3 * H}}]:sigmoid()
        cur_gates[{{}, {3 * H + 1, 4 * H}}]:tanh()
        -- pointer to gates
        local i = cur_gates[{{}, {1, H}}]
        local f = cur_gates[{{}, {H + 1, 2 * H}}]
        local o = cur_gates[{{}, {2 * H + 1, 3 * H}}]
        local g = cur_gates[{{}, {3 * H + 1, 4 * H}}]
        next_h:cmul(i,g)
        next_c:cmul(f,prev_c):add(next_h)
        next_h:tanh(next_c):cmul(o)
        prev_h, prev_c = next_h, next_c
    end

    return self.output
end


function layer:backward(input, gradOutput, scale)
    scale = scale or 1.0
    local c0, h0 = self.c0, self.h0

    local grad_c0, grad_h0, gradInput = self.grad_c0, self.grad_h0, self.gradInput
    local h, c = self.output, self.cell
    local grad_h = gradOutput

    local N, T = self:_get_sizes(input)
    local D, H = self.input_size, self.hidden_size

    local Wx = self.weight[{{1, D}}]
    local Wh = self.weight[{{D + 1, D + H}}]

    local grad_Wx = self.gradWeight[{{1, D}}]
    local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
    local grad_b = self.gradBias

    gradInput:resizeAs(input):zero()

    
    local grad_next_h = self.buffer_h:resizeAs(h0)
    local grad_next_c = self.buffer_c:resizeAs(c0)

    if not self._init_grad_state then
        -- reset gradient
        grad_h0:resizeAs(h0):zero()
        grad_c0:resizeAs(c0):zero()
    end
    grad_next_h:copy(grad_h0)
    grad_next_c:copy(grad_c0)

    for t = T, 1, -1 do
        local next_h, next_c = h[{{}, t}], c[{{}, t}]
        local prev_h, prev_c
        if t == 1 then
            prev_h, prev_c = h0, c0
        else
            prev_h, prev_c = h[{{}, t - 1}], c[{{}, t - 1}]
        end
        grad_next_h:add(grad_h[{{}, t}])

        local i = self.gates[{{}, t, {1, H}}]
        local f = self.gates[{{}, t, {H + 1, 2 * H}}]
        local o = self.gates[{{}, t, {2 * H + 1, 3 * H}}]
        local g = self.gates[{{}, t, {3 * H + 1, 4 * H}}]

        local grad_a = self.grad_a_buffer:resize(N, 4*H):zero()
        local grad_ai = grad_a[{{}, {1, H}}]       -- gradient to input gate
        local grad_af = grad_a[{{}, {H + 1, 2 * H}}]   -- gradient to forget gate
        local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}] -- gradient to output gate
        local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}] -- gradient to update gate

        -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
        -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
        -- to compute grad_ao; the other values can be overwritten after we compute
        -- grad_next_c
        local tanh_next_c = grad_ai:tanh(next_c)
        local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
        local my_grad_next_c = grad_ao
        my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_next_h)
        grad_next_c:add(my_grad_next_c)

        -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
        -- that we can overwrite it
        grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_next_h)

        -- Use grad_ai as a temporary buffer for computing grad_ag
        local g2 = grad_ai:cmul(g,g)
        grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

        -- We don't need any temporary storage for these so do them last
        grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
        grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)

        gradInput[{{}, t}]:mm(grad_a, Wx:t())
        grad_Wx:addmm(scale, input[{{}, t}]:t(), grad_a)
        grad_Wh:addmm(scale, prev_h:t(), grad_a)
        local grad_a_sum = self.buffer_b:resize(4 * H):sum(grad_a, 1)
        grad_b:add(scale, grad_a_sum)

        grad_next_h:mm(grad_a, Wh:t())
        grad_next_c:cmul(f)
    end
    grad_h0:copy(grad_next_h)
    grad_c0:copy(grad_next_c)
    return self.gradInput
end


--- this follows torch convention ---
function layer:clearState()
    self.cell:set()
    self.gates:set()
    self.buffer_h:set()
    self.buffer_c:set()
    self.buffer_b:set()
    self.grad_a_buffer:set()

    self.grad_c0:set()
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
