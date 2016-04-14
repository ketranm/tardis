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
    self.softmax = nn.SoftMax()
    self.output = torch.Tensor()
    self:reset()
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
    self.att = self.softmax(buffer_att)
    self.att = self.att:view(N, Ty, Tx)
    self.output:resize(N, Ty, D):bmm(self.att, x)
    return self.output

end

function Glimpse:backward(input, gradOutput, scale)
    scale = scale or 1.0
    local x, y = input[1], input[2]
    local N, Tx, Ty = x:size(1), x:size(2), y:size(2)
    local D = self.input_size  -- for readability

    local att = self.att:transpose(2,3) -- (N, Tx, Ty)
    local dx, dy, dz = self.gradInput[1], self.gradInput[2], self.deriv_buffer
    -- dx: (N, Tx, D)
    dx:resizeAs(x):bmm(att, gradOutput)
    -- derivative of att
    dz:resize(N, Ty, Tx):bmm(gradOutput, self.xt)

    -- (N * Ty, D)
    local buffer_ax = self.att_buffer
    local deriv_a = self.softmax:backward(buffer_ax, dz:view(N * Ty, Tx))
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
