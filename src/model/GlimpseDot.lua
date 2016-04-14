local GlimpseDot, parent = torch.class('nn.GlimpseDot', 'nn.Module')


function GlimpseDot:__init(input_size)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    -- buffer
    self.att_buffer = torch.Tensor()
    self.deriv_buffer = torch.Tensor() -- buffer derivative
    -- helper
    self.softmax = nn.SoftMax()
    self.output = torch.Tensor()
end

function GlimpseDot:updateOutput(input)
    --[[
    Args
        input: a table {x, y} of two tensors x: (N, Tx, D) Tensor y: (N, Ty, D) Tensor
        output: context tensor of (N, Ty, D)
    ]]
    local x, y = input[1], input[2]
    assert(x:size(1) == y:size(1))
    local N, Tx, Ty = x:size(1), x:size(2), y:size(2)
    -- xt: (N, D, Tx)
    self.xt = x:transpose(2,3)
    -- buffer_a: (N, Ty, Tx)
    self.att_buffer:resize(N, Ty, Tx):bmm(y, self.xt)
    -- 2D view
    local buffer_att = self.att_buffer:view(N * Ty, Tx)
    self.att = self.softmax(buffer_att)
    self.att = self.att:view(N, Ty, Tx)
    self.output:resizeAs(y):bmm(self.att, x)
    return self.output
end

function GlimpseDot:backward(input, gradOutput, scale)
    scale = scale or 1.0
    local x, y = input[1], input[2]
    local N, Tx, Ty = x:size(1), x:size(2), y:size(2)

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

    dx:baddbmm(deriv_a:transpose(2,3), y)
    dy:resizeAs(y):bmm(deriv_a, x)
    return self.gradInput
end

function GlimpseDot:accGradParameters(input, gradOutput, scale)
    scale = scale or 1.0
    return GlimpseDot:backward(input, gradOutput, scale)

end

GlimpseDot.sharedAccUpdateGradParameters = GlimpseDot.accUpdateGradParameters

function GlimpseDot:getAttention()
    -- return attention distribution
    return self.att
end

function GlimpseDot:clearState()
    self.att_buffer.set()
    self.deriv_buffer.set()
end
