--[[ Memory Block
Implementation of Memory Block described in

Recurrent Memory Networks for Language Modeling
Ke Tran, Arianna Bisazza, and Christof Monz
In Proceedings of NAACL 2015
--]]

local layer, parent = torch.class('nn.Memory', 'nn.Module')

function layer:__init(memorySize)
    parent.__init(self)

    self.memorySize = memorySize
    self.output = torch.Tensor()

    self.gradInput = {torch.Tensor(), torch.Tensor(), torch.Tensor()}

    self.ht = torch.Tensor()
    self.att_buffer = torch.Tensor()
    self.softmax = nn.SoftMax()

    self.bufferAtt2Dims = torch.Tensor()
    self.attProb = torch.Tensor()
    self.attProb3Dims = torch.Tensor()
    self.att_deriv = torch.Tensor()
end


function layer:__slice(mem)
    --[[ Slicing memory `mem`
    In this simple implementation, we follow Tran et al
    TODO: use repeated input to get nicer gradient from lookup table
    That is if the input indices is 1 2 3 4 5 and memorySize = 3
    we present the input sequence 1 2 3 2 3 4 3 4 5 to the lookup table
    nn.LookupTable will normalize gradient by the count frequency, thus gradient will be nicer
    --]]
    local T = mem:size(2)
    assert(T > self.memorySize, 'sequence length should be longer than the memory size!')
    local slice = {}
    for t = 0, T - self.memorySize do
        local m = mem[{{},{t + 1, t + self.memorySize}, {}}]
        table.insert(slice, m)
    end
    return slice
end

function layer:updateOutput(input)
    local memIn, memOut, h = unpack(input)
    local N, T, D = h:size(1), h:size(2), h:size(3)
    local Mx = self.memorySize

    -- this will be a table of T 3D tensors
    -- each of them has size N, Mx, D, where Mx <= self.memorySize
    self.sliceMemIn = self:__slice(memIn)
    self.sliceMemOut = self:__slice(memOut)


    self.ht = h:transpose(2, 3)  -- this will be (N, D, T)

    self.att_buffer:resize(N, T * Mx, 1)

    for t = 1, T do
        local htt = self.ht[{{}, {}, {t}}] -- this will be a (N, D, 1) tensor
        local memIn_t = self.sliceMemIn[t]

        self.att_buffer[{{},{(t-1) * Mx + 1, t * Mx},{}}]:bmm(memIn_t, htt)
    end

    self.bufferAtt2Dims = self.att_buffer:view(-1, Mx)  -- this will be (N * T, Mx)
    self.attProb = self.softmax(self.bufferAtt2Dims)
    self.attProb3Dims = self.attProb:view(N, T, Mx)

    self.output:resize(N, T, D)
    for t = 1, T do
        self.output[{{},{t},{}}]:bmm(self.attProb3Dims[{{},{t},{}}], self.sliceMemOut[t])
    end

    return self.output
end


function layer:backward(input, gradOutput)
    local memIn, memOut, h = unpack(input)
    local N, T, D = h:size(1), h:size(2), h:size(3)
    local Mx = self.memorySize

    local gradMemIn, gradMemOut, grad_h = unpack(self.gradInput)
    gradMemIn:resizeAs(memIn):zero()
    gradMemOut:resizeAs(memOut):zero()
    grad_h:resizeAs(self.ht):zero()

    -- each element of the slice is a (N, Mx, D) tensor
    local sliceGradMemIn = self:__slice(gradMemIn)
    local sliceGradMemOut = self:__slice(gradMemOut)


    self.att_deriv:resize(N, T, Mx)  -- same size as attProb3Dims
    self.attProb3Dims = self.attProb3Dims:transpose(2, 3)  -- (N, Mx, T)
    for t = 1, T do
        local att_t = self.attProb3Dims[{{}, {}, {t}}]  -- (N, Mx, 1)
        local dout_t = gradOutput[{{}, {t}, {}}]  -- (N, 1, D)
        sliceGradMemOut[t]:bmm(att_t, dout_t)  -- (N, Mx, D)
        self.att_deriv[{{}, {t}, {}}]:bmm(dout_t, self.sliceMemOut[t]:transpose(2, 3))  -- (N, 1, Mx)
    end

    self.att_deriv = self.att_deriv:view(-1, Mx)  -- (N * T, Mx)
    local gradSoftMax = self.softmax:backward(self.bufferAtt2Dims, self.att_deriv)
    gradSoftMax = gradSoftMax:view(N, -1 , 1)  -- (N, T * Mx, 1)
    -- double check this part
    for t = 1, T do
        local ht = h[{{}, {t}, {}}] -- (N, 1, D)
        -- get out grad of the unnormalized attention
        local gradAtt = gradSoftMax[{{},{(t-1) * Mx + 1, t * Mx}, {}}] -- (N, Mx, 1)
        grad_h[{{}, {}, {t}}]:bmm(self.sliceMemIn[t]:transpose(2, 3), gradAtt)
        sliceGradMemIn[t]:bmm(gradAtt, ht)
    end

    self.gradInput[3] = grad_h:transpose(2, 3)
    return self.gradInput
end


function layer:clearState()
    nn.utils.clear(self, {
        'output',
        'gradInput'
    })
end
