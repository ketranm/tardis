--[[ Memory Block
Implementation of Memory Block described in

Recurrent Memory Networks for Language Modeling
Ke Tran, Arianna Bisazza, and Christof Monz
In Proceedings of NAACL 2015
--]]

local layer, parent = torch.class('nn.Memory', 'nn.Module')

function layer:__init(memorySize, memInp, memOut, posMat)
    parent.__init(self)

    self.memorySize = memorySize
    self.output = torch.Tensor()

    self.gradInput = {torch.Tensor(), torch.Tensor()}
    self.softmax = nn.SoftMax()
    -- assigning memory lookup table
    self.posMat = posMat
    self.memInp = memInp
    self.memOut = memOut

    self.time = torch.range(1, self.memorySize)
    -- temporary tensors
    self.ht = torch.Tensor()  -- copy of h transpose
    self.attScore3Dims = torch.Tensor()

    self.attProb3Dims = torch.Tensor()
    self.gradAttProb = torch.Tensor()

    self.cache = {}
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
    assert(T % self.memorySize == 0)
    local slice = {}
    for t = 1, T, self.memorySize do
        local m = mem[{{},{t, t + self.memorySize - 1}, {}}]
        table.insert(slice, m)
    end
    return slice
end

function layer:updateOutput(input)
    local x, h = unpack(input)
    local N, T, D = h:size(1), h:size(2), h:size(3)
    local Mx = self.memorySize

    local mem_inp = self.memInp:forward(x)
    local mem_out = self.memOut:forward(x)

    local time_b = self.time:repeatTensor(N, T)  -- (N, T * Mx)

    local pos_mat = self.posMat:forward(time_b)  -- positional bias

    -- we reuse mem_inp
    mem_inp:add(pos_mat)  -- inject bias

    self.cache = {pos_mat, time_b}

    -- this will be a table of T 3D tensors
    -- each of them has size N, Mx, D, where Mx <= self.memorySize
    self.slicedMemInp = self:__slice(mem_inp)
    self.slicedMemOut = self:__slice(mem_out)


    self.ht = h:transpose(2, 3)  -- (N, D, T)

    self.attScore3Dims:resize(N, T * Mx, 1)

    for t = 1, T do
        local htt = self.ht[{{}, {}, {t}}]  -- (N, D, 1) tensor
        local mem_inp_t = self.slicedMemInp[t]

        self.attScore3Dims[{{}, {(t-1) * Mx + 1, t * Mx}, {}}]:bmm(mem_inp_t, htt)
    end


    self.attScore2Dims = self.attScore3Dims:view(-1, Mx)  -- (N * T, Mx)

    local attProb2Dims = self.softmax(self.attScore2Dims)

    self.attProb3Dims = attProb2Dims:view(N, T, Mx)

    self.output:resize(N, T, D)

    for t = 1, T do
        self.output[{{},{t},{}}]:bmm(self.attProb3Dims[{{},{t},{}}], self.slicedMemOut[t])
    end

    return self.output
end

function layer:backward(input, gradOutput)
    local x, h = unpack(input)
    local N, T, D = h:size(1), h:size(2), h:size(3)
    local Mx = self.memorySize

    local grad_h = self.gradInput[2]
    grad_h:resizeAs(h):zero()

    -- First, we use pos_mat to store grad mem_out
    local pos_mat, time_b = unpack(self.cache)
    local gradMemOut = pos_mat
    local slicedGradMemOut = self:__slice(gradMemOut)


    self.gradAttProb:resize(N, T, Mx)  -- same size as attProb3Dims
    self.attProb3Dims = self.attProb3Dims:transpose(2, 3)  -- (N, Mx, T)

    for t = 1, T do
        local attn_t = self.attProb3Dims[{{}, {}, {t}}]  -- (N, Mx, 1)
        local dout_t = gradOutput[{{}, {t}, {}}]  -- (N, 1, D)
        slicedGradMemOut[t]:bmm(attn_t, dout_t)  -- (N, Mx, D)
        self.gradAttProb[{{}, {t}, {}}]:bmm(dout_t, self.slicedMemOut[t]:transpose(2, 3))  -- (N, 1, Mx)
    end

    -- back propagate memOut

    self.memOut:backward(x, gradMemOut)

    -- keep working
    self.gradAttProb = self.gradAttProb:view(-1, Mx)  -- (N * T, Mx)

    local gradAttScore = self.softmax:backward(self.attScore2Dims, self.gradAttProb)

    gradAttScore = gradAttScore:view(N, -1 , 1)  -- (N, T * Mx, 1)

    -- we are done with memory output, reuse gradMemOut
    local gradMemInp = pos_mat

    local slicedGradMemInp = self:__slice(gradMemInp)

    for t = 1, T do
        local ht = h[{{}, {t}, {}}] -- (N, 1, D)
        -- get out grad of the unnormalized attention
        local grad_attn_t = gradAttScore[{{},{(t-1) * Mx + 1, t * Mx}, {}}]  -- (N, Mx, 1)
        grad_h[{{}, {t}, {}}]:bmm(grad_attn_t:transpose(2, 3), self.slicedMemInp[t])
        slicedGradMemInp[t]:bmm(grad_attn_t, ht)
    end

    -- sweet, we can backpropagate memInp
    self.memInp:backward(x, gradMemInp)
    -- now we can backpropage position matrix
    self.posMat:backward(time_b, gradMemInp)

    return self.gradInput
end

function layer:clearState()
    nn.utils.clear(self, {
        'output',
        'gradInput',
        'ht',
        'attScore3Dims',
        'attProb3Dims',
        'gradAttProb',
        'attScore2Dims'
    })
end
