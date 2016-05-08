--[[ Recurrent Memory Networks Translation
--]]

require 'model.Transducer'
require 'model.GlimpseDot'
require 'model.Memory' -- memory block


local model_utils = require 'model.model_utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')


function NMT:__init(kwargs)
    -- over write option
    kwargs.vocabSize = kwargs.srcVocabSize
    self.encoder = nn.Transducer(kwargs)

    -- over write option
    kwargs.vocabSize = kwargs.trgVocabSize
    self.offset = kwargs.trgVocabSize
    self.decoder = nn.Transducer(kwargs)

    self.glimpse = nn.GlimpseDot(kwargs.hiddenSize)
    local totalVocabSize = kwargs.trgVocabSize + kwargs.srcVocabSize
    self.memInp = nn.LookupTable(totalVocabSize, kwargs.hiddenSize)
    self.memOut = nn.LookupTable(totalVocabSize, kwargs.hiddenSize)
    local memorySize = kwargs.memorySize or 5
    self.memorySize = memorySize
    self.posMat = nn.LookupTable(memorySize, kwargs.hiddenSize)

    self.MB = nn.Memory(memorySize, self.memInp, self.memOut, self.posMat)

    --
    self.layer = nn.Sequential()
    -- joining inputs, can be coded more efficient
    local pt = nn.ParallelTable()
    pt:add(nn.Identity())
    pt:add(nn.Identity())
    pt:add(nn.Identity())

    self.layer:add(pt)
    self.layer:add(nn.JoinTable(3))
    self.layer:add(nn.View(-1, 3 * kwargs.hiddenSize))
    self.layer:add(nn.Linear(3 * kwargs.hiddenSize, kwargs.hiddenSize, false))
    self.layer:add(nn.ELU(1, true))
    self.layer:add(nn.Linear(kwargs.hiddenSize, kwargs.trgVocabSize, true))
    self.layer:add(nn.LogSoftMax())

    self.criterion = nn.ClassNLLCriterion()

    self.params, self.gradParams = model_utils.combine_all_parameters(self.encoder, 
        self.decoder, self.glimpse, 
        self.layer, self.memInp, 
        self.memOut, self.posMat)

    self.maxNorm = kwargs.maxNorm or 5

    -- use buffer to store all the information needed for forward/backward
    self.buffers = {}
    self.memseq = torch.Tensor()
    self.mem = torch.Tensor()
end


function NMT:forward(input, target)
    --[[ Forward pass of NMT

    Parameters:
    - `input` : table of source and target tensor
    - `target` : a tensor of next words

    Return:
    - `logProb` : negative log-likelihood of the minibatch
    --]]
    local Tx = input[1]:size(2)
    local Ty = input[2]:size(2)
    local N = input[2]:size(1)
    local memseq = self.memseq
    local Mx = self.memorySize
    memseq:resize(N, Ty + Mx - 1):fill(1)
    memseq[{{}, {Mx, -1}}]:copy(input[2])
    local M = Mx - 1
    --print(input[1])
    if Tx > M then
        memseq[{{}, {1, M}}] = input[1][{{},{Tx-M+1, -1}}]:clone():add(self.offset)
    else
        --print(input[1], Tx, M-Tx+1, M)
        --print(memseq[{{}, {M-Tx+1, M}}], input[1])
        memseq[{{}, {M-Tx+1, M}}] = input[1]:clone():add(self.offset)
    end
    -- maybe it's faster to use TemporalConvolution with Identity Kernel
    self.mem:resize(N, Ty*Mx)
    for i = 1, Ty do
        local lo = (i-1) * Mx + 1
        local hi = i * Mx
        self.mem[{{}, {lo, hi}}] = memseq[{{},{i, i + Mx - 1}}]
    end

    --print(self.mem:size(), N, Tx, Ty)
    self:stepEncoder(input[1])
    local logProb = self:stepDecoder(input[2], self.mem)
    return self.criterion:forward(logProb, target)
end


function NMT:backward(input, target)
    -- zero grad manually here
    self.gradParams:zero()

    -- make it more explicit

    local buffers = self.buffers
    local outputEncoder = buffers.outputEncoder
    local outputDecoder = buffers.outputDecoder
    local context = buffers.context
    local logProb = buffers.logProb
    local memory = buffers.memory

    -- all good. Ready to backprop

    local gradLoss = self.criterion:backward(logProb, target)
    local gradLayer = self.layer:backward({context, outputDecoder, memory}, gradLoss)
    local gradDecoder = gradLayer[2] -- grad to decoder
    local gradMem = self.MB:backward({self.mem, outputDecoder}, gradLayer[3])
    local gradGlimpse = self.glimpse:backward({outputEncoder, outputDecoder}, gradLayer[1])

    gradDecoder:add(gradGlimpse[2]) -- accummulate gradient in-place 
    gradDecoder:add(gradMem[2])

    self.decoder:backward(input[2], gradDecoder)
    -- init gradient from decoder
    self.encoder:setGradState(self.decoder:getGradState())
    -- backward to encoder
    local gradEncoder = gradGlimpse[1]
    self.encoder:backward(input[1], gradEncoder)
end


function NMT:update(learningRate)
    local gradNorm = self.gradParams:norm()
    local scale = learningRate
    if gradNorm > self.maxNorm then
        scale = scale*self.maxNorm /gradNorm
    end
    self.params:add(self.gradParams:mul(-scale)) -- do it in-place
end


function NMT:parameters()
    return self.params
end

function NMT:training()
    self.encoder:training()
    self.decoder:training()
end

function NMT:evaluate()
    self.encoder:evaluate()
    self.decoder:evaluate()
end


function NMT:load(fileName)
    local params = torch.load(fileName)
    self.params:copy(params)
end

function NMT:save(fileName)
    torch.save(fileName, self.params)
end

-- useful interface for beam search

function NMT:stepEncoder(x)
    --[[ Encode the source sequence
    All the information produced by the encoder is stored in buffers
    Parameters:
    - `x` : source tensor, can be a matrix (batch)
    --]]

    local outputEncoder = self.encoder:updateOutput(x)
    local prevState = self.encoder:lastState()
    self.buffers = {outputEncoder = outputEncoder, prevState = prevState}
end

function NMT:stepDecoder(x, mem)
    --[[ Run the decoder
    If it is called for the first time, the decoder will be initialized
    from the last state of the encoder. Otherwise, it will continue from
    its last state. This is useful for beam search or reinforce training

    Parameters:
    - `x` : target sequence, can be a matrix (batch)

    Return:
    - `logProb` : cross entropy loss of the sequence
    --]]

    -- get out necessary information from the buffers
    local buffers = self.buffers
    local outputEncoder, prevState = buffers.outputEncoder, buffers.prevState

    self.decoder:initState(prevState)
    local outputDecoder = self.decoder:updateOutput(x)
    local memory = self.MB:forward{mem, outputDecoder}
    local context = self.glimpse:forward({outputEncoder, outputDecoder})
    local logProb = self.layer:forward({context, outputDecoder, memory})

    -- update buffer, adding information needed for backward pass
    buffers.outputDecoder = outputDecoder
    buffers.prevState = self.decoder:lastState()
    buffers.context = context
    buffers.logProb = logProb
    buffers.memory = memory

    return logProb
end

function NMT:indexDecoderState(index)
    --[[ This method is useful for beam search.
    It is similar to torch.index function, return a new state of kept index
    
    Parameters:
    - `index` : torch.LongTensor object

    Return:
    - `state` : new hidden state of the decoder, indexed by the argument
    --]]

    local currState = self.decoder:lastState()
    local newState = {}
    for _, state in ipairs(currState) do
        local sk = {}
        for _, s in ipairs(state) do
            table.insert(sk, s:index(1, index))
        end
        table.insert(newState, sk)
    end

    -- here, it make sense to update the buffer as well
    local buffers = self.buffers
    buffers.prevState = newState
    buffers.outputEncoder = buffers.outputEncoder:index(1, index)

    return newState
end

function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end
