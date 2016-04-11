require 'torch'
require 'nn'

require 'util.DataLoader'

torch.manualSeed(42)
local configuration = require 'pl.config'
local kwargs = configuration.read(arg[1])

if kwargs.gpuid >= 0 then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(kwargs.gpuid + 1)
    cutorch.manualSeed(kwargs.seed or 42)
else
    kwargs.gpuid = -1
end

print('Experiment Setting: ', kwargs)

if kwargs.attention == 1 then
    require 'model.NMTA'
else
    require 'model.NMT'
end

local model = nn.NMT(kwargs)

if kwargs.gpuid >= 0 then
    model:cuda()
end

local loader = DataLoader(kwargs)

-- prepare data
function prepro(batch)
    local source, target = unpack(batch)
    local trgLength = target:size(2)
    -- make contiguous
    source = source:contiguous()
    prevTrg = target:narrow(2,1,trgLength-1):contiguous()
    nextTrg = target:narrow(2,2,trgLength-1):contiguous()
    if kwargs.gpuid >= 0 then
        source = source:float():cuda()
        prevTrg = prevTrg:float():cuda()
        nextTrg = nextTrg:float():cuda()
    end
    return source, prevTrg, nextTrg
end

function train()
    local exp = math.exp
    for epoch = 1,kwargs.maxEpoch do
        loader:read("train")
        model:training()
        local nll = 0
        local nBatch = loader:nBatch()
        print('number of batches: ', nBatch)
        for i = 1, nBatch do
            local src, trg, nextTrg = prepro(loader:nextBatch())
            nll = nll + model:forward({src, trg}, nextTrg:view(-1))
            model:backward({src, trg}, nextTrg:view(-1))
            model:update(kwargs.learningRate)
            --model:clearState()
            if i % kwargs.reportEvery == 0 then
                xlua.progress(i, nBatch)
                print(string.format('epoch %d\t train perplexity = %.4f', epoch, exp(nll/i)))
                collectgarbage()
            end
        end

        if epoch > kwargs.dacayAfter then
            kwargs.learningRate = kwargs.learningRate * kwargs.decayRate
        end

        loader:read("valid")
        model:evaluate()
        local valid_nll = 0
        local nBatch = loader:nBatch()
        for i = 1, nBatch do
            local src, trg, nextTrg = prepro(loader:nextBatch())
            valid_nll = valid_nll + model:forward({src, trg}, nextTrg:view(-1))
            if i % 50 == 0 then collectgarbage() end
        end

        prev_valid_nll = valid_nll
        print(string.format('epoch %d\t valid perplexity = %.4f', epoch, exp(valid_nll/nBatch)))
        local checkpoint = string.format("%s/tardis_epoch_%d_%.4f.t7", kwargs.modelDir, epoch, valid_nll/nBatch)
        paths.mkdir(paths.dirname(checkpoint))
        print('save model to: ' .. checkpoint)
        print('learningRate: ', kwargs.learningRate)
        model:save(checkpoint)

    end
end

local eval = kwargs.modelFile and kwargs.textFile

if not eval then
    -- training mode
    train()
else
    -- use dictionary
    model:use_vocab(loader.vocab)
    model:evaluate()
    model:load(kwargs.modelFile)
    local file = io.open('translation.txt', 'w')
    io.output(file)
    for line in io.lines(kwargs.textFile) do
        local translation = model:translate(line, kwargs.beamSize)
        --print(translation)
        io.write(translation .. '\n')
        io.flush()
    end
    io.close(file)
end


