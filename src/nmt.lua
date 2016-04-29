require 'torch'
require 'nn'

-- make sure this script can be run from any folder
package.path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'util.DataLoader'
require 'model.NMTA' -- use attention by default
require 'tardis.BeamSearch'


local timer = torch.Timer()
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
io:flush()

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

        if epoch > kwargs.decayAfter then
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
    kwargs.transFile =  kwargs.transFile or 'translation.txt'

    local startTime = timer:time().real
    print('loading model...')
    model:load(kwargs.modelFile)
    local loadTime = timer:time().real - startTime
    print(string.format('done, loading time: %.4f sec', loadTime))
    timer:reset()

    local file = io.open(kwargs.transFile, 'w')
    local nbestFile = io.open(kwargs.transFile .. '.nbest', 'w')
    -- if reference is provided compute BLEU score of each n-best
    local refFile
    if kwargs.refFile then
        refFile = io.open(kwargs.refFile, 'r')  
    end

    -- create beam search object
    kwargs.srcVocab, kwargs.trgVocab = unpack(loader.vocab)
    local bs = BeamSearch(kwargs)
    bs:use(model)

    local refLine
    local nbLines = 0 
    for line in io.lines(kwargs.textFile) do
        nbLines = nbLines + 1
        if refFile then refLine = refFile:read() end
        local translation, nbestList = bs:search(line, kwargs.maxTrgLength, refLine)
        file:write(translation .. '\n')
        file:flush()
        if nbestList then
            nbestFile:write('SENTID=' .. nbLines .. '\n')
            nbestFile:write(table.concat(nbestList, '\n') .. '\n')
            nbestFile:flush()
        end
    end
    file:close()
    nbestFile:close()

    local transTime = timer:time().real
    print(string.format('Done (%d) sentences translated', nbLines))
    print(string.format('Total time: %.4f sec', transTime))
    print(string.format('Time per sentence: %.4f', transTime/nbLines))
end
