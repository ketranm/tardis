-- Reinforcement training NMT

require 'torch'
require 'nn'


require 'util.DataLoader'
require 'model.NMTA' -- use attention by default
require 'tardis.FastBeamSearch'


local timer = torch.Timer()
torch.manualSeed(42)
local configuration = require 'pl.config'
local config = configuration.read(arg[1])

if config.gpuid >= 0 then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(config.gpuid + 1)
    cutorch.manualSeed(config.seed or 42)
else
    config.gpuid = -1
end

print('Experiment Setting: ', config)
io:flush()


local loader = DataLoader(config)

config.pad_idx = loader.pad_idx
config.eos_idx = loader.eos_idx
config.unk_idx = loader.unk_idx

local model = nn.NMT(config)
-- overwrite config

if config.gpuid >= 0 then
    model:cuda()
end



-- prepare data
function prepro(batch)
    local source, target = unpack(batch)
    local trgLength = target:size(2)
    --print('target\n',target)
    -- make contiguous
    source = source:contiguous()
    prevTrg = target:narrow(2,1,trgLength-1):contiguous()
    nextTrg = target:narrow(2,2,trgLength-1):contiguous()
    if config.gpuid >= 0 then
        source = source:float():cuda()
        prevTrg = prevTrg:float():cuda()
        nextTrg = nextTrg:float():cuda()
    else
        source = source:double()
        prevTrg = prevTrg:double()
        nextTrg = nextTrg:double()
    end
    return source, prevTrg, nextTrg
end


-- build cumulative reward predictor

function train()
    local exp = math.exp
    for epoch = 1,config.maxEpoch do
        loader:readTrain()
        model:training()
        model:initMixer(config)
        local nll = 0
        local reward = 0
        local crp_err = 0
        local nbatches = loader:nbatches()
        local n = 0
        --print('number of batches: ', nbatches)
        for i = 1, nbatches do
            local src, trg, nextTrg = prepro(loader:nextBatch())
            -- test rf here
            local trgLength = trg:size(2)
            local delta = 3
            local skips = trgLength - delta

            local batchSize = src:size(1)
            if skips > 1 then
                local res = model:trainMixer({src, trg}, nextTrg, skips, 1)
                local _nil, _rw, _crp_err = unpack(res)
                nll = nll + _nil
                reward = reward + _rw
                crp_err = crp_err + _crp_err
                n = n + 1
            end
            if n % config.reportEvery == 0 then
                local msg = string.format(
                                'epoch = %d xent = %.4f reward = %.7f  mse err = %.7f',
                                epoch, exp(nll/n), reward/n, crp_err/n)
                print(msg)
                xlua.progress(i, nbatches)
                collectgarbage()
            end
        end

        if epoch > config.decayAfter then
            config.learningRate = config.learningRate * config.decayRate
        end

        loader:readValid()
        model:evaluate()
        local valid_nll = 0
        local nbatches = loader:nbatches()
        for i = 1, nbatches do
            local src, trg, nextTrg = prepro(loader:nextBatch())
            valid_nll = valid_nll + model:forward({src, trg}, nextTrg:view(-1))
            if i % 50 == 0 then collectgarbage() end
        end

        prev_valid_nll = valid_nll
        print(string.format('epoch %d\t valid perplexity = %.4f', epoch, exp(valid_nll/nbatches)))
        local checkpoint = string.format("%s/tardis_epoch_%d_%.4f.t7", config.modelDir, epoch, valid_nll/nbatches)
        paths.mkdir(paths.dirname(checkpoint))
        print('save model to: ' .. checkpoint)
        print('learningRate: ', config.learningRate)
        model:save(checkpoint)

    end
end

local eval = config.modelFile and config.textFile

if not eval then
    -- training mode
    train()
else
    config.transFile =  config.transFile or 'translation.txt'

    local startTime = timer:time().real
    print('loading model...')
    model:load(config.modelFile)
    local loadTime = timer:time().real - startTime
    print(string.format('done, loading time: %.4f sec', loadTime))
    timer:reset()

    local file = io.open(config.transFile, 'w')
    local nbestFile = io.open(config.transFile .. '.nbest', 'w')
    -- if reference is provided compute BLEU score of each n-best
    local refFile
    if config.refFile then
        refFile = io.open(config.refFile, 'r')  
    end

    -- create beam search object
    config.srcVocab, config.trgVocab = unpack(loader.vocab)
    local bs = BeamSearch(config)
    bs:use(model)

    local refLine
    local nbLines = 0 
    for line in io.lines(config.textFile) do
        nbLines = nbLines + 1
        if refFile then refLine = refFile:read() end
        local translation, nbestList = bs:search(line, config.maxTrgLength, refLine)
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
