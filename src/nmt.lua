require 'torch'
require 'nn'


local Seq2seq
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
    local len_ys = target:size(2)
    -- make contiguous
    source = source:contiguous()
    prev_target = target:narrow(2,1,len_ys-1):contiguous()
    next_target = target:narrow(2,2,len_ys-1):contiguous()
    if kwargs.gpuid >= 0 then
        source = source:float():cuda()
        prev_target = prev_target:float():cuda()
        next_target = next_target:float():cuda()
    end
    return source, prev_target, next_target
end

function train()
    local exp = math.exp
    for epoch = 1,kwargs.max_epoch do
        loader:read("train")
        model:training()
        local nll = 0
        local num_batches = loader:num_batches()
        print('number of batches: ', num_batches)
        for i = 1, num_batches do
            local s, t, tn = prepro(loader:next_batch())
            nll = nll + model:forward({s,t}, tn:view(-1))
            model:backward({s,t},tn:view(-1))
            model:update(kwargs.learning_rate)
            --model:clearState()
            if i % kwargs.report_every == 0 then
                xlua.progress(i, num_batches)
                print(string.format('epoch %d\t train perplexity = %.4f', epoch, exp(nll/i)))
                collectgarbage()
            end
        end

        if epoch > kwargs.learning_rate_decay_after then
            kwargs.learning_rate = kwargs.learning_rate * kwargs.decay_rate
        end

        loader:read("valid")
        local valid_nll = 0
        local num_batches = loader:num_batches()
        for i = 1, num_batches do
            local s,t,tn = prepro(loader:next_batch())
            valid_nll = valid_nll + model:forward({s,t}, tn:view(-1))
            if i % 50 == 0 then collectgarbage() end
        end

        prev_valid_nll = valid_nll
        print(string.format('epoch %d\t valid perplexity = %.4f', epoch, exp(valid_nll/num_batches)))
        local save_file = string.format("%s/tardis_epoch_%d_%.4f.t7", kwargs.checkpoint_dir, epoch, valid_nll/num_batches)
        paths.mkdir(paths.dirname(save_file))
        print('save model to: ' .. save_file)
        print('learning_rate: ', kwargs.learning_rate)
        model:save_model(save_file)

    end
end

train()
