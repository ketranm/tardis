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
            local s, t, next_t = prepro(loader:next_batch())
            nll = nll + model:forward({s,t}, next_t:view(-1))
            model:backward({s,t},next_t:view(-1))
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
        model:evaluate()
        local valid_nll = 0
        local num_batches = loader:num_batches()
        for i = 1, num_batches do
            local s,t,next_t = prepro(loader:next_batch())
            valid_nll = valid_nll + model:forward({s,t}, next_t:view(-1))
            if i % 50 == 0 then collectgarbage() end
        end

        prev_valid_nll = valid_nll
        print(string.format('epoch %d\t valid perplexity = %.4f', epoch, exp(valid_nll/num_batches)))
        local checkpoint = string.format("%s/tardis_epoch_%d_%.4f.t7", kwargs.checkpoint_dir, epoch, valid_nll/num_batches)
        paths.mkdir(paths.dirname(checkpoint))
        print('save model to: ' .. checkpoint)
        print('learning_rate: ', kwargs.learning_rate)
        model:save_model(checkpoint)

    end
end

local eval = kwargs.model_file and kwargs.text_file

if not eval then
    -- training mode
    train()
else
    -- use dictionary
    model:use_vocab(loader.vocab)
    model:evaluate()
    model:load_model(kwargs.model_file)
    local file = io.open('translation.txt', 'w')
    io.output(file)
    for line in io.lines(kwargs.text_file) do
        local translation = model:translate(line, kwargs.beam_size)
        --print(translation)
        io.write(translation .. '\n')
        io.flush()
    end
    io.close(file)
end


