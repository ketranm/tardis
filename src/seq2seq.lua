require 'torch'
require 'nn'


local Seq2seq
require 'util.DataLoader'

--torch.manualSeed(42)
local configuration = require 'pl.config'
local kwargs = configuration.read(arg[1])

if kwargs.gpuid >= 0 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. kwargs.gpuid .. '...')
    cutorch.setDevice(kwargs.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    --cutorch.manualSeed(kwargs.seed or 42)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    kwargs.gpuid = -1 -- overwrite user setting
  end
end

print('Experiment Setting:',kwargs)

local loader = DataLoader(kwargs)
if kwargs.attention then
    Seq2seq = require 'model.Seq2seqAtt'
else
    Seq2seq = require 'model.Seq2seq'
end

local model = Seq2seq.create(kwargs)

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
    local prev_valid_nll = math.huge
    for epoch = 1,kwargs.max_epoch do
        loader:read("train")
        local nll = 0
        local num_batches = loader:num_batches()
        print('number of batches: ', num_batches)
        for i = 1, num_batches do
            local s, t, tn = prepro(loader:next_batch())
            local cur_nll = model:forward({s,t}, tn:view(-1))

            if cur_nll ~= cur_nll then
                print('shit happen: ', cur_nll)
                --print(model.params)
                for j = 1, model.params:nElement() do
                    x = model.params[i]
                    if x ~= x then
                        print("found the bitch!")
                    end
                end
            else
                nll = nll + cur_nll
            end

            model:backward({s,t},tn:view(-1))
            model:update(kwargs.learning_rate)
            --model:clearState()
            if i % 20 == 0 then
                if not model:paramcheck() then
                    print('fucked up parameters')
                end
                xlua.progress(i, num_batches)
                print(string.format('epoch %d\t train perplexity = %.4f', epoch, exp(nll/i)))
                collectgarbage()
            end
        end

        if prev_valid_nll > nll then
            kwargs.learning_rate = kwargs.learning_rate * 0.6
        end
        prev_valid_nll = nll

        --[[
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
        --]]

        local save_file = string.format("%s/tardis_epoch_%d_%.4f.t7", kwargs.checkpoint_dir, epoch, nll/num_batches)
        paths.mkdir(paths.dirname(save_file))
        print('save model to: ' .. save_file)
        print('learning_rate: ', kwargs.learning_rate)
        model:save_model(save_file)

    end
end

train()