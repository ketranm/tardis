require 'torch'
require 'nn'
require 'util.DataLoader'
require 'model.NMTA' -- use attention by default
require 'tardis.PolicySearch'
require 'model.DQN'
require 'optim'
require 'xlua'

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

print('Creating model...')
local model = nn.NMT(kwargs)

if kwargs.gpuid >= 0 then
    model:cuda()
end

--local loader = DataLoader(kwargs)
print('loading parameters: ' .. kwargs.modelFile)
model:load(kwargs.modelFile)

-- TODO: maybe creating the Q-network inside DQN.lua?

local function createNetwork(stateSize, hiddenSize, numActions)
    --[[ Creating Q network
    Parameters:
    - `stateSize` : integer, size of the state of the NMT, it's customizable
    - `hiddenSize` : integer, for linear transformation
    - `numActions` : integer, number of Actions we consider at a time (e.g set equal to beamSize)

    Returns:
    - `qnetwork` : network for estimating q value
    --]]

    local state_transformer = nn.Sequential()
    state_transformer:add(nn.Linear(stateSize, hiddenSize))
    state_transformer:add(nn.ReLU())

    local fixActionEmb = nn.LookupTable(kwargs.trgVocabSize, kwargs.hiddenSize)
    -- this is not cool! Shouldn't do direct access. If this shit works, TODO: write a better code
    fixActionEmb.weight:copy(model.layer:get(6).weight)
    local action_transformer = nn.Sequential()
    action_transformer:add(fixActionEmb) -- shit, I forgot that we update the action embeddings as well
    action_transformer:add(nn.Linear(kwargs.hiddenSize, hiddenSize))

    local pt = nn.ParallelTable()
    pt:add(state_transformer)
    pt:add(action_transformer)

    local q_network = nn.Sequential()
    q_network:add(pt)
    q_network:add(nn.MM(false, true))

    return q_network
end

print('creating q network')
local q_network = createNetwork(kwargs.hiddenSize, 500, 5)
if kwargs.gpuid >= 0 then
    q_network:cuda()
end

local target_q_network = q_network:clone()

local dqn_config = {
    stateSize = kwargs.hiddenSize,
    numActions = kwargs.beamSize,
    hiddenSize = 500,
    discount = 0.9,
    network = q_network 
}

local dqn = nn.DQN(dqn_config)

-- overwrite some of the option
kwargs.sample = true -- use sample for now
kwargs.srcVocab, kwargs.trgVocab = unpack(torch.load(kwargs.vocabFile))
local ps = PolicySearch(kwargs)
ps:use(model)

local refFile, refLine = nil, nil
-- hard code for now
local memory = {}


local w, dw = q_network:getParameters()
local mse = nn.MSECriterion():cuda()
local optim_config = {epsilon = 1e-5, learningRate = 1e-2}
local optim_state = {}

function updateQ(args)
    local s, a, s2, a2, r = unpack(args)
    local q2_max = target_q_network:forward{s2, a2}:max(2)
    local q2 = q2_max:mul(0.9)
    delta = torch.CudaTensor({r}):view(1, 1)
    delta:add(q2)
    local feval = function(x)
        if x ~= w then w:copy(x) end
        local q = q_network:forward{s, a:view(-1)}
        delta:add(-1, q)
        f = delta[1]*delta[1]/2 -- MSE
        delta:clamp(-1, 1) -- make it stable
        delta:mul(-1)
        dw:zero()
        q_network:backward({s, a:view(-1)}, delta)
        return f, dw
    end
    --local _, fx = optim.sgd(feval, w, optim_config, optim_state)
    local _, fx = optim.adagrad(feval, w, optim_config, optim_state)
    return fx[1]
end

for epoch = 1, 200 do
    if kwargs.refFile then refFile = io.open(kwargs.refFile, 'r') end
    local num_sent = 0
    for line in io.lines(kwargs.textFile) do
        if refFile then refLine = refFile:read() end
        local state = ps:search(line, refLine)
        for _, sx in pairs(state) do
            table.insert(memory, sx)
        end
        num_sent = num_sent + 1
        if #memory > 4000 then
            local loss = 0
            local shuffle = torch.randperm(#memory)
            for i = 1, #memory do
                local id = shuffle[i]
                loss = loss +  updateQ(memory[id])
                if i % 200 == 0 then
                    target_q_network = q_network:clone()
                    xlua.progress(i, #memory)
                    print(string.format('epoch %d #sent %d \t loss: %.7f', epoch, num_sent, loss/i))
                end
            end
            memory = {}
            collectgarbage()
        end
    end
    print('save q network')
    torch.save('qnetwork2.t7', q_network)
end
