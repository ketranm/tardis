require 'torch'
require 'nn'
require 'util.DataLoader'
require 'model.NMTA' -- use attention by default
require 'tardis.PolicySearch'


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

print('loading parameters: ' .. kwargs.modelFile)
model:load(kwargs.modelFile)


