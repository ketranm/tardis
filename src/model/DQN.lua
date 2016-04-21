--[[ Deep-Q Network
This is experimental code! DO NOT USE

This is an implementation of Deep Q learning described in
http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf

Author: Ke Tran <m.k.tran@uva.nl>

--]]
local DQN, parent = torch.class('nn.DQN', 'nn.Module')

function DQN:__init(kwargs)
    self.stateSize = kwargs.stateSize
    self.actions = self.actions
    self.numActions = self.numActions

    -- epsilon annealing
    self.ep_start = kwargs.ep or 1
    self.ep = self.ep_start -- exploration probability

    -- learning rate annealing
    self.lr_start = kwargs.learingRate or 0.01
    self.lr = self.lr_start

    self.replayMemory = kwargs.replayMemory or 100000
    -- TODO: add some more info

    -- Q-learning parameters
    self.discount = kwargs.discount or 0.9  -- discount factor
    self.numReplay = kwargs.numRelay or 1
    self.minReward = kwargs.minReward
    self.maxReward = kwargs,maxReward

    self.network = kwargs.network or self:createNetwork()

    -- other important shit
    self.numSteps = 0 -- number of perceived states
    self.lastState = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average
    self.tderr_avg = 0 -- TD error running average

    self.q_max = 1
    self.r_max = 1
end


function DQN:getQupdate(kwargs)
    --[[ Get out tuples of (state, action, reward, nextState, topActions) from replayMemory
    We need to compute the max Q value over the topActions (this is due to large space of actions).
    Also we need to use a separate network (called target_q) to compute the reward, this will make learning stable

    --]]

    -- TODO: set up some data structure
    -- topActions can be represented as a matrix, should use lookup table for this
    -- use the lookup table from the decoder or the projection matrix to the output layer

end


function DQN:createNetwork(inputSize, hiddenSize, numTopAction)
    local network = nn.Squential()
    network:add(nn.Linear(inputSize, hiddenSize))
    network:add(nn.ReLU())

    -- maybe write it more explicit
    local pt = nn.ParallelTable()
    pt:add(nn.Identity())
    pt:add(nn.Identity())
    local qnetwork = nn.Sequential()
    qnetwork:add(pt)
    qnetwork:add() -- need to multiply two tensors
end

