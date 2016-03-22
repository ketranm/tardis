--[[
Transducer: Stacking multiple RNNs
--]]
require 'torch'
require 'nn'
require 'model.LSTM'
require 'model.GRU'

local Transducer, parent = torch.class('nn.Transducer', 'nn.Module')


function Transducer:__init(kwargs)
    self.vocab_size = kwargs.vocab_size
    self.embedding_size = kwargs.embedding_size
    self.hidden_size = kwargs.hidden_size
    self.num_layers = kwargs.num_layers
    self.dropout = kwargs.dropout or 0
    self.batch_norm = kwargs.batch_norm
    self.rnn = kwargs.rnn or 'lstm'
    -- build transducer
    self.transducer = nn.Sequential()
    self.transducer:add(nn.LookupTable(self.vocab_size, self.embedding_size))
    self.rnns = {}
    -- for batch normalization
    self.bn_view_in = {}
    self.bn_view_out = {}
    self.output = torch.Tensor()

    for i = 1, self.num_layers do
        local prev_dim = self.hidden_size
        if i == 1 then prev_dim = self.embedding_size end
        local rnn
        if self.rnn == 'lstm' then
            rnn = nn.LSTM(prev_dim, self.hidden_size)
        elseif self.rnn == 'gru' then
            rnn = nn.GRU(prev_dim, self.hidden_size)
        else
            error("only support LSTM or GRU!")
        end
        self.transducer:add(rnn)
        table.insert(self.rnns, rnn)
        if self.batch_norm then
            local view_in = nn.View(1, 1, -1):setNumInputDims(3)
            table.insert(self.bn_view_in, view_in)
            self.transducer:add(view_in)

            local view_out = nn.View(1, -1):setNumInputDims(2)
            table.insert(self.bn_view_out, view_out)
            self.transducer:add(view_out)
        end
        if self.dropout > 0 then
            self.transducer:add(nn.Dropout(self.dropout))
        end
    end
end


function Transducer:updateOutput(input)
    local batch_size, length = input:size(1), input:size(2)

    for _,view_in in ipairs(self.bn_view_in) do
        view_in:resetSize(batch_size * length, -1)
    end
    for _,view_out in ipairs(self.bn_view_out) do
        view_out:resetSize(batch_size, length, -1)
    end
    return self.transducer:forward(input)
end


function Transducer:backward(input, gradOutput, scale)
    return self.transducer:backward(input, gradOutput, scale)
end


function Transducer:parameters()
    return self.transducer:parameters()
end


function Transducer:last_state()
    local state = {}
    for _,rnn in ipairs(self.rnns) do
        table.insert(state, rnn:last_state())
    end
    return state
end


function Transducer:training()
    self.transducer:training()
    parent.training(self)
end


function Transducer:evaluate()
    self.transducer:evaluate()
    parent.evaluate(self)
end


function Transducer:init_state(state)
    assert(#state == #self.rnns)
    for i, state in ipairs(state) do
        self.rnns[i]:init_state(state)
    end
end


function Transducer:get_grad_state()
    local grad_state = {}
    for _, rnn in ipairs(self.rnns) do
        table.insert(grad_state, rnn:get_grad_state())
    end
    return grad_state
end


function Transducer:set_grad_state(grad_state)
    for i, grad in ipairs(grad_state) do
        self.rnns[i]:set_grad_state(grad)
    end
end


function Transducer:updateGradInput(input, gradOutput)
    self:backward(input, gradOutput, 0)
end

function Transducer:accGradParameters(input, gradOutput, scale)
    self:backward(input, gradOutput, scale)
end

function Transducer:clearState()
    for _, rnn in ipairs(self.rnns) do
        rnn:clearState()
    end
end
