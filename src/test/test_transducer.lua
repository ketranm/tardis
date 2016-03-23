require 'nn'
require 'model.Transducer'


local gradcheck = require 'util.gradcheck'
local tests = {}
local tester = torch.Tester()


local function check_size(x, dims)
    tester:assert(x:dim() == #dims)
        for i, d in ipairs(dims) do
            tester:assert(x:size(i) == d)
        end
end

local kwargs = {vocab_size = 100000,
                embedding_size = 10,
                hidden_size = 10,
                num_layers = 4,
                dropout = 0,
                rnn = 'gru',
                batch_norm = false}

function tests.testForward()
    -- generate example

    local N = torch.random(20, 200)
    local T = torch.random(10, 50)
    local x = torch.range(1, N * T):reshape(N, T)


    local trans = nn.Transducer(kwargs)

    -- test number of parameter
    local D, H = kwargs.embedding_size, kwargs.hidden_size

    local num_params 
    if kwargs.rnn == 'lstm' then
        num_params = (H + D + 1) * 4 * H + (kwargs.num_layers - 1) * (2 * H + 1) * (4 * H) + kwargs.vocab_size * D
    elseif kwargs.rnn == 'gru' then
        num_params = (H + D + 1) * 3 * H + (kwargs.num_layers - 1) * (2 * H + 1) * (3 * H) + kwargs.vocab_size * D
    else
        error('only support LSTM or GRU!')
    end

    local params, grad_params = trans:getParameters()
    tester:assert(params:nElement() == num_params)
    tester:assert(grad_params:nElement() == num_params)

    local lt = nn.LookupTable(kwargs.vocab_size, kwargs.embedding_size)
    lt.weight:copy(trans.transducer:get(1).weight)
    local rnns = {}

    local init_state = {}

    for i = 1, kwargs.num_layers do
        local c0 = torch.randn(N, kwargs.hidden_size)
        local h0 = torch.randn(N, kwargs.hidden_size)
        init_state[i] = {c0, h0}
    end

    for i = 1, kwargs.num_layers do
        local prev_h = kwargs.hidden_size
        if i == 1 then prev_h = kwargs.embedding_size end
        local rnn
        if kwargs.rnn == 'lstm' then
            rnn = nn.LSTM(prev_h, kwargs.hidden_size)
        elseif kwargs.rnn == 'gru' then
            rnn = nn.GRU(prev_h, kwargs.hidden_size)
        else
            error("only support LSTM or GRU!")
        end
        rnn.weight:copy(trans.rnns[i].weight)  -- reset weight
        rnn.bias:copy(trans.rnns[i].bias)
        rnn:init_state(init_state[i])
        table.insert(rnns, rnn)
    end
    -- set state of transducer
    trans:init_state(init_state)

    local wemb = lt:forward(x)  -- word embeddings
    local h = wemb
    for i = 1, kwargs.num_layers do
        local h_next = rnns[i]:forward(h)
        h = h_next
    end

    local h_trans = trans:forward(x)
    tester:assertTensorEq(h, h_trans, 1e-10)
end


tester:add(tests)
tester:run()
