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

local kwargs = {vocabSize = 100000,
                embeddingSize = 4,
                hiddenSize = 4,
                numLayer = 3,
                dropout = 0,
                rnn = 'lstm'}

function tests.testForward()
    -- generate example

    local N = torch.random(20, 200)
    local T = torch.random(10, 50)
    local x = torch.range(1, N * T):reshape(N, T)


    local transducer = nn.Transducer(kwargs)

    -- test number of parameter
    local D, H = kwargs.embeddingSize, kwargs.hiddenSize

    local num_params 
    if kwargs.rnn == 'lstm' then
        num_params = (H + D + 1) * 4 * H + (kwargs.numLayer - 1) * (2 * H + 1) * (4 * H) + kwargs.vocabSize * D
    elseif kwargs.rnn == 'gru' then
        num_params = (H + D + 1) * 3 * H + (kwargs.numLayer - 1) * (2 * H + 1) * (3 * H) + kwargs.vocabSize * D
    else
        error('only support LSTM or GRU!')
    end

    local params, grad_params = transducer:getParameters()
    tester:assert(params:nElement() == num_params)
    tester:assert(grad_params:nElement() == num_params)

    local lt = nn.LookupTable(kwargs.vocabSize, kwargs.embeddingSize)
    lt.weight:copy(transducer.transducer:get(1).weight)
    local rnns = {}

    local initState = {}

    for i = 1, kwargs.numLayer do
        local c0 = torch.randn(N, kwargs.hiddenSize)
        local h0 = torch.randn(N, kwargs.hiddenSize)
        initState[i] = {c0, h0}
    end

    for i = 1, kwargs.numLayer do
        local prev_h = kwargs.hiddenSize
        if i == 1 then prev_h = kwargs.embeddingSize end
        local rnn
        if kwargs.rnn == 'lstm' then
            rnn = nn.LSTM(prev_h, kwargs.hiddenSize)
        elseif kwargs.rnn == 'gru' then
            rnn = nn.GRU(prev_h, kwargs.hiddenSize)
        else
            error("only support LSTM or GRU!")
        end
        rnn.weight:copy(transducer.rnns[i].weight)  -- reset weight
        rnn.bias:copy(transducer.rnns[i].bias)
        rnn:initState(initState[i])
        table.insert(rnns, rnn)
    end
    -- set state of transducer
    transducer:initState(initState)

    local wemb = lt:forward(x)  -- word embeddings
    local h = wemb
    local hx = {[0] = h}
    for i = 1, kwargs.numLayer do
        local h_next = rnns[i]:forward(h)
        h = h_next
        hx[i] = h
    end

    local h_trans = transducer:forward(x)
    tester:assertTensorEq(h, h_trans, 1e-10)

    -- test backward
    local grad = torch.Tensor():resizeAs(h_trans):uniform(0,1):mul(0.1)
    transducer:backward(x, grad)
    local prev_grad
    for i = kwargs.numLayer, 1, -1 do
        if i == kwargs.numLayer then 
            prev_grad = grad
        end
        local grad_i = rnns[i]:backward(hx[i-1], prev_grad)
        prev_grad = grad_i
    end
    lt:backward(x, prev_grad)

    tester:assertTensorEq(transducer.transducer:get(1).gradWeight, lt.gradWeight, 1e-10)

    for i = 1, kwargs.numLayer do
        tester:assertTensorEq(transducer.rnns[i].gradWeight, rnns[i].gradWeight, 1e-10)
        tester:assertTensorEq(transducer.rnns[i].gradBias, rnns[i].gradBias, 1e-10)
    end
end


function tests.gradcheck()
    -- generate example

    local N = 2
    local T = 3
    local x = torch.range(1, N * T):reshape(N, T)


    local transducer = nn.Sequential()
    transducer:add(nn.LookupTable(kwargs.vocabSize, kwargs.embeddingSize))
    transducer:add(nn.LSTM(kwargs.embeddingSize, kwargs.hiddenSize))
    transducer:add(nn.LSTM(kwargs.hiddenSize, kwargs.hiddenSize))
    transducer:add(nn.LSTM(kwargs.hiddenSize, kwargs.hiddenSize))
    

    local initState = {}
    --[[
    for i = 1, kwargs.numLayer do
        local c0 = torch.randn(N, kwargs.hiddenSize)
        local h0 = torch.randn(N, kwargs.hiddenSize)
        initState[i] = {c0, h0}
    end
    ]]
    local c0 = torch.randn(N, kwargs.hiddenSize)
    local h0 = torch.randn(N, kwargs.hiddenSize)
    for i = 1, 3 do
        transducer:get(i + 1):initState({c0, h0})
    end
    local h = transducer:forward(x)
    local grad = torch.randn(#h)
    transducer:backward(x, grad)

    local function fh0(h0)
        transducer:get(3):initState({c0,h0})
        return transducer:forward(x)
    end

    local dc0, dh0 = unpack(transducer:get(3):getGradState())
    local dh0_num = gradcheck.numeric_gradient(fh0, h0, grad, 1e-12)

    local dh0_error = gradcheck.relative_error(dh0_num, dh0)
    tester:assertle(dh0_error, 1e-2)

end

function tests.gradcheck2()
    -- generate example

    local N = 2
    local T = 3
    local x = torch.range(1, N * T):reshape(N, T)


    local transducer = nn.Transducer(kwargs)
    
    local state0 = {}
    
    for i = 1, kwargs.numLayer do
        local c0, h0
        c0 = torch.randn(N, kwargs.hiddenSize)
        h0 = torch.randn(N, kwargs.hiddenSize)
        state0[i] = {c0, h0}
    end

    transducer:initState(state0)


    local h = transducer:forward(x)
    local grad = torch.randn(#h)
    transducer:backward(x, grad)

    local function fh0(h0)
        state0[3][2] = h0
        transducer:initState(state0)
        return transducer:forward(x)
    end

    local function fc0(c0)
        state0[3][1] = c0
        transducer:initState(state0)
        return transducer:forward(x)
    end

    local c0, h0 = unpack(state0[3])
    local grad_state = transducer:getGradState()
    local dc0, dh0 = unpack(grad_state[3])

    local dh0_num = gradcheck.numeric_gradient(fh0, h0, grad, 1e-12)
    local dc0_num = gradcheck.numeric_gradient(fc0, c0, grad, 1e-12)

    local dh0_error = gradcheck.relative_error(dh0_num, dh0)
    local dc0_error = gradcheck.relative_error(dc0_num, dc0)

    tester:assertle(dh0_error, 1e-2, "gradcheck2")
    tester:assertle(dc0_error, 1e-2, "gradcheck2")
end

tester:add(tests)
tester:run()
