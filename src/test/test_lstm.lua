package.path = package.path .. ";../model/?.lua;../util/?.lua"

require 'nn'
require 'LSTM'

local gradcheck = require 'gradcheck'
local tests = {}
local tester = torch.Tester()


local function check_size(x, dims)
    tester:assert(x:dim() == #dims)
        for i, d in ipairs(dims) do
            tester:assert(x:size(i) == d)
        end
end


function tests.testForward()
    local N, T, D, H = 3, 4, 5, 6

    local h0 = torch.randn(N, H)
    local c0 = torch.randn(N, H)
    local x  = torch.randn(N, T, D)
    local lstm = nn.LSTM(D, H)
    lstm:initState({c0,h0})
    local params, grad_params = lstm:getParameters()

    local num_params = (H + D +1) * 4 * H
    tester:assert(params:nElement(), num_params)
    
    local h = lstm:forward(x)

    -- Do a naive forward pass
    local naive_h = torch.Tensor(N, T, H)
    local naive_c = torch.Tensor(N, T, H)

    -- Unpack weight, bias for each gate
    local Wxi = lstm.weight[{{1, D}, {1, H}}]
    local Wxf = lstm.weight[{{1, D}, {H + 1, 2 * H}}]
    local Wxo = lstm.weight[{{1, D}, {2 * H + 1, 3 * H}}]
    local Wxg = lstm.weight[{{1, D}, {3 * H + 1, 4 * H}}]

    local Whi = lstm.weight[{{D + 1, D + H}, {1, H}}]
    local Whf = lstm.weight[{{D + 1, D + H}, {H + 1, 2 * H}}]
    local Who = lstm.weight[{{D + 1, D + H}, {2 * H + 1, 3 * H}}]
    local Whg = lstm.weight[{{D + 1, D + H}, {3 * H + 1, 4 * H}}]

    local bi = lstm.bias[{{1, H}}]:view(1, H):expand(N, H)
    local bf = lstm.bias[{{H + 1, 2 * H}}]:view(1, H):expand(N, H)
    local bo = lstm.bias[{{2 * H + 1, 3 * H}}]:view(1, H):expand(N, H)
    local bg = lstm.bias[{{3 * H + 1, 4 * H}}]:view(1, H):expand(N, H)

    local prev_h, prev_c = h0:clone(), c0:clone()
    for t = 1, T do
        local xt = x[{{}, t}]
        local i = torch.sigmoid(torch.mm(xt, Wxi) + torch.mm(prev_h, Whi) + bi)
        local f = torch.sigmoid(torch.mm(xt, Wxf) + torch.mm(prev_h, Whf) + bf)
        local o = torch.sigmoid(torch.mm(xt, Wxo) + torch.mm(prev_h, Who) + bo)
        local g =    torch.tanh(torch.mm(xt, Wxg) + torch.mm(prev_h, Whg) + bg)
        local next_c = torch.cmul(prev_c, f) + torch.cmul(i, g)
        local next_h = torch.cmul(o, torch.tanh(next_c))
        naive_h[{{}, t}] = next_h
        naive_c[{{}, t}] = next_c
        prev_h, prev_c = next_h, next_c
    end

    tester:assertTensorEq(naive_h, h, 1e-10)
end

function tests.gradcheck()
    local N, T, D, H = 2, 3, 4, 5

    local x = torch.randn(N, T, D)
    local h0 = torch.randn(N, H)
    local c0 = torch.randn(N, H)

    local lstm = nn.LSTM(D, H)
    lstm:initState({c0,h0})
    local h = lstm:forward(x)

    local dh = torch.randn(#h)

    lstm:zeroGradParameters()

    local dx = lstm:backward(x, dh)
    local dc0, dh0 = unpack(lstm:getGradState())
    local dw = lstm.gradWeight:clone()
    local db = lstm.gradBias:clone()

    local function fx(x) return lstm:forward(x) end
    local function fh0(h0)
        lstm:initState({c0,h0})
        return lstm:forward(x)
    end

    local function fc0(c0)
        lstm:initState({c0,h0})
        return lstm:forward(x)
    end

    local function fw(w)
        local old_w = lstm.weight
        lstm.weight = w
        local out = lstm:forward(x)
        lstm.weight = old_w
        return out
    end

    local function fb(b)
        local old_b = lstm.bias
        lstm.bias = b
        local out = lstm:forward(x)
        lstm.bias = old_b
        return out
    end

    local dx_num = gradcheck.numeric_gradient(fx, x, dh)
    local dh0_num = gradcheck.numeric_gradient(fh0, h0, dh)
    local dc0_num = gradcheck.numeric_gradient(fc0, c0, dh)
    local dw_num = gradcheck.numeric_gradient(fw, lstm.weight, dh)
    local db_num = gradcheck.numeric_gradient(fb, lstm.bias, dh)

    local dx_error = gradcheck.relative_error(dx_num, dx)
    local dh0_error = gradcheck.relative_error(dh0_num, dh0)
    local dc0_error = gradcheck.relative_error(dc0_num, dc0)
    local dw_error = gradcheck.relative_error(dw_num, dw)
    local db_error = gradcheck.relative_error(db_num, db)
    tester:assertle(dh0_error, 1e-4)
    tester:assertle(dc0_error, 1e-5)
    tester:assertle(dx_error, 1e-5)
    tester:assertle(dw_error, 1e-4)
    tester:assertle(db_error, 1e-5)
end

function tests.carrystateTest()
    local N, T, D, H = 5, 6, 7, 8
    local lstm = nn.LSTM(D, H)

    local final_h, final_c = nil, nil
    local dstate = nil
    local grad_c0, grad_h0 = unpack(lstm:getGradState())
    local lr = 0.01
    for t = 1, 4 do
        local x = torch.randn(N, T, D)
        local dout = torch.randn(N, T, H)
        local out = lstm:forward(x)
        lstm:zeroGradParameters()
        -- test zero out gradients
        tester:assertTensorEq(lstm.gradWeight, torch.zeros(D + H, 4 *H), 0)
        tester:assertTensorEq(lstm.gradBias, torch.zeros(4 *H), 0)
        if t > 1 then
            lstm:setGradState(dstate)
            tester:assert(lstm._init_grad_state)
        end

        local din = lstm:backward(x, dout)
        lstm:updateParameters(lr)
        -- test gradient updated
        tester:assertTensorNe(lstm.gradWeight, torch.zeros(D + H, 4 *H), 0)
        tester:assertTensorNe(lstm.gradBias, torch.zeros(4 *H), 0)
        
        if t == 1 then
            tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
            tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
        elseif t > 1 then

            tester:assertTensorEq(lstm.c0, final_c, 0)
            tester:assertTensorEq(lstm.h0, final_h, 0)
        end
        final_c = lstm.cell[{{}, T}]:clone()
        final_h = out[{{}, T}]:clone()
        local state = lstm:lastState()
        lstm:initState(state)
        dstate = lstm:getGradState()
    end

    -- Initial state should reset to zero after we call resetstate
    lstm:resetState()
    local x = torch.randn(N, T, D)
    local dout = torch.randn(N, T, H)
    lstm:forward(x)
    lstm:backward(x, dout)
    tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
    tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
end

tester:add(tests)
tester:run()
