package.path = package.path .. ";../model/?.lua;../util/?.lua"

require 'nn'
require 'Glimpse'

local gradcheck = require 'gradcheck'
local tests = {}
local tester = torch.Tester()


function tests.testForward()
    local N, Tx, Ty, D = 2, 3, 4, 5
    local glimpse = nn.Glimpse(D)
    local params, grad_params = glimpse:getParameters()
    tester:assert(params:nElement() == grad_params:nElement())
    tester:assert(params:nElement() == D*D)
    -- manual check with torch.nn
    local MM1 = nn.MM(false, true)
    local MM2 = nn.MM(false, false)
    local softmax = nn.SoftMax()

    local linear = nn.Linear(D, D, false)
    linear.weight:copy(glimpse.weight:t())
    tester:assertTensorEq(linear.weight:t(), glimpse.weight, 1e-10)


    local gout
    for i = 1, 3 do
        local input = {torch.randn(N, Tx, D), torch.randn(N, Ty, D)}

        gout = glimpse:forward(input)

        local x, y = unpack(input)
        local y1 = linear:forward(y:view(Ty * N, -1))
        local score = MM1:forward{y1:view(N, Ty, -1), x}
        local patt = softmax:forward(score:view(-1, Tx))
        local c = MM2:forward{patt:view(N, Ty, Tx), x}

        tester:assertTensorEq(c, gout, 1e-10)
    end
end


function tests.gradcheck()
    local N, Tx, Ty, D = 2, 3, 4, 5
    local glimpse = nn.Glimpse(D)
    local x = {torch.randn(N, Tx, D):mul(0.1), torch.randn(N, Ty, D):mul(0.1)}
    local g = glimpse:forward(x)
    local dg = torch.randn(#g)
    glimpse:backward(x, dg)
    --print(glimpse.gradInput)
    local dx = glimpse.gradInput
    local dw = glimpse.gradWeight:clone()

    local function fx1(x1)
        local x_new = {x1, x[2]}
        return glimpse:forward(x_new) 
    end

    local function fx2(x2)
        local x_new = {x[1], x2}
        return glimpse:forward(x_new) 
    end

    local function fw(w)
        local old_w = glimpse.weight
        glimpse.weight = w
        local out = glimpse:forward(x)
        glimpse.weight = old_w
        return out
    end

    local dx1_num = gradcheck.numeric_gradient(fx1, x[1], dg)
    local dx2_num = gradcheck.numeric_gradient(fx2, x[2], dg)
    local dw_num = gradcheck.numeric_gradient(fw, glimpse.weight, dg)

    local dx1_error = gradcheck.relative_error(dx1_num, dx[1])
    local dx2_error = gradcheck.relative_error(dx2_num, dx[2])
    local dw_error = gradcheck.relative_error(dw_num, dw)

    tester:assertle(dx1_error, 1e-2, "dx[1]")
    tester:assertle(dx2_error, 1e-2, "dx[2]")
    tester:assertle(dw_error, 1e-2, "dw")
end


function tests.testBackward()
    local N, Tx, Ty, D = 10, 12, 15, 32
    local glimpse = nn.Glimpse(D)

    -- manual check with torch.nn
    local MM1 = nn.MM(false, true)
    local MM2 = nn.MM(false, false)
    local softmax = nn.SoftMax()

    local linear = nn.Linear(D, D, false)
    linear.weight:copy(glimpse.weight:t():clone())
    tester:assertTensorEq(linear.weight:t(), glimpse.weight, 1e-10)

    local gout
    local grad = torch.Tensor()
    for i = 1, 2 do
        local input = {torch.randn(N, Tx, D), torch.randn(N, Ty, D)}

        gout = glimpse:forward(input)
        grad:resizeAs(gout):uniform(0, 1)
        glimpse.gradWeight:zero() -- make sure to zero out weight
        local gradInput = glimpse:backward(input, grad)

        -- manually check
        local x, y = unpack(input)
        local y1 = linear:forward(y:view(Ty * N, -1))
        local score = MM1:forward{y1:view(N, Ty, -1), x}
        local patt = softmax:forward(score:view(-1, Tx))
        local c = MM2:forward{patt:view(N, Ty, Tx), x}

        MM2:backward({patt:view(N, Ty, Tx), x}, grad)
        local grad_patt, grad_x = unpack(MM2.gradInput)
        local grad_score = softmax:backward(score:view(-1, Tx), grad_patt:view(-1, Tx))
        MM1:backward({y1:view(N, Ty, -1), x}, grad_score:view(N, Ty, Tx))

        local grad_y1 = MM1.gradInput[1]
        grad_x:add(MM1.gradInput[2])
        linear.gradWeight:zero()
        local grad_y = linear:backward(y:view(Ty * N, -1), grad_y1:view(N * Ty, -1))
        grad_y = grad_y:view(N, Ty, D)

        tester:assertTensorEq(grad_x, gradInput[1], 1e-10)
        tester:assertTensorEq(linear.gradWeight, glimpse.gradWeight:t(), 1e-8)
        tester:assertTensorEq(grad_y, gradInput[2], 1e-10)
    end
end


tester:add(tests)
tester:run()
