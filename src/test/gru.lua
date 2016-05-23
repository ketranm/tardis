require 'nn'
require 'core.GRU'
require 'core.LSTM'

local gradcheck = require 'misc.gradcheck'
local tests = {}
local tester = torch.Tester()

-- test forward pass
function tests.testForward()
	local N, T, D, H = 10, 20, 32, 32
	local x = torch.randn(N, T, D)
	local gru = nn.GRU(D, H)
	local my_h = gru:forward(x)
	
	local bias = gru.bias
	local W = gru.weight
	-- compute GRU with nn
	local update_gate = nn.Sequential()
	update_gate:add(nn.Linear(D + H, H))
	update_gate:add(nn.Sigmoid())

	update_gate:get(1).weight:copy(W[{{}, {1, H}}]:t())
	update_gate:get(1).bias:copy(bias[{{1, H}}])
	--
	local reset_gate = nn.Sequential()
	reset_gate:add(nn.Linear(D + H, H))
	reset_gate:add(nn.Sigmoid())

	reset_gate:get(1).weight:copy(W[{{}, {H + 1, 2 * H}}]:t())
	reset_gate:get(1).bias:copy(bias[{{H + 1, 2 * H}}])
	--
	local hidden_candiate = nn.Sequential()
	hidden_candiate:add(nn.Linear(D + H, H))
	hidden_candiate:add(nn.Tanh())

	hidden_candiate:get(1).weight:copy(W[{{}, {2 * H + 1, 3 * H}}]:t())
	hidden_candiate:get(1).bias:copy(bias[{{2 * H + 1, 3 * H}}])

	local jhx = nn.JoinTable(2)
	local jhr = nn.JoinTable(2)

	local h0 = torch.Tensor(N, H):zero()
	local h = torch.Tensor(N, T, H):zero()
	local prev_h = h0
	for t = 1, T do
		local xt = x[{{}, t}]
		local hx = jhx:forward{xt, prev_h}
		local z = update_gate:forward(hx)
		local r = reset_gate:forward(hx)
		local hr = r:cmul(prev_h)
		xhr = jhr:forward{xt, hr}
		local hc = hidden_candiate:forward(xhr)
		local next_h = h[{{}, t}]
		next_h:fill(1):add(-1, z):cmul(prev_h)
		next_h:addcmul(z, hc)
		prev_h = next_h
	end

	tester:assertTensorEq(my_h, h, 1e-10)
end


function tests.gradcheck()
	local N, T, D, H = 2, 3, 4, 5
    local x = torch.randn(N, T, D)
    local gru = nn.GRU(D, H)

    local h = gru:forward(x)
    gru:zeroGradParameters()
    local dh = torch.randn(#h)
    local dx = gru:backward(x, dh)
    local dw = gru.gradWeight:clone()
    local db = gru.gradBias:clone()
    

    local function fx(x) return gru:forward(x) end

    local function fw(w)
        local old_w = gru.weight
        gru.weight = w
        local out = gru:forward(x)
        gru.weight = old_w
        return out
    end

    local function fb(b)
        local old_b = gru.bias
        gru.bias = b
        local out = gru:forward(x)
        gru.bias = old_b
        return out
    end

    local dx_num = gradcheck.numeric_gradient(fx, x, dh)
    local dw_num = gradcheck.numeric_gradient(fw, gru.weight, dh)
    local db_num = gradcheck.numeric_gradient(fb, gru.bias, dh)

    local dx_error = gradcheck.relative_error(dx_num, dx)
    local dw_error = gradcheck.relative_error(dw_num, dw)
    local db_error = gradcheck.relative_error(db_num, db)

    tester:assertle(dx_error, 1e-5)
    tester:assertle(dw_error, 1e-4)
    tester:assertle(db_error, 1e-5)

end

-- test speed between LSTM and GRU
function testSpeed()
	local N, T, D, H = 64, 20, 200, 200
	local x = torch.randn(N, T, D)
	local gru = nn.GRU(D, H)
	local lstm = nn.LSTM(D, H)
	local n = 42
	local grad = torch.Tensor(N, T, H):normal(0,0.01)
	local scale = 1e-2
	local timer = torch.Timer()
	for i = 1, n do
		gru:forward(x)
		gru:backward(x, grad, scale)
	end
	print('Time elapsed for GRU: ' .. timer:time().real .. ' seconds')

	timer:reset()
	for i = 1, n do
		lstm:forward(x)
		lstm:backward(x, grad, scale)
	end
	print('Time elapsed for LSTM: ' .. timer:time().real .. ' seconds')
end

tester:add(tests)
tester:run()
testSpeed()