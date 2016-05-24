require 'nn'
require 'core.ReverseTensor'
local tests = {}
local tester = torch.Tester()


function tests.forward()
    for i = 1, 10 do
        local d = torch.random(1, 3)
        local rt = nn.ReverseTensor(d)
        local x = torch.randn(10, 20, 30)
        local y = rt:forward(x)
        local z = rt:forward(y)
        x:cdiv(z)
        tester:assertTensorEq(x, torch.ones(#x), 0)
    end
end

function tests.backward()
    for i = 1, 10 do
        local d = torch.random(1, 3)
        local rt = nn.ReverseTensor(d)
        local x = torch.randn(10, 20, 30)
        local y = rt:forward(x)
        local dy = torch.randn(#y)
        local dx = rt:backward(x, dy)
        local z = rt:forward(dy)
        dx:cdiv(z)
        tester:assertTensorEq(dx, torch.ones(#dx), 0)
    end
end

tester:add(tests)
tester:run()

