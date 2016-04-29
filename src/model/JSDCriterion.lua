-- url: http://arxiv.org/pdf/1511.05101v1.pdf
local JSDCriterion, parent = torch.class('nn.JSDCriterion', 'nn.Criterion')

function JSDCriterion:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end

    self.target = torch.zeros(1):long()
    self.buffer = torch.Tensor() -- for (q + q) / 2
    self.crit1 = nn.CrossEntropyCriterion()
    self.crit2 = nn.DistKLDivCriterion()
    self.log = nn.Log()
    self.grad_buffer = torch.Tensor()
    self._input = torch.Tensor()
    self.tau = 0.5
end

function JSDCriterion:updateOutput(input, target)
    if type(target) == 'number' then
        if input:type() ~= 'torch.CudaTensor' then
            self.target = self.target:long()
        end
        self.target[1] = target
    elseif target:type() == 'torch.CudaTensor' then
        self.target = target
    else
        self.target = target:long()
    end

    self.buffer:resizeAs(input):zero():scatter(2, self.target:view(-1,1), 1)
    self.buffer:add(input):div(2)

    self.output = self.crit1:forward(self.buffer, self.target)
    -- compute log buffer
    self.buffer:log()
    self.output = self.output * self.tau + (1-self.tau) * self.crit2:forward(self.buffer, input)
    return self.output

end



function JSDCriterion:updateGradInput(input, target)
	--[[ Derivative of K[Q || (P+Q)/2] w.r.t Q
	= deriv {q[i] ( log q[i] - log[(p[i]+q[i])/2] )}
	= 1 + log q[i] - log (p[i]+q[i])/2 ) - q[i]/(p[i]+q[i])
	]]
    self._input:resizeAs(input):copy(input)
    self.grad_buffer:resizeAs(self.buffer):copy(self.buffer):mul(-1):add(1)
    self.buffer:exp()
    self.grad_buffer:add(-1, self._input:cdiv(self.buffer):div(2))
    self.grad_buffer:add(self._input:log())
	self.grad_buffer:div(target:numel()) -- sizeAverage

    self.gradInput = self.crit1:backward(self.buffer, self.target)
    self.gradInput:mul(2*self.tau):add(self.grad_buffer:mul(1-self.tau))
    return self.gradInput
end
