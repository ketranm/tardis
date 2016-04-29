--[[ Implementing a JSD loss

Reference:
How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?
Ferenc Huszár (2015)
url: http://arxiv.org/pdf/1511.05101v1.pdf
--]]
local JSDCriterion, parent = torch.class('nn.JSDCriterion', 'nn.Criterion')

function JSDCriterion:__init(sizeAverage)
    parent.__init(self)
    -- by default we do sizeAverage

    self.target = torch.zeros(1):long()
    self.distAverage = torch.Tensor() -- store (Q + P) / 2

    self.crit1 = nn.CrossEntropyCriterion()
    self.crit2 = nn.DistKLDivCriterion()
    self.grad_buffer = torch.Tensor()
    self.buffer = torch.Tensor()
end

function JSDCriterion:updateOutput(input, target)
    --[[ Forward pass

    Parameters:
    - `input` :  tensor of model distribution Q
    - `target` : tensor or number of the correct labels

    Returns:
    - JSD loss
    --]]
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

    -- assign probability 1 to the ground truth
    self.distAverage:resizeAs(input):zero():scatter(2, self.target:view(-1,1), 1)
    self.distAverage:add(input):div(2)
    -- KL (P | distAverage)
    self.output = self.crit1:forward(self.distAverage, self.target)

    -- compute log of the averaged distribution
    self.distAverage:log()
    self.output = self.output + self.crit2:forward(self.distAverage, input)

    return self.output / 2
end



function JSDCriterion:updateGradInput(input, target)
    --[[ Derivative of KL[Q || (P+Q)/2] w.r.t Q

    dKL/dQ {q[i] ( log q[i] - log[(p[i]+q[i])/2] )}
    = 1 + log q[i] - log (p[i]+q[i])/2 ) - q[i]/(p[i]+q[i])
    ]]

    -- (a) 1 - log(P + Q) / 2
    self.grad_buffer:resizeAs(input):fill(1)
    self.grad_buffer:add(-1, self.distAverage)


    self.distAverage:exp()
    self.buffer:cdiv(input, self.distAverage)
    -- (b) - Q / (P + Q)
    self.grad_buffer:add(-1, self.buffer:div(2))

    -- compute log Q
    torch.log(self.buffer, input)
    self.grad_buffer:add(self.buffer)

    self.grad_buffer:div(target:numel()) -- sizeAverage

    self.gradInput = self.crit1:backward(self.distAverage, self.target)

    -- have to rescale the derivative by 2 * 0.5
    self.gradInput:add(0.5, self.grad_buffer)
    return self.gradInput
end
