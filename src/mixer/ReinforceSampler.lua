--[[ Sampling from Categorical distribution
This class is useful for REINFORCE criterion

TODO: rename to MultinomilaSampler
--]]

local ReinforceSampler, parent = torch.class('nn.ReinforceSampler', 'nn.Module')

function ReinforceSampler:__init()
    parent.__init(self)
    -- by default, we always sample from categorical distribution
    self.prob = torch.Tensor()
end

function ReinforceSampler:updateOutput(input)
    --[[ Sampling
    Parameters:
    - `input` : 1D Tensor of log-probability
    Returns:
    - `output` : sample generated from this log-prob

    --]]
    self.prob:resizeAs(input)
    self.prob:copy(input)
    self.prob:exp()
    self.output:resize(input:size(1), 1)
    torch.multinomial(self.output, self.prob, 1)
    return self.output
end

function ReinforceSampler:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    for ss = 1, self.gradInput:size(1) do
        -- adding round because sometimes multinomial returns a float 1e-6
        -- far from an integer.
        self.gradInput[ss][torch.round(self.output[ss][1])] = 
            gradOutput[ss][1]
    end
    return self.gradInput
end
