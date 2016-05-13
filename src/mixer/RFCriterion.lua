-- REINFORCE Criterion
-- author: Ke Tran <m.k.tran@uva.nl>

local RFCriterion, parent = torch.class('RFCriterion', 'nn.Criterion')


function RFCriterion:__init(reward_func, eos_idx, pad_idx, skips,
                            weight, weight_predictive_reward)
    parent.__init(self)

    self.sizeAverage = false
    self.reward_func = reward_func
    self.reward = torch.Tensor()
    self.cumreward = torch.Tensor()
    self.skips = (skips == nil) and 1 or skips
    self.weight_predictive_reward = 
        (weight_predictive_reward == nil) and 0.01 or weight_predictive_reward
    self.weight = (weight == nil) and 1 or weight
    self.num_samples = 0
    self.normalizing_coeff = 1
    self.eos_idx = eos_idx
    self.pad_idx = pad_idx
    self.reset = torch.Tensor()
end


function RFCriterion:type(tp)
    parent.type(self, tp)
    self.reward = self.reward:type(tp)
    self.cumreward = self.cumreward:type(tp)
    return self
end

function RFCriterion:setWeight(ww)
    self.weight = ww
end

function RFCriterion:setSkips(ss)
    self.skips = ss
    self.reward_func:set_start(ss)
end


