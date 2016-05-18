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
    --[[
    self.weight_predictive_reward =
        (weight_predictive_reward == nil) and 0.01 or weight_predictive_reward
    --]]
    self.weight = (weight == nil) and 1 or weight
    self.normalizing_coeff = 1
    self.eos_idx = eos_idx
    self.pad_idx = pad_idx
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end


function RFCriterion:type(tp)
    parent.type(self, tp)
    self.reward = self.reward:type(tp)
    self.cumreward = self.cumreward:type(tp)
    self.gradInput[1] = self.gradInput[1]:type(tp)
    self.gradInput[2] = self.gradInput[2]:type(tp)
    self.reward_func:type(tp)
    return self
end

function RFCriterion:setWeight(ww)
    self.weight = ww
end

function RFCriterion:setSkips(ss)
    self.skips = ss
    self.reward_func:set_start(ss)
end

function RFCriterion:updateOutput(input, target)
    --[[
    Parameters:
    - `input` : table of two 2D tensors
        chosen_word and predictive_cumreward
    - `target` : 2D tensor of reference
    --]]
    local mbsz, seq_length = target:size(1), target:size(2)
    local num_steps = seq_length - self.skips + 1

    self.reward:resize(mbsz, num_steps)
    self.cumreward:resize(mbsz, num_steps)

    local input_cpu = input[1]:double()
    local target_cpu = target:double()

    for tt = seq_length, self.skips, -1 do
        local shifted_tt = tt - self.skips + 1
        self.reward[{{}, {shifted_tt}}] =
            self.reward_func:get_reward(target_cpu, input_cpu, tt)

            self.cumreward:select(2, shifted_tt):copy(
            self.reward:select(2, shifted_tt))
        if tt ~= seq_length then
            self.cumreward[{{}, shifted_tt}]:add(
                self.cumreward[{{}, shifted_tt + 1}])
        end
    end

    -- normalize
    local num_samples = self.cumreward:numel()
    assert(num_samples > 0, 'number of samples must not be zeros')
    self.normalizing_coeff = self.weight / (self.sizeAverage and num_samples or 1)
    self.output = -self.cumreward:sum() / num_samples
    return self.output
end

function RFCriterion:updateGradInput(input, target)
    local mbsz, seq_length = target:size(1), target:size(2)
    self.gradInput[1]:resizeAs(input[1]):zero()
    self.gradInput[2]:resizeAs(target):zero()
    for tt = seq_length, self.skips, -1 do
        local shifted_tt = tt - self.skips + 1
        self.gradInput[1]:select(2, tt):add(
            input[2]:select(2, tt),
            -1, self.cumreward:select(2, shifted_tt))
    end
    self.gradInput[2]:copy(self.gradInput[1])
    self.gradInput[1]:mul(self.normalizing_coeff)
    --self.gradInput[2]:mul(self.weight_predictive_reward)
    return self.gradInput
end

function RFCriterion:training_mode()
    self.reward_func:training_mode()
end

function RFCriterion:test_mode()
    self.reward_func:test_mode()
end
