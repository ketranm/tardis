local ReinforceCriterion, parent = torch.class('nn.ReinforceCriterion', 'nn.Criterion')


function ReinforceCriterion:__init(reward_func, seq_length, eos_index,
                                    padding_index, skips, weight,
                                    weight_predictive_reward)
    --[[ Initialize the Reinforce objective
    Parameters:
    - `reward_func` : function, which is used to compute the reward, given
        the ground truth input sequence, the generated sequence and the
        current time step.
    - `seq_length` : integer, length of the sequence we use
        (can we do compute it on the fly?)
    - `skips` : integer, the number of time steps we skip from
        the input and the target (init)
    - `weight` : double, weight on the loss produced by this criterion
        (we interpolate with xment)
    - `weight_predictive_reward` : double, weight on the gradient
        of the cumulative reward predictor
    --]]

    parent.__init(self)
    self.gradInput = {}
    self.seq_length = seq_length -- TODO: maxSeqLength
    for tt = 1, seq_length do
        self.gradInput[tt] = {}
        self.gradInput[tt][1] = torch.Tensor()
        self.gradInput[tt][2] = torch.Tensor()
    end
    self.sizeAverage = false
    self.reward_func = reward_func
    self.reward = torch.Tensor()
    self.cumreward = torch.Tensor()
    self.skips = (skips == nil) and 1 or skips
    assert(self.skips <= self.seq_length)
    -- by default, update the cumulative reward predictor
    -- at a slower pace
    self.weight_predictive_reward = 
        (weight_predictive_reward == nil) and 0.01 or weight_predictive_reward
    self.weight = (weight == nil) and 1 or weight
    self.num_samples = 0
    self.normalizing_coeff = 1
    self.eos = eos_index
    self.padding = padding_index
    self.reset = torch.Tensor()
end

function ReinforceCriterion:type(tp)
    -- torch.DoubleTensor or torch.CudaTensor
    -- convert all tensor to type tp
    parent.type(self, tp)
    for tt = 1, self.seq_length do
        self.gradInput[tt][1] = self.gradInput[tt][1]:type(tp)
        self.gradInput[tt][2] = self.gradInput[tt][2]:type(tp)
    end
    self.reward = self.reward:type(tp)
    self.cumreward = self.cumreward:type(tp)
    return self
end

function ReinforceCriterion:set_weight(ww)
    self.weight = ww
end

function ReinforceCriterion:set_skips(ss)
    self.skips = ss
    self.reward_func:set_start(ss) -- TODO: check in reward function the set_start method
end


function ReinforceCriterion:updateOutput(input, target)
    --[[
    Parameters:
    - `input` : table, storing the tupe (chosen_word, predictive_cumulative_reward)_t, t = 1,..., T
    - `target` : table, storing label at each time step
    --]]

    -- compute the reward at each time step
    local mbsz = target[1]:size(1)
    local num_steps = self.seq_length - self.skips + 1
    self.reward:resize(mbsz, num_steps)
    self.cumreward:resize(mbsz, num_steps)
    self.num_samples = 0
    for tt = self.seq_length, self.skips, -1 do
        -- reinforce only applies from skips to end of sentence
        local shifted_tt = tt - self.skips + 1
        self.reward:select(2, shifted_tt):copy(self.reward_func:get_reward(target, input, tt))
        -- this compute cumulative reward recursively on the fly
        if tt == self.seq_length then
            self.cumreward:select(2, shifted_tt):copy(self.reward:select(2, shifted_tt))
        else
            self.cumreward:select(2, shifted_tt):add(
                self.cumreward:select(2, shifted_tt + 1), self.reward:select(2, shifted_tt))
        end
    end

    self.num_samples = self.reward_func:num_samples(target, input) -- TODO: check the reward factory
    self.normalizing_coeff = self.weight / (self.sizeAverage and self.num_samples or 1)
    -- here there is a "-" because we minimize
    self.output = - self.cumreward:select(2, 1):sum() * self.normalizing_coeff
    return self.output, self.num_samples
end

function ReinforceCriterion:updateGradInput(input, target)
    --[[ Back-propagation through input at each time step
    derivative through chosen action is
    (predictive_cumulative_reward - actual_cumulative_reward)_t
    note that we are minimizing the loss, so we flip the sign of the gradient
    --]]

    local mbsz = target[1]:size(1)
    for tt = self.seq_length, self.skips, -1 do
        local shifted_tt = tt - self.skips + 1
        -- derivative w.r.t chosen action
        self.gradInput[tt][1]:resizeAs(input[tt][1]) -- this will be gradient to the lucky sampled words
        -- minus the baseline
        self.gradInput[tt][1]:add(input[tt][2]:squeeze(), -1, self.cumreward:select(2, shifted_tt))
        self.gradInput[tt]:mul(self.normalizing_coeff)
        -- I don't think this is necessary in TARIDS, as we don't do padding
        -- reset gradient to 0 if any input (at any time) has PAD
        self.reset:resize(mbsz)
        self.reset:ne(input[tt][1], self.padding) -- set in RNNreinforce
        self.gradInput[tt][1]:cmul(self.reset)
        -- copy over to the other input gradient as well
        -- this gradient will go to the predictor
        -- TODO: check with the tandem RNN idea
        self.gradInput[tt][2]:resizeAs(input[tt][2])
        self.gradInput[tt][2]:copy(self.gradInput[tt][1])
        self.gradInput[tt][2]:mul(self.weight_predictive_reward)
    end
    -- fill in the skipped steps with 0s
    for tt = self.skips - 1, 1, -1 do
        self.gradInput[tt][1]:resizeAs(input[tt][1])
        self.gradInput[tt][1]:fill(0)
        self.gradInput[tt][2]:resizeAs(input[tt][2])
        self.gradInput[tt][2]:fill(0)
    end
    return self.gradInput
end

-- helper functions
function ReinforceCriterion:get_num_samples(input, target)
    return self.reward_func:num_samples(target, reward)
end

function ReinforceCriterion:reset_reward()
    return self.reward_func:reset_vars()
end

function ReinforceCriterion:training_mode()
    self.reward_func:training_mode()
end

function ReinforceCriterion:test_mode()
    self.reward_func:test_mode()
end
