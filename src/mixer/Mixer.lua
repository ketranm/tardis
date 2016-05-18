--[[ Mixed Incremental Cross-Entropy REINFORCE
This class perform MIXER training
Author: Ke Tran <m.k.tran@uva.nl>
--]]

require 'mixer.RewardFactory'
require 'mixer.RFCriterion'
require 'nn'
require 'optim'

local Mixer = torch.class('Mixer')

function Mixer:__init(config, net)
    self.eos_idx = config.eos_idx
    self.pad_idx = config.pad_idx
    self.unk_idx = config.unk_idx
    self.vocab_size = config.trgVocabSize
    self.topk = config.topk  -- if we sample top K predictions

    local dtype = config.dtype or 'torch.CudaTensor'
    -- setup Cross-Entropy Criterion
    local weights = torch.ones(self.vocab_size)
    weights[self.pad_idx] = 0
    self.criterion_xe = nn.ClassNLLCriterion(weights, false)
    self.criterion_xe:type(dtype)

    -- setup REINFORCE Criterion
    local reward_func =
        RewardFactory(self.vocab_size,
                      self.eos_idx,
                      self.unk_idx,
                      self.pad_idx)
    self.criterion_rf = RFCriterion(reward_func, self.eos_idx, self.pad_idx, 1)
    self.criterion_rf:setWeight(1)
    self.criterion_rf:type(dtype)
    self.criterion_rf:training_mode()

    -- set up a cumulative predictor
    self.crp = nn.Linear(config.hiddenSize, 1)
    self.crp:type(dtype)
    self.wcrp, self.dwcrp = self.crp:getParameters()
    self.crp_cfg = {}
    self.crp_sta = {}
    -- buffer for REINFORCE derivative
    self.drf = torch.Tensor():type(dtype)
    self.buffer = torch.Tensor():type(dtype)
    self.net = net
end


function Mixer:save(fname)
    local data = {self.net.params, self.wcrp
                  self.crp_cfg, self.crp_sta}
    torch.save(fname, model)
end

function Mixer:load(fname)
    local data = torch.load(fname)
    self.net.params:copy(data[1])
    self.wcrp:copy(data[2])
    self.crp_cfg = data[3]
    self.crp_sta = data[4]
end

function Mixer:padify(pred)
    -- when previous token was eos or pad
    -- pad is produced deterministically
    local mbsz, length = pred:size(1), pred:size(2)
    local mask1 = torch.Tensor(mbsz):typeAs(pred)
    local mask2 = torch.Tensor(mbsz):typeAs(pred)
    for tt = 2, length do
        torch.eq(mask1, pred[{{},tt - 1}], self.eos_idx)
        torch.eq(mask2, pred[{{},tt - 1}], self.pad_idx)
        mask1:add(mask2)

        torch.ne(mask2, mask1, 1) -- negate
        mask2:cmul(pred[{{}, tt}])
        mask1:mul(self.pad_idx)
        mask1:add(mask2)
        pred[{{}, tt}]:copy(mask1)
    end
    return pred
end

function Mixer:trainOneBatch(input, target, skips, learning_rate)
    local src, trg_inp = unpack(input)
    local mbsz, length = trg_inp:size(1), trg_inp:size(2)
    local length_xe = skips - 1
    local length_rf = length - skips + 1

    -- run the encoder
    self.net:stepEncoder(src)
    local x = nil
    -- roll-in with the ground truth
    if length_xe > 0 then
        x = trg_inp:narrow(2, 1, length_xe)
        self.net:stepDecoder(x)
    end
    -- one-step
    x = trg_inp[{{}, {skips}}]
    self.net:stepDecoder(x)
    -- running example
    -- skips = 4, length = 6
    -- length_rf = 6 - 4 + 1 = 3
    -- inp - - - x0 s1 s2
    -- out - - - s1 s2 s3

    local sampled_w = self.net:sample(length_rf, self.topk)
    self:padify(sampled_w)

    -- we do not need trg_inp any more. Overwrite it!
    trg_inp[{{}, {skips + 1, -1}}] = sampled_w[{{}, {1, -2}}]
    local y = target:clone()
    y[{{}, {skips, -1}}] = sampled_w

    -- roll-out with samples
    --x = trg_inp
    self.net.buffers.prevState = self.net.encoder:lastState()
    local logProb = self.net:stepDecoder(trg_inp)

    -- compute REINFORCE loss and gradients
    self.criterion_rf:setSkips(skips)
    local state = self.net:selectState()
    local pred_cumreward = self.crp:forward(state)
    assert(pred_cumreward:numel() == trg_inp:numel())
    local baseline = pred_cumreward:viewAs(trg_inp)
    local reward = self.criterion_rf:forward({y, baseline}, target)
    local grad_rf = self.criterion_rf:backward({y, baseline}, target)
    local grad_crp = grad_rf[2]

    -- optimize the cumulative reward predictor
    local feval = function(w)
        if w ~= self.wcrp then
            self.wcrp:copy(w)
        end

        self.dwcrp:zero()
        local crp_err = grad_crp:norm()
        local nsamples = mbsz * length_rf
        crp_err = crp_err^2 / nsamples

        self.crp:backward(state, grad_crp:view(-1, 1))
        return crp_err, self.dwcrp
    end

    -- optimize this shit
    local _, fx = optim.adadelta(feval, self.wcrp, self.crp_cfg, self.crp_sta)
    local crp_err = fx[1]

    -- overwrite target as we do not need it anymore
    -- use padding to ignore sampled words
    target[{{}, {skips, -1}}]:fill(self.pad_idx)
    target = target:view(-1)

    local nsamples = target:ne(self.pad_idx):sum()
    local nll = self.criterion_xe:forward(logProb, target)
    nll = nll / nsamples

    -- compute the gradient of Mixer
    -- (1) take gradient of xent
    local gradLoss = self.criterion_xe:backward(logProb, target)
    -- (2) normalize it by the number of samples
    gradLoss:div(nsamples)
    -- (3) take gradient of REINFORCE
    self.drf:resize(mbsz, length, self.vocab_size):zero()
    self.drf:scatter(3, y:view(mbsz, -1, 1), grad_rf[1]:view(mbsz, -1, 1))
    -- (4) add it to the gradient of xent
    local sizes = torch.LongStorage{mbsz * length, self.vocab_size}
    torch.view(self.buffer, self.drf, sizes)  -- more memory efficient
    gradLoss:add(self.buffer)

    ----------- OK! We are ready to backprop --------------
    self.net:backward({src, trg_inp}, target, gradLoss)
    self.net:update(learning_rate)

    return nll, -reward, crp_err
end
