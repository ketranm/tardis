--[[
Sequence to Sequence model.
It has an encoder and a decoder.
--]]

require 'model.Transducer'
local model_utils = require 'model.model_utils'

local NMT, parent = torch.class('nn.NMT', 'nn.Module')

function NMT:__init(kwargs)
    local kwargs = kwargs;

    -- over write option
    kwargs.vocab_size = kwargs.source_vocab_size
    self.encoder = nn.Transducer(kwargs)

    -- over write option
    kwargs.vocab_size = kwargs.target_vocab_size
    self.decoder = nn.Transducer(kwargs)

    self.layer = nn.Sequential()
    self.layer:add(nn.View(-1, kwargs.hidden_size))
    self.layer:add(nn.Linear(kwargs.hidden_size, kwargs.target_vocab_size))
    self.layer:add(nn.LogSoftMax())

    self.criterion = nn.ClassNLLCriterion()
    self.grad_encoder = torch.Tensor()  -- always zeros

    -- get parameters and gradients for optimization
    self.params, self.gradParams = model_utils.combine_all_parameters(self.encoder, self.decoder, self.layer)
    self.max_norm = kwargs.max_norm or 5
    self.records = {}
end


function NMT:forward(input, target)
    --[[Forward pass
    Args:
        input: a table of {x, y} where x = (batch_size, len_xs) Tensor and
        y = (batch_size, len_ys) Tensor
        target: a tensor of (batch_size, len_ys)
    --]]
    -- encode pass
    local output_encoder = self.encoder:updateOutput(input[1])
    local last_encoder_state = self.encoder:last_state()
    self.decoder:init_state(last_encoder_state)
    local output_decoder = self.decoder:updateOutput(input[2])
    -- forward to fully connected layer
    local log_prob = self.layer:forward(output_decoder)
    -- record all the temporal tensors for back-propagation
    -- this is cheap because we use reference
    self.records = {output_encoder, output_decoder, log_prob}
    return self.criterion:forward(log_prob, target)
end


function NMT:backward(input, target)
    -- zero out gradients
    self.gradParams:zero()

    -- unpack records
    local output_encoder, output_decoder, log_prob = unpack(self.records)
    local grad_encoder = self.grad_encoder

    local grad_loss = self.criterion:backward(log_prob, target)
    local grad_decoder = self.layer:backward(output_decoder, grad_loss)

    -- backward pass
    self.decoder:backward(input[2], grad_decoder)
    local grad_state = self.decoder:get_grad_state()
    self.encoder:set_grad_state(grad_state)

    grad_encoder:resizeAs(output_encoder):zero()
    self.encoder:backward(input[1], grad_encoder)
end


function NMT:update(learning_rate)
    local grad_norm = self.gradParams:norm()
    local scale = learning_rate
    if grad_norm > self.max_norm then
        scale = scale*self.max_norm /grad_norm
    end
    self.params:add(self.gradParams:mul(-scale)) -- do it in-place
end


function NMT:parameters()
    return self.params
end


function NMT:training()
    self.encoder:training()
    self.decoder:training()
end


function NMT:evaluate()
    self.encoder:evaluate()
    self.decoder:evaluate()
end


function flat_to_rc(v, indices, flat_index)
    local row = math.floor((flat_index - 1)/v:size(2)) + 1
    return row, indices[row][(flat_index - 1) % v:size(2) + 1]
end


function NMT:_decode_string(x)
    local ws = {}
    local vocab = self.target_vocab
    for i = 1, x:nElement() do
        local idx = x[i]
        if idx ~= vocab['<s>'] and idx ~= vocab['</s>'] then
            ws[#ws + 1] = self.id2word[idx]
        end
    end
    return table.concat(ws, ' ')
end


function NMT:_encode_string(x)
    -- encode source sentence
    local xs = stringx.split(x)
    local xids = {}
    for i = #xs, 1, -1 do
        local w = xs[i]
        local idx = self.source_vocab[w] or self.source_vocab['<unk>']
        table.insert(xids, idx)
    end
    return torch.Tensor(xids):view(1, -1)
end


function NMT:use_vocab(vocab)
    self.source_vocab = vocab[1]
    self.target_vocab = vocab[2]
    self.id2word = {}
    for w, id in pairs(self.target_vocab) do
        self.id2word[id] = w
    end
end


function NMT:load_model(filename)
    local params = torch.load(filename)
    self.params:copy(params)
end


function NMT:save_model(filename)
    torch.save(filename, self.params)
end


function NMT:translate(x, beam_size, max_length)
    --[[Translate input sentence with beam search
    Args:
        x: source sentence, (1, T) Tensor
        beam_size: size of the beam search
    --]]
    local source_vocab, target_vocab = self.source_vocab, self.target_vocab
    local idx_GO = target_vocab['<s>']
    local idx_EOS = target_vocab['</s>']

    local K = beam_size or 10
    local T = max_length or 50

    x = self:_encode_string(x)
    x = x:expand(K, x:size(2)):typeAs(self.params)

    local output_encoder = self.encoder:updateOutput(x)
    local last_encoder_state = self.encoder:last_state()

    local scores = torch.Tensor():typeAs(x):resize(K, 1):zero()
    local hyps = torch.Tensor():typeAs(x):resize(T, K):zero():fill(idx_GO)
    local prev_state = last_encoder_state
    local complete_hyps = {}
    for i = 1, T-1 do
        self.decoder:init_state(prev_state)
        local cur_y = hyps[i]:view(-1, 1)
        local output_encoder = self.decoder:forward(cur_y)
        local log_prob = self.layer:forward(output_encoder)
        local max_scores, indices = log_prob:topk(K, true)
        -- previous scores
        local cur_scores = scores:repeatTensor(1, K)
        -- add them to current ones
        max_scores:add(cur_scores)

        local flat = max_scores:view(max_scores:size(1) * max_scores:size(2))
        local next_indices = {}
        local expand_k = {}
        local k = 1
        while k <= K do
            local score, index = flat:max(1)
            local prev_k, yi = flat_to_rc(max_scores, indices, index[1])
            -- make it -INF so we will not select it next time 
            flat[index[1]] = -math.huge
            if yi == idx_EOS then
                -- complete hypothesis
                local hypo = self:_decode_string(hyps[{{}, prev_k}])
                complete_hyps[hypo] = scores[prev_k][1]/i
            elseif yi ~= idx_GO then
                table.insert(next_indices,  yi)
                table.insert(expand_k, prev_k)
                scores[k] = score[1]
                k = k + 1
            end
        end
        expand_k = torch.Tensor(expand_k):long()
        local next_hyps = hyps:index(2, expand_k)  -- remember to convert to cuda
        next_hyps[i+1]:copy(torch.Tensor(next_indices))
        hyps = next_hyps
        -- carry over the state of selected k
        local cur_state = self.decoder:last_state()
        local next_state = {}
        for _, state in ipairs(cur_state) do
            local my_state = {}
            for _, s in ipairs(state) do
                table.insert(my_state, s:index(1, expand_k))
            end
            table.insert(next_state, my_state)
        end
        prev_state = next_state
    end
    for k = 1, K do
        local hypo = self:_decode_string(hyps[{{}, k}])
        complete_hyps[hypo] = scores[k][1]/(T-1)
    end
    local n_best = {}
    for hypo in pairs(complete_hyps) do n_best[#n_best + 1] = hypo end
    -- sort the result and pick the best one
    table.sort(n_best, function(s1, s2)
        return complete_hyps[s1] > complete_hyps[s2] or complete_hyps[s1] > complete_hyps[s2] and s1 > s2
    end)
    return n_best[1]
end


function NMT:clearState()
    self.encoder:clearState()
    self.decoder:clearState()
    self.layer:clearState()
end