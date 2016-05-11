--[[ For computing reward
Rewrite to make it work in TARDIS

Getting the right reward is the most crucial part in training REINFORCE
So we need to do it nice, slow and well documented
--]]

require 'math'
require 'xlua'
local evals = require 'eval'
local threads = require 'threads'

local RewardFactory = torch.class('RewardFactory')

--[[ This class returns an object for computing the reward at a
given time step.
The reward is partial BLEU score, smoothed

Parameters:
- `start` : index of time step at which we start computing the reward
- `bptt` :  the maximum length of a sequence.
- `vocab_size` : size of the dictionary
- `eosidx` : the id of the end of sentence token. Symbols after
    the first occurrence of eos (if any) are skipped.
- `padidx` : the id of the padding token
- `mbsz` : mini-batch size, which will be computed dynamically
--]]

function RewardFactory:__init(vocab_size, eosidx, unkidx, padidx)
    --[[ Initiate the Factory class.
    NOTE: MIXER compute the reward at the end of an episode, this reward
        will be bproped to the sequence till the point where we start drawn
        samples.

        Some variance reduction technique for REINFORCE is not used in MIXER
        TODO: Double check

    Parameters:
    - `vocab_size` : integer, vocabulary size
    - `eosidx` : integer, end of sentence index
    - `unkidx` : integer, unknown index
    - `padidx` : integer, index of padding symbol <pad>
    --]]

    self.start = 1

    self.vocab_size = vocab_size
    self.eosidx = eosidx
    self.padidx = padidx

    if unkidx == nil then
        print('dictionary does not have <unk>, ' ..
              'we are not skipping then while computing BLEU')
        self.unkidx = -1
    else
        self.unkidx = unkidx
    end

    self.reward_val = torch.Tensor()
    -- auxiliary variables
    self.inputt = torch.Tensor()
    self.targett = torch.Tensor()
    self.reset = torch.Tensor()
    self.input = torch.Tensor()
    self.target = torch.Tensor()
    -- Since counting works on cpu, we speed up by multi-threading.
    self.nthreads = 4
    threads.serialization('threads.sharedserialize')
    self.pool = threads.Threads(self.nthreads)
    self.pool:specific(true)
    for i = 1, self.nthreads do
        self.pool:addjob(
            i,
            function()
                require 'xlua'
                local evals = require 'eval'
                --require 'cutorch'
                require 'math'
            end
         )
     end

    self.pool:specific(false)
    self.order = 4 -- BLEU order
    self.score = torch.zeros(self.order, 3)
    self.sentence_bleu = 0
    self.length_input = 0
    self.length_target = 0
    self.counter = 0
    self.smoothing_val = 1
    self.adjust_bp = true
end

function RewardFactory:test_mode()
    self.smoothing_val = 0
    self.adjust_bp = false
end

function RewardFactory:training_mode()
    -- this is done only for training since BLEU at sentence level is not stable
    -- shouldn't use brevity penalty
    self.smoothing_val = 1
    self.adjust_bp = true
end

function RewardFactory:reset_vars()
    -- reset all the previous state
    self.length_input = 0
    self.length_target = 0
    self.counter = 0
    self.score:fill(0)
    self.sentence_bleu = 0
end

function RewardFactory:set_start(val)
    -- this is for bptt when we start sampling
    assert(val > 0)
    self.start = val
end

function RewardFactory:cuda()
    self.reset = self.reset:cuda()
end

function RewardFactory:get_reward(target, input, tt)
    --[[ Note on computing the reward
    TARDIS is implemented efficiently (without cloning model parameters),
    RNNs in TARDIS take a 2D tensor input and target (mbsz, bptt) and
    compute a forward pass over all time steps.

    For that reason, here, target and input will be 2D tensors (mbsz, bptt)

    --]]
    --[[
    Parameters:
    - `target` : table, each entry is a tensor of size mini-batch size
        storing the reference at a certain time step
    - `input` : table of tables, each table stores in its first entry the word
        we have sampled, the second entry stores an estimate of cumulative reward,
        and it is not used here
    - `tt` : integer, time step at which we wish to compute the reward


    DISCLAIMER: the score is smoothed
    because our sequences are short and it's likely that some scores are 0
    (which would make the geometric mean be 0 as well). Smoothing should be
    used only at training time (since at test time we evaluate at the
    corpus level).

    NOTE: target and input are tables with the same number of entries,
    however, each sequence can have an <eos> at different time steps
    (so effectively we do not assume that input and target have the same length).

    --]]

    -- first attempt
    -- It is OK to use 1D tensor for the reward values, we will duplicate it
    -- to the number of generation steps
    self.reward_val:fill(0)

    -- get the bptt steps here
    -- TODO: double check
    local bptt = #input

    function compute_bleu(target, input, tt, args, i)
        local bptt = target:size(1) -- this will be the length of the target
        -- get local copy of class member variables
        local start = args.start
        local vocab_size = args.vocab_size
        local eosidx = args.eosidx
        local unkidx = args.unkidx
        local mbsz = args.mbsz
        local reward_val = args.reward_val
        local inputt = torch.Tensor(bptt - start + 1)
        local targett = torch.Tensor(bptt - start + 1)
        local nthreads = args.nthreads
        local order = args.order
        local smoothing_val = args.smoothing_val
        local adjust_bp = args.adjust_bp
        local num_samples = math.floor(mbsz / nthreads)
        local first = (i-1) * num_samples + 1
        local last = (i < nthreads) and first + num_samples - 1 or mbsz
        print('inside ' .. i)
        print('bptt ' .. bptt)
        for ss = first, last do
            -- compute the length of the input and target sequences
            -- default values if <eos> is not found is bptt
            local target_length = bptt
            local input_length = bptt

            for step = 1, bptt do
                if target[step][ss] == eosidx then
                    target_length = step - 1
                    break
                end
            end

            for step = 1, bptt do
                if input[step][ss] == eosidx then
                    input_length = step - 1
                    break
                end
            end

            -- note that TARDIS does not use padding, so we remove some code
            assert(target_length >= 0 and input_length >= 0)
            -- we go up to 4-grams
            -- Note: if <eos> is detected before self.start then reward = 0
            local n = math.min(order, input_length - start + 1, target_length - start + 1)

            -- non-zero reward only if an <eos> has been found in the input
            -- or we reached the max length. We add 1 because input_length
            -- is the length up to the symbol before eos but we want to give
            -- reward when we encounter eos.
            if tt == math.min(input_length + 1, bptt) and n > 0 then
                local score = torch.Tensor(n):fill(0)
                -- extracts the ending part of the input and target sequences,
                -- taking into account ngrams that overlap between the
                -- conditioning part and the generated part.

                -- effective length, handling the previous n-1 grams
                -- consider ngrams overlapping with part we condition upon
                -- but be careful not to run out of words.
                local offset = math.min(n - 1, start - 1)
                local eff_seq_length_input = input_length - start + 1 + offset
                inputt:resize(eff_seq_length_input)

                local eff_seq_length_target = target_length - start + 1 + offset
                targett:resize(eff_seq_length_target)

                -- copy data from tables to tensors
                for step = 1, eff_seq_length_input do
                    inputt[step] = input[start + step - 1 - math.min(n - 1, start - 1)][ss]
                end

                for step = 1, eff_seq_length_target do
                    targett[step] = target[
                        start + step - 1 - math.min(n - 1, start - 1)][ss]
                end

                local counts_input = {}  -- store counts hashes for each n
                local counts_target = {}
                for nn = 1, n do
                    -- restrict counting to ngrams that depend on the
                    -- generated sequence (yet potentially overlapping with
                    -- the conditioning part of the sequence).
                    local curr_offs = math.max(offset + 1 - nn + 1, 1)
                    --[[ note that we do not count existing uni-gram, bi-gram, ...
                        when a word w is generated, it produces only one new n gram, so we count this one
                        this requires to shift the offset to the left n-1 words
                    --]]
                    counts_input[nn] = evals.get_counts(
                        inputt:narrow(1, curr_offs, eff_seq_length_input - curr_offs + 1),
                        nn, vocab_size)

                    counts_target[nn] = evals.get_counts(
                        targett:narrow(1, curr_offs, eff_seq_length_target - curr_offs + 1),
                        nn, vocab_size, unkidx)

                    score[nn] = evals.compute_score(
                                    counts_input[nn], counts_target[nn],
                                    smoothing_val)
                end
                -- compute bleu score: exp(1/N sum_n log score_n)
                reward_val[ss] = score:log():sum(1):div(n):exp()
                -- add brevity penalty
                local bp = 1
                if input_length < target_length then
                    math.exp(1 - (target_length +
                                (adjust_bp and smoothing_val or 0))
                            / input_length)
                end
                reward_val[ss] = reward_val[ss] * bp
            end -- reward > 0 only at the very end of the sequence only
        end -- end loop over samples
        collectgarbage()
    end

    -- resize vector here
    assert(#target == #input and #target > 0)
    local mbsz = target[1]:numel()

    self.reward_val:resize(mbsz)
    self.inputt:resize(bptt - self.start + 1)
    self.targett:resize(bptt - self.start + 1)
    self.reset:resize(mbsz)
    self.target:resize(#target, mbsz)
    self.input:resize(#input, mbsz)

    -- sweet! Now we process
    for cc = 1, #target do
        self.target:select(1, cc):copy(target[cc])
        self.input:select(1, cc):copy(input[cc][1])
    end
    print(self.input)
    print(self.target)

    local args = {start = self.start, vocab_size = self.vocab_size,
                eosidx = self.eosidx,
                unkidx = self.unkidx,
                mbsz = mbsz, reward_val = self.reward_val,
                inputt = self.inputt, targett = self.targett,
                nthreads = self.nthreads, order = self.order,
                smoothing_val = self.smoothing_val, adjust_bp = self.adjust_bp}

    for i = 1, self.nthreads do
        self.pool:addjob(compute_bleu, function() end, self.target, self.input, tt, args, i)
    end

    self.pool:synchronize()
    return self.reward_val
end

function RewardFactory:num_samples(target, input)
    -- we do not do padding, this code needs to be rewritten
    local padidx = -1
    self.reset:ne(target[self.start], padidx)
    -- can actually return the batch size
    -- return self.reset:numel()
    return self.reset:sum()
end
