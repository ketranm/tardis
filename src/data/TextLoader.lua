-- Monolingual text loader
-- Author: Ke Tran <m.k.tran@uva.nl>

local DataLoader = torch.class('DataLoader')
local _ = require 'moses'

function DataLoader:__init(config)
    self._bos = '<s>'
    self._eos = '</s>'
    self._unk = '<unk>'
    self._pad = '<pad>'

    self._start_vocab = {self._bos, self._eos, self._unk, self._pad}

    -- just in case
    self.bos_idx = 1
    self.eos_idx = 2
    self.unk_idx = 3
    self.pad_idx = 4

    -- data path

    local trainFile = path.join(config.dataDir, 'train.txt')

    local validFile = path.join(config.dataDir, 'valid.txt')

    -- helper
    local vocabFile = path.join(config.dataDir, 'vocab.t7')

    -- auxiliary file to store additional information about shards
    local indexFile = path.join(config.dataDir, 'index.t7')

    self.vocabSize = config.vocabSize
    self.batchSize = config.batchSize

    self.vocab = {}
    self.tracker = {train = {tensorFiles = {}, nbatches = 0},
                    valid = {tensorFiles = {}, nbatches = 0}}

    local trainPrefix = path.join(config.dataDir, 'train')
    local validPrefix = path.join(config.dataDir, 'valid')

    if not path.exists(vocabFile) then
        print('run pre-processing, one-time setup!')
        print('creating source vocabulary ...')
        self.vocab = self:_makeVocab(trainFile, self.vocabSize)

        torch.save(vocabFile, self.vocab)

        print('create training tensor files...')
        self:text2Tensor(trainFile, trainPrefix,
            config.shardSize, self.tracker['train'])

        print('create validation tensor files...')
        self:text2Tensor(validFile, validPrefix,
            config.shardSize, self.tracker['valid'])

        torch.save(indexFile, self.tracker)
    else
        self.vocab = torch.load(vocabFile)
        self.tracker = torch.load(indexFile)
    end

end

function DataLoader:_read(mode)
    -- shuffle training shard
    assert(mode == 'train' or mode == 'valid')
    self.curr_tracker = self.tracker[mode]
    self.curr_tracker.tensorFiles = _.shuffle(self.curr_tracker.tensorFiles)
    self.shard_idx = 0
    self.batch_idx = 0
    self.curr_batch = 0
    self.max_batch_idx = -1
end

function DataLoader:readTrain()
    self:_read('train')
end

function DataLoader:readValid()
    self:_read('valid')
end

function DataLoader:nbatches()
    return self.curr_tracker.nbatches
end

function DataLoader:nextBatch()
    --return the next mini-batch, shuffle data in a shard
    assert(self.curr_batch < self.curr_tracker.nbatches)
    self.curr_batch = self.curr_batch + 1
    if self.batch_idx < self.max_batch_idx then
        self.batch_idx = self.batch_idx + 1
        local idx = self.shuffle_idx[self.batch_idx]
        return self.data[idx]
    else
        self.shard_idx = self.shard_idx + 1
        self.data = torch.load(self.curr_tracker.tensorFiles[self.shard_idx])
        assert(#self.data > 0)
        self.shuffle_idx = torch.randperm(#self.data)
        self.batch_idx = 1
        self.max_batch_idx = #self.data
        local idx = self.shuffle_idx[self.batch_idx]
        return self.data[idx]
    end
end

function DataLoader:_makeVocab(textFile, vocabSize)
    --[[ Create vocabulary with maximum vocabSize words.
    Parameters:
    - `textFile` : source or target file, tokenized, lowercased
    - `vocabSize` : the number of top frequent words in the textFile
    --]]
    local _start_vocab = self._start_vocab
    local word_freq = {}
    print('reading in ' .. textFile)
    for line in io.lines(textFile) do
        for w in line:gmatch('%S+') do
            word_freq[w] = (word_freq[w] or 0) + 1
        end
    end

    local words = {}
    for w in pairs(word_freq) do
        words[#words + 1] = w
    end

    -- sort by frequency
    table.sort(words, function(w1, w2)
        return word_freq[w1] > word_freq[w2] or
            word_freq[w1] == word_freq[w2] and w1 < w2
    end)

    local w2idx = {}

    for i, w in ipairs(_start_vocab) do
        w2idx[w] = i
    end

    local offset = #_start_vocab
    for i = 1, vocabSize - offset do
        local w = words[i]
        w2idx[w] = i + offset
    end

    -- free memory
    collectgarbage()
    return w2idx
end

function DataLoader:_createShard(buckets, tensorPrefix, shard_idx, tracker)
    local shard = {}
    for idx, bucket in pairs(buckets) do
        -- make a big torch.IntTensor matrix
        local bx = torch.IntTensor(bucket):split(self.batchSize, 1)
        buckets[idx] = nil -- free memory
        for i = 1, #bx do
            table.insert(shard, bx[i])
        end
    end

    local tensorFile = string.format('%s.shard_%d.t7', tensorPrefix, shard_idx)
    table.insert(tracker.tensorFiles, tensorFile)
    torch.save(tensorFile, shard)

    tracker.nbatches = tracker.nbatches + #shard
end

function DataLoader:text2Tensor(textFile, tensorPrefix, shardSize, tracker)
    --[[Load source and target text file and save to tensor format.
        If the files are too large, process a shard of shardSize sentences at a time
    --]]

    local vocabSize = self.vocabSize
    local vocab = self.vocab

    -- helper
    local batchSize = self.batchSize
    local shard_idx = 0
    local count = 0 -- sentence counter
    local buckets = {}
    local nbatches = 0

    local diff = 1 -- maximum different in length of the target

    for line in io.lines(textFile) do
        count = count + 1

        local tokens = stringx.split(line)

        local length = #tokens + diff - (#tokens % diff)

        local widx, w

        -- add BOS and EOS to target
        local ids = {vocab[self._bos]}
        for _, w in ipairs(tokens) do
            widx = vocab[w] or vocab[self._unk]
            table.insert(ids, widx)
        end

        table.insert(ids, vocab[self._eos])
        -- add PAD to the end after EOS
        for i = 1, length - #tokens do
            table.insert(ids, vocab[self._pad])
        end

        -- put sentence pair to corresponding bucket
        buckets[length] = buckets[length] or {}
        table.insert(buckets[length], ids)

        if count % shardSize == 0 then
            shard_idx = shard_idx + 1
            self:_createShard(buckets, tensorPrefix, shard_idx, tracker)
            buckets = {}
        end
    end

    if count % shardSize  > 1 then
        -- process the remaining
        shard_idx = shard_idx + 1
        self:_createShard(buckets, tensorPrefix, shard_idx, tracker)
    end
end
