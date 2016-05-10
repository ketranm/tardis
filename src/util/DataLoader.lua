local DataLoader = torch.class('DataLoader')
local _ = require 'moses'

function DataLoader:__init(config)
    self._bos = '<s>'
    self._eos = '</s>'
    self._unk = '<unk>'
    self._pad = '<pad>'

    self._start_vocab = {self._bos, self._eos, self._unk, self._pad}

    -- just in case
    self.bosidx = 1
    self.eosidx = 2
    self.unkidx = 3
    self.padidx = 4

    local langs = {config.src, config.trg}

    -- data path

    local trainFiles = _.map(langs, function(i, ext)
        return path.join(config.dataDir, string.format('%s.%s', config.trainPrefix, ext))
    end)

    local validFiles = _.map(langs, function(i, ext)
        return path.join(config.dataDir, string.format('%s.%s', config.validPrefix, ext))
    end)

    -- helper
    local vocabFile = path.join(config.dataDir, 'vocab.t7')
    -- auxiliary file to store additional information about chunks
    local indexFile = path.join(config.dataDir, 'index.t7')

    self.srcVocabSize = config.srcVocabSize
    self.trgVocabSize = config.trgVocabSize
    self.batchSize = config.batchSize

    self.vocab = {}
    self.tracker = {train = {tensorFiles = {}, nbatches = 0},
                    valid = {tensorFiles = {}, nbatches = 0}}

    local trainPrefix = path.join(config.dataDir, config.trainPrefix)
    local validPrefix = path.join(config.dataDir, config.validPrefix)

    if not path.exists(vocabFile) then
        print('run pre-processing, one-time setup!')
        print('creating source vocabulary ...')
        self.vocab[1] = self:_make_vocab(trainFiles[1], self.srcVocabSize)

        print('creating target vocabulary ...')
        self.vocab[2] = self:_make_vocab(trainFiles[2], self.trgVocabSize)
        torch.save(vocabFile, self.vocab)

        print('create training tensor files...')
        self:text2Tensor(trainFiles, trainPrefix,
            config.chunkSize, self.tracker['train'])

        print('create validation tensor files...')
        self:text2Tensor(validFiles, validPrefix,
            config.chunkSize, self.tracker['valid'])

        torch.save(indexFile, self.tracker)
    else
        self.vocab = torch.load(vocabFile)
        self.tracker = torch.load(indexFile)
    end

end


function DataLoader:_read(mode)
    -- shuffle training chunks
    assert(mode == 'train' or mode == 'valid')
    self.curr_tracker = self.tracker[mode]
    self.curr_tracker.tensorFiles = _.shuffle(self.curr_tracker.tensorFiles)
    self.chunk_idx = 0
    self.batch_idx = 0
    self.curr_batch = 0
    self.max_batch_idx = -1
end

function DataLoader:nbatches()
    return self.curr_tracker.nbatches
end

function DataLoader:nextBatch()
    --return the next mini-batch, shuffle data in a chunk
    assert(self.curr_batch < self.curr_tracker.nbatches)
    self.curr_batch = self.curr_batch + 1
    if self.batch_idx < self.max_batch_idx then
        self.batch_idx = self.batch_idx + 1
        local idx = self.shuffle_idx[self.batch_idx]
        return self.data[idx]
    else
        self.chunk_idx = self.chunk_idx + 1
        self.data = torch.load(self.curr_tracker.tensorFiles[self.chunk_idx])
        assert(#self.data > 0)
        self.shuffle_idx = torch.randperm(#self.data)
        self.batch_idx = 1
        self.max_batch_idx = #self.data
        local idx = self.shuffle_idx[self.batch_idx]
        return self.data[idx]
    end
end

function DataLoader:read_train()
    self:_read('train')
end

function DataLoader:read_valid()
    self:_read('valid')
end

function DataLoader:_make_vocab(textFile, vocabSize)
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

    local widx = {}

    for i, w in ipairs(_start_vocab) do
        widx[w] = i
    end

    local offset = #_start_vocab
    for i = 1, vocabSize - offset do
        local w = words[i]
        widx[w] = i + offset
    end

    -- free memory
    collectgarbage()
    return widx
end

function DataLoader:_create_chunk(buckets, tensorFile)
    local chunks = {}
    for bidx, bucket in pairs(buckets) do
        -- make a big torch.IntTensor matrix
        local bx = torch.IntTensor(bucket.source):split(self.batchSize, 1)
        local by = torch.IntTensor(bucket.target):split(self.batchSize, 1)
        buckets[bidx] = nil -- free memory
        -- sanity check
        assert(#bx == #by)
        for i = 1, #bx do
            assert(bx[i]:size(1) == by[i]:size(1))
            table.insert(chunks, {bx[i], by[i]})
        end
    end
    torch.save(tensorFile, chunks)
    return #chunks
end

function DataLoader:text2Tensor(textFiles, tensorPrefix, chunkSize, tracker)
    --[[Load source and target text file and save to tensor format.
        If the files are too large, process a chunk of chunkSize sentences at a time
    --]]

    local files = _.map(textFiles, function(i, file)
        return io.lines(file)
    end)

    local srcVocab, trgVocab = unpack(self.vocab)
    -- helper
    local batchSize = self.batchSize
    local chunk_idx = 0
    local count = 0 -- sentence counter
    local buckets = {}
    local nbatches = 0

    local diff = 5 -- maximum different in length of the target

    for source, target in seq.zip(unpack(files)) do
        count = count + 1

        local srcTokens = stringx.split(source)
        local trgTokens = stringx.split(target)
        local trgLength = #trgTokens + diff - (#trgTokens % diff)
        local bidx =  string.format('%d|%d', #srcTokens, trgLength)

        local token_idx, token
        -- reverse the source sentence
        local rev_src_rev_idx = {}
        for i = #srcTokens, 1, -1 do
            token = srcTokens[i]
            token_idx = srcVocab[token] or srcVocab[self._unk]
            table.insert(rev_src_rev_idx, token_idx)
        end

        -- pad GO and EOS to target
        local trg_idx = {trgVocab[self._bos]}
        for _, token in ipairs(trgTokens) do
            token_idx = trgVocab[token] or trgVocab[self._unk]
            table.insert(trg_idx, token_idx)
        end
        table.insert(trg_idx, trgVocab[self._eos])
        -- add PAD to the end after <EOS>
        for i = 1, trgLength - #trgTokens do
            table.insert(trg_idx, trgVocab[self._pad])
        end

        -- put sentence pairs to corresponding bucket
        buckets[bidx] = buckets[bidx] or {source = {}, target = {}}
        local bucket = buckets[bidx]
        table.insert(bucket.source, rev_src_rev_idx)
        table.insert(bucket.target, trg_idx)

        if count % chunkSize == 0 then
            chunk_idx = chunk_idx + 1

            local tensorFile = tensorPrefix .. chunk_idx .. '.t7'
            table.insert(tracker.tensorFiles, tensorFile)
            tracker.nbatches = tracker.nbatches + self:_create_chunk(buckets, tensorFile)
            buckets = {}
        end
    end
    if count % chunkSize  > 1 then
        -- process the remaining
        chunk_idx = chunk_idx + 1
        local tensorFile = tensorPrefix .. chunk_idx .. '.t7'
        table.insert(tracker.tensorFiles, tensorFile)
        tracker.nbatches = tracker.nbatches + self:_create_chunk(buckets, tensorFile)
    end
end
