--[[
DataLoader: process large bitext data
--]]
require 'torch'
local DataLoader = torch.class('DataLoader')
local _ = require 'moses'

function DataLoader:__init(config)
    self._GO = "<s>"
    self._EOS = "</s>"
    self._UNK = "<unk>"
    self._PAD = '<pad>'
    self._START_VOCAB = {self._GO, self._EOS, self._UNK, self._PAD}

    local langs = {config.src, config.trg}

    -- data path

    local trainFiles = _.map(langs, function(i, ext)
        return path.join(config.dataDir, string.format("%s.%s", config.trainPrefix, ext))
    end)

    local validFiles = _.map(langs, function(i, ext)
        return path.join(config.dataDir, string.format("%s.%s", config.validPrefix, ext))
    end)

    -- helper
    local vocabFile = path.join(config.dataDir, 'vocab.t7')
    -- auxiliary file to store additional information about chunks
    local indexFile = path.join(config.dataDir, 'index.t7')

    self.srcVocabSize = config.srcVocabSize
    self.trgVocabSize = config.trgVocabSize
    self.batchSize = config.batchSize

    self.vocab = {}
    self.tracker = {train = {tensorFiles = {}, nBatch = 0},
                    valid = {tensorFiles = {}, nBatch = 0}}

    local trainPrefix = path.join(config.dataDir, config.trainPrefix)
    local validPrefix = path.join(config.dataDir, config.validPrefix)

    if not path.exists(vocabFile) then
        print('run pre-processing, one-time setup!')
        print('creating source vocabulary ...')
        self.vocab[1] = self:makeVocab(trainFiles[1], self.srcVocabSize)

        print('creating target vocabulary ...')
        self.vocab[2] = self:makeVocab(trainFiles[2], self.trgVocabSize)
        torch.save(vocabFile, self.vocab)

        print('create training tensor files...')
        self:text2Tensor(trainFiles, trainPrefix,
            config.chunkSize, self.tracker["train"])

        print('create validation tensor files...')
        self:text2Tensor(validFiles, validPrefix,
            config.chunkSize, self.tracker["valid"])

        torch.save(indexFile, self.tracker)
    else
        self.vocab = torch.load(vocabFile)
        self.tracker = torch.load(indexFile)
    end

end


function DataLoader:read(mode)
    -- shuffle training chunks
    assert(mode == "train" or mode == "valid")
    self.curTracker = self.tracker[mode]
    self.curTracker.tensorFiles = _.shuffle(self.curTracker.tensorFiles)
    self.chunkIdx = 0
    self.batchIdx = 0
    self.curBatch = 0
    self.maxBatchIdx = -1
end

function DataLoader:nBatch()
    return self.curTracker.nBatch
end

function DataLoader:nextBatch()
    --return the next mini-batch, shuffle data in a chunk
    assert(self.curBatch < self.curTracker.nBatch)
    self.curBatch = self.curBatch + 1
    if self.batchIdx < self.maxBatchIdx then
        self.batchIdx = self.batchIdx + 1
        local idx = self.shuffledIdx[self.batchIdx]
        return self.data[idx]
    else
        self.chunkIdx = self.chunkIdx + 1
        self.data = torch.load(self.curTracker.tensorFiles[self.chunkIdx])
        assert(#self.data > 0)
        self.shuffledIdx = torch.randperm(#self.data)
        self.batchIdx = 1
        self.maxBatchIdx = #self.data
        local idx = self.shuffledIdx[self.batchIdx]
        return self.data[idx]
    end
end

function DataLoader:makeVocab(textFile, vocabSize)
    --[[Create vocabulary with maximum vocabSize words.
    Args:
        - textFile: source or target file, tokenized, lowercased
        - vocabSize: the number of top frequent words in the textFile
    --]]
    local _START_VOCAB = self._START_VOCAB
    local wordFreq = {}
    print('reading in ' .. textFile)
    for line in io.lines(textFile) do
        for w in line:gmatch("%S+") do
            wordFreq[w] = (wordFreq[w] or 0) + 1
        end
    end

    local words = {}
    for w in pairs(wordFreq) do
        words[#words + 1] = w
    end

    -- sort by frequency
    table.sort(words, function(w1, w2)
        return wordFreq[w1] > wordFreq[w2] or
            wordFreq[w1] == wordFreq[w2] and w1 < w2
    end)

    local wordIdx = {}

    for i, w in ipairs(_START_VOCAB) do
        wordIdx[w] = i
    end

    local offset = #_START_VOCAB
    for i = 1, vocabSize - offset do
        local w = words[i]
        wordIdx[w] = i + offset
    end

    -- free memory
    collectgarbage()
    return wordIdx
end

function DataLoader:_createChunk(buckets, tensorFile)
    local chunks = {}
    for _bucketId, bucket in pairs(buckets) do
        -- make a big torch.IntTensor matrix
        local bx = torch.IntTensor(bucket.source):split(self.batchSize, 1)
        local by = torch.IntTensor(bucket.target):split(self.batchSize, 1)
        buckets[_bucketId] = nil -- free memory
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
    local chunkIdx = 0
    local count = 0 -- sentence counter
    local buckets = {}
    local nBatch = 0

    local diff = 5 -- maximum different in length of the target

    for source, target in seq.zip(unpack(files)) do
        count = count + 1

        local srcTokens = stringx.split(source)
        local trgTokens = stringx.split(target)
        local trgLength = #trgTokens + diff - (#trgTokens % diff)
        local _bucketId =  string.format('%d|%d', #srcTokens, trgLength)

        local tokenIdx, token
        -- reverse the source sentence
        local revSrcIds = {}
        for i = #srcTokens, 1, -1 do
            token = srcTokens[i]
            tokenIdx = srcVocab[token] or srcVocab[self._UNK]
            table.insert(revSrcIds, tokenIdx)
        end

        -- pad GO and EOS to target
        local trgIds = {trgVocab[self._GO]}
        for _, token in ipairs(trgTokens) do
            tokenIdx = trgVocab[token] or trgVocab[self._UNK]
            table.insert(trgIds, tokenIdx)
        end
        table.insert(trgIds, trgVocab[self._EOS])
        -- add PAD to the end after <EOS>
        for i = 1, trgLength - #trgTokens do
            table.insert(trgIds, trgVocab[self._PAD])
        end

        -- put sentence pairs to corresponding bucket
        buckets[_bucketId] = buckets[_bucketId] or {source = {}, target = {}}
        local bucket = buckets[_bucketId]
        table.insert(bucket.source, revSrcIds)
        table.insert(bucket.target, trgIds)

        if count % chunkSize == 0 then
            chunkIdx = chunkIdx + 1

            local tensorFile = tensorPrefix .. chunkIdx .. '.t7'
            table.insert(tracker.tensorFiles, tensorFile)
            tracker.nBatch = tracker.nBatch + self:_createChunk(buckets, tensorFile)
            buckets = {}
        end
    end
    if count % chunkSize  > 1 then
        -- process the remaining
        chunkIdx = chunkIdx + 1
        local tensorFile = tensorPrefix .. chunkIdx .. '.t7'
        table.insert(tracker.tensorFiles, tensorFile)
        tracker.nBatch = tracker.nBatch + self:_createChunk(buckets, tensorFile)
    end
end
