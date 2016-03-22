--[[
DataLoader: process large bitext data
--]]
require 'torch'
local DataLoader = torch.class('DataLoader')
local _ = require 'moses'

function DataLoader:__init(kwargs)
    self._GO = "<s>"
    self._EOS = "</s>"
    self._UNK = "<unk>"
    self._START_VOCAB = {self._GO, self._EOS, self._UNK}
    -- data path
    local extension = {kwargs.source, kwargs.target}
    local train_files = _.map(extension, function(i, ext)
        return path.join(kwargs.data_dir, string.format("%s.%s", kwargs.train_file, ext))
    end)

    local valid_files = _.map(extension, function(i, ext)
        return path.join(kwargs.data_dir, string.format("%s.%s", kwargs.valid_file, ext))
    end)
    -- helper
    local vocab_file = path.join(kwargs.data_dir, 'vocab.t7')
    -- auxilary file to store additional infomation about chunks
    local index_file = path.join(kwargs.data_dir, 'index.t7')

    self.train_tensor_prefix = path.join(kwargs.data_dir, 'train')
    self.valid_tensor_prefix = path.join(kwargs.data_dir, 'valid')

    self.source_vocab_size = kwargs.source_vocab_size
    self.target_vocab_size = kwargs.target_vocab_size
    self.batch_size = kwargs.batch_size

    self.vocab = {}
    self.tracker = {train = {tensor_files = {}, num_batches = 0},
                    valid = {tensor_files = {}, num_batches = 0}}
    
    if not path.exists(vocab_file) then
        print('run pre-processing, one-time setup!')
        print('creating source vocabulary ...')
        self.vocab[1] = self:create_vocabulary(train_files[1], self.source_vocab_size)
        --print(self.vocab[1])
        print('creating target vocabulary ...')
        self.vocab[2] = self:create_vocabulary(train_files[2], self.target_vocab_size)
        torch.save(vocab_file, self.vocab)

        print('create training tensor files...')
        self:text_to_tensor(train_files, self.train_tensor_prefix, kwargs.chunk_size, self.tracker["train"])
        print('create validation tensor files...')
        self:text_to_tensor(valid_files, self.valid_tensor_prefix, kwargs.chunk_size, self.tracker["valid"])
        torch.save(index_file, self.tracker)
    else
        self.vocab = torch.load(vocab_file)
        self.tracker = torch.load(index_file)
    end

end


function DataLoader:read(mode)
    -- shuffle training chunks
    assert(mode == "train" or mode == "valid")
    self.cur_tracker = self.tracker[mode]
    self.chunk_idx = 0
    self.batch_idx = 0
    self.cur_batch = 0
    self.batch_idx_max = -1
end


function DataLoader:num_batches()
    return self.cur_tracker.num_batches
end

function DataLoader:next_batch()
    assert(self.cur_batch < self.cur_tracker.num_batches)
    self.cur_batch = self.cur_batch + 1
    if self.batch_idx < self.batch_idx_max then
        self.batch_idx = self.batch_idx + 1
        return self.data[self.batch_idx]
    else
        self.chunk_idx = self.chunk_idx + 1
        self.data = torch.load(self.cur_tracker.tensor_files[self.chunk_idx])
        assert(#self.data > 0)
        self.batch_idx = 1
        self.batch_idx_max = #self.data
        return self.data[1]
    end
end

function DataLoader:create_vocabulary(text_file, vocab_size)
    --[[Create vocabulary with maximum vocab_size words.
    Args:
        - text_file: source or target file, tokenized, lowercased
        - vocab_size: the number of top frequent words in the text_file
    --]]
    local _START_VOCAB = self._START_VOCAB
    local word_freq = {}
    print('reading in ' .. text_file)
    for line in io.lines(text_file) do
        for w in line:gmatch("%S+") do word_freq[w] = (word_freq[w] or 0) + 1 end
    end

    local words = {}
    for w in pairs(word_freq) do
        words[#words + 1] = w
    end

    -- sort by frequency
    table.sort(words, function(w1, w2)
        return word_freq[w1] > word_freq[w2] or word_freq[w1] == word_freq[w2] and w1 < w2
    end)

    local word_idx = {}

    for i, w in ipairs(_START_VOCAB) do
        word_idx[w] = i
    end

    local offset = #_START_VOCAB
    for i = 1, vocab_size - offset do
        local word_ith = words[i]
        word_idx[word_ith] = i + offset
    end

    -- free memory
    collectgarbage()
    return word_idx
end

function DataLoader:_create_chunk(buckets, tensor_out_file)
    local data_set = {}
    for _bucket_id, bucket in pairs(buckets) do
        -- make a big torch.IntTensor matrix
        local bx = torch.IntTensor(bucket.source):split(self.batch_size, 1)
        local by = torch.IntTensor(bucket.target):split(self.batch_size, 1)
        buckets[_bucket_id] = nil -- free memory
        -- sanity check
        assert(#bx == #by)
        for i = 1, #bx do
            assert(bx[i]:size(1) == by[i]:size(1))
            table.insert(data_set, {bx[i], by[i]})
        end
    end
    torch.save(tensor_out_file, data_set)
    return #data_set
end

function DataLoader:text_to_tensor(text_files, tensor_prefix, chunk_size, tracker)
    --[[Load source and target text file and save to tensor format.
        If the files are too large, process a chunk of chunk_size sentences at a time
    --]]

    local files = _.map(text_files, function(i, file)
        return io.lines(file)
    end)

    local source_vocab, target_vocab = unpack(self.vocab)
    -- helper
    local batch_size = self.batch_size
    local chunk_idx = 0
    local count = 0 -- sentence counter
    local buckets = {}
    local num_batches = 0

    for source, target in seq.zip(unpack(files)) do
        count = count + 1

        local source_tokens = stringx.split(source)
        local target_tokens = stringx.split(target)

        local _bucket_id = string.format("%d|%d", #source_tokens, #target_tokens)

        local token_idx, token
        -- reverse the source sentence
        local reversed_source_token_ids = {}
        for i = #source_tokens, 1, -1 do
            token = source_tokens[i]
            token_idx = source_vocab[token] or source_vocab[self._UNK]
            table.insert(reversed_source_token_ids, token_idx)
        end

        -- pad GO and EOS to target
        local target_token_ids = {target_vocab[self._GO]}
        for _, token in ipairs(target_tokens) do
            token_idx = target_vocab[token] or target_vocab[self._UNK]
            table.insert(target_token_ids, token_idx)
        end
        table.insert(target_token_ids, target_vocab[self._EOS])

        -- put sentence pairs to corresponding bucket
        buckets[_bucket_id] = buckets[_bucket_id] or {source = {}, target = {}}
        local bucket = buckets[_bucket_id]
        table.insert(bucket.source, reversed_source_token_ids)
        table.insert(bucket.target, target_token_ids)

        if count % chunk_size == 0 then
            chunk_idx = chunk_idx + 1
            
            local tensor_out_file = tensor_prefix .. chunk_idx .. '.t7'
            table.insert(tracker.tensor_files, tensor_out_file)
            local nbatches = self:_create_chunk(buckets, tensor_out_file)
            tracker.num_batches = tracker.num_batches + nbatches
            -- save chunk
            buckets = {}
        end
    end
    if count % chunk_size  > 1 then
        -- process the remaining
        chunk_idx = chunk_idx + 1  
        local tensor_out_file = tensor_prefix .. chunk_idx .. '.t7'
        table.insert(tracker.tensor_files, tensor_out_file)
        local nbatches = self:_create_chunk(buckets, tensor_out_file)
        tracker.num_batches = tracker.num_batches + nbatches
    end
end
