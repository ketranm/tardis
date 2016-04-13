local utils = {}
local _ = require 'moses'

function utils.round(num, idp)
    local mult = 10^(idp or 0)
    return math.floor(num * mult + 0.5) / mult
end

function utils.encodeString(input, vocab, reverse)
    -- map a sentence to a tensor of idx
    local xs = stringx.split(input)
    if reverse then
        xs = _.reverse(xs)
    end
    local ids = {}
    for _, w  in ipairs(xs) do
        local idx = vocab[w] or vocab['<unk>']
        table.insert(ids, idx)
    end
    return torch.Tensor(ids):view(1, -1)
end

function utils.decodeString(x, id2word, _ignore)
    -- map tensor if indices to surface words
    local words = {}
    for i = 1, x:numel() do
        local idx = x[i]
        if not _ignore[idx] then
            table.insert(words, id2word[idx])
        end
    end
    return table.concat(words, ' ')
end


function utils.flat_to_rc(v, indices, flat_index)
    -- from flat tensor recover row and column index of an element
    local row = math.floor((flat_index - 1)/v:size(2)) + 1
    return row, indices[row][(flat_index - 1) % v:size(2) + 1]
end

return utils
