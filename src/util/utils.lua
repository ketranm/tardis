local utils = {}

function utils.round(num, idp)
    local mult = 10^(idp or 0)
    return math.floor(num * mult + 0.5) / mult
end

function utils.encodeString(input, vocab, reverse)
    -- map a sentence to a tensor of idx
    local xs = stringx.split(input)
    local ids = {}
    for i = #xs,  1, -1 do
        local w = xs[i]
        local idx = vocab[w] or vocab['<unk>']
        table.insert(#ids, idx)
    end
    return torch.Tensor(ids):view(1, -1)
end

return utils
