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
    --[[ Map tensor if indices to surface words

    Parameters:
    - `x` : tensor input, 1D for now
    - `id2word` : mapping dictionary
    - `_ignore` : dictionary of ignored words such as <s>, </s>

    Return:
    - `s` : string
    -]]
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

function utils.topk(k, mat, index)
    --[[ find top k elements in the matrix

    Parameters:
    - `k` : number of elements
    - `mat` : a matrix
    - `index: a matrix of LongTensor, same size as mat and store corresponding index
    --]]
    local res, flat_idx = mat:view(-1):topk(k, true)

    if mat:type() == 'torch.CudaTensor' then
        flat_idx = flat_idx:long() -- need long tensor here
    end

    flat_idx:add(-1)
    local dim2 = mat:size(2)
    local row = flat_idx:clone():div(dim2):add(1)
    local _idx = flat_idx:mod(dim2):add(1):view(-1,1):typeAs(mat)
    local col = index:index(1, row):gather(2, _idx)

    if mat:type() == 'torch.CudaTensor' then
        row = row:type('torch.CudaTensor')
    else
        col = col:long()
    end

    return res, row, col
end


function utils.find_topk(k, mat)
    --[[ find top k elements in the matrix

    Parameters:
    - `k` : number of elements
    - `mat` : a matrix

    Return:
    - `value` : k values
    - `row` : corresponding row
    - `col` : corresponding column
    --]]
    local res, idx = mat:view(-1):topk(k, true)
    local dim2 = mat:size(2)

    idx:add(-1)
    local row = idx:clone():div(dim2):add(1)
    local col = idx:mod(dim2):add(1)
    return res, row, col
end

function utils.reverse(t, dim)
    --[[ Reverse tensor along the specified dimension

    Parameters:
    - `t` : tensor to be reversed
    - `dim` : integer 1 or 2

    Return:
    - `rev_t` : reversed tensor
    --]]
    local dtype = 'torch.LongTensor'
    if t:type() == 'torch.CudaTensor' then dtype = 'torch.CudaTensor' end

    local rev_idx
    if not dim then
        -- this is a 1 D tensor
        rev_idx = torch.range(t:numel(), 1, -1):type(dtype)
        return t:index(1, rev_idx)
    end

    assert(t:dim() == 2)
    if t:size(d) == 1 then return t:clone() end

    rev_idx = torch.range(t:size(dim), 1, -1):type(dtype)

    return t:index(dim, rev_idx)
end

function utils.scale_clip(v, max_norm)
    local norm = v:norm()
    if norm > max_norm then
        v:div(norm/max_norm)
    end
end

return utils
