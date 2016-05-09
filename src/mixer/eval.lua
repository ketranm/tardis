--[[ Smooth BLEU for REINFORCE
Modified from Facebook's MIXER code
--]]

require 'math'
local evals = {}

function evals.compute_hash(vec, n, V)
    --[[ util function to compute BLEU
    hash the first n entry (integers) of the input vector `vec`

    Parameters:
    - `vec` : tensor of (D)
    - `n` : integer, number of entries (such as n-gram)
    - `V` : integer, Vocabulary size

    Returns:
    - `hash` : integer
    --]]

    assert(vec:size(1) >= n)
    local hash = 0
    for cnt = 1, n do
        hash = hash + (vec[cnt] - 1) * math.pow(V, n - cnt)
    end
    return hash + 1 -- start counting from one
end


function evals.get_counts(input, nn, V, skip_id, output)
    --[[ Counting n-gram statistics
    Parameters:
    - `input` : tensor 1D
    - `nn` : integer, n-gram order
    - `V` : integer, vocabulary size
    - `skip_id` : integer, skip counting n-gram of that id
    - `output`: table, optional

    Returns:
    - `output` : table of hashed counts
    --]]

    local sequence_length = input:size(1)
    assert(nn <= sequence_length)
    local out = (output == nil) and {} or output

    for tt = 1, sequence_length - nn + 1 do
        local curr_window = input:narrow(1, tt, nn)

        if skip_id == nil or curr_window:eq(skip_id):sum() == 0 then
            local hash = evals.compute_hash(curr_window, nn, V)
            out[hash] = (out[hash] or 0) + 1
        end
    end
    return out

end


function evals.compute_score(counts_input, counts_target, smoothing_eval)
    -- compute partial bleu score for given counts
    local tot = 0
    local score = 0
    for k, v in pairs(counts_input) do
        tot = tot + v
        if counts_target[k] then
            if counts_input[k] > counts_target[k] then
                score = score + counts_target[k]
            else
                score = score + counts_input[k]
            end
        end
    end

    tot = tot + smoothing_eval
    score = score + smoothing_eval
    score = (tot > 0) and score / tot or 0
    return score
end

return evals
