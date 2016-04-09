-- implementation of multi-bleu.perl script for sentence bleu
local _ = require 'moses'
local BLEU = {}

function BLEU.count_ngrams(words, n)
    if not n then n = 4 end
    local counts = {}
    for k = 1, n do
        for i = 1, #words - k + 1 do
            ngram = table.concat(_.slice(words, i, i + k -1), ' ')
            counts[ngram] = (counts[ngram] or 0) + 1
        end
    end
    return counts 
end


function BLEU.score(target, reference)
    local trg_tokens = stringx.split(target)
    local ref_tokens = stringx.split(reference)
    local trg_length = #trg_tokens
    local ref_length = #ref_tokens

    -- make sure BLEU works for any length
    local nt, nr
    if trg_length >= 4 then nt = 4 else nt = trg_length end
    if ref_length >= 4 then nr = 4 else nr = ref_length end

    local ref_ngrams = BLEU.count_ngrams(ref_tokens, nr)
    local trg_ngrams = BLEU.count_ngrams(trg_tokens, nt)
    local total = {0, 0, 0, 0}
    local correct = {0, 0, 0, 0}

    for ngram, c_ in pairs(trg_ngrams) do
        local n = stringx.count(ngram, ' ') + 1
        total[n] = total[n] + c_
        if ref_ngrams[ngram] then
            if ref_ngrams[ngram] > c_ then
                correct[n] = correct[n] + c_
            else
                correct[n] = correct[n] + ref_ngrams[ngram]
            end
        end
    end

    local brevity_penalty = 1
    local bleu_ngram = {}

    for n = 1, nr do
        if total[n] > 0 then
            bleu_ngram[n] =  correct[n]/total[n]
        else
            bleu_ngram[n] = 0
        end
        if bleu_ngram[n] == 0 then bleu_ngram[n] = 1e-13 end
    end

    if ref_length == 0 then return 0 end
    if trg_length < ref_length then
        brevity_penalty = math.exp(1 - ref_length/trg_length)
    end

    local log_bleu_ngram = _.map(bleu_ngram, function(i, v) return math.log(v) end)
    local bleu = brevity_penalty * math.exp(_.reduce(log_bleu_ngram, function(s, v) return s+v end) / 4)
    return bleu
end

return BLEU
