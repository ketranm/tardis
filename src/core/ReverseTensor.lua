-- reverse a tensor on given dimension
-- this is useful for bidirectional rnn
-- author: Ke Tran <m.k.tran@uva.nl>
local ReverseTensor, parent = torch.class('nn.ReverseTensor', 'nn.Module')

function ReverseTensor:__init(dim, bprop)
    parent.__init(self)
    self.dim = dim or 1
    self.max_seq = 1000
    self.rev_index = torch.range(self.max_seq, 1, -1):long()
    if bprop ~= nil then
        self.bprop = bprop
    else
        self.bprop = true
    end
    self.idx = nil
end

function ReverseTensor:updateOutput(input)
    assert(input:dim() >= self.dim, 'invalid input!')
    local T = input:size(self.dim)
    self.idx = self.rev_index:narrow(1, self.max_seq - T + 1, T)
    self.output = input:index(self.dim, self.idx)
    return self.output
end

function ReverseTensor:updateGradInput(input, gradOutput)
    if self.bprop then
        self.gradInput = gradOutput:index(self.dim, self.idx)
    else
        self.gradInput:resize(0)
    end
    return self.gradInput
end
