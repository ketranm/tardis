-- reverse a tensor on given dimension
-- this is useful for bidirectional rnn
-- author: Ke Tran <m.k.tran@uva.nl>
local ReverseTensor, parent = torch.class('nn.ReverseTensor', 'nn.Module')

function ReverseTensor:__init(dim, bprop)
    parent.__init(self)
    self.dim = dim or 1
    self.rev_index = torch.LongTensor()
    self.max_seq = 1000
    self.rev_index = torch.range(self.max_seq, 1, -1)
    if bprop ~= nil then
        self.bprop = bprop
    else
        self.bprop = true
    end
end

function ReverseTensor:updateOutput(input)
    assert(input:dim() >= self.dim, 'invalid input!')
    local D = input:size(self.dim)
    --self.rev_index:range(input:size(self.dim), 1, -1)
    self.output = input:index(self.dim, self.rev_index:narrow(1, self.max_seq - D + 1, D))
    return self.output
end

function ReverseTensor:updateGradInput(input, gradOutput)
    if self.bprop then
        print('check--->', input:size())
        local D = input:size(self.dim)
        self.gradInput = gradOutput:index(self.dim, self.rev_index:narrow(1, self.max_seq - D + 1, D))
    else
        self.gradInput:resize(0)
    end
    return self.gradInput
end
