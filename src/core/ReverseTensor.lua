-- reverse a tensor on given dimension
-- this is useful for bidirectional rnn
-- author: Ke Tran <m.k.tran@uva.nl>
local ReverseTensor, parent = torch.class('nn.ReverseTensor', 'nn.Module')

function ReverseTensor:__init(dim)
    parent.__init(self)
    self.dim = dim or 1
    self.rev_index = torch.LongTensor()
end

function ReverseTensor:updateOutput(input)
    assert(input:dim() >= self.dim, 'invalid input!')
    torch.range(self.rev_index, input:size(self.dim), 1, -1)
    self.output = input:index(self.dim, self.rev_index)
    return self.output
end

function ReverseTensor:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:index(self.dim, self.rev_index)
    return self.gradInput
end
