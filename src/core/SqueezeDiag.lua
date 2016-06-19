local SqueezeDiag, parent = torch.class('nn.SqueezeDiag', 'nn.Module')
-- squeezing diagonal matrix

function SqueezeDiag:__init(max_diag)
    self.max_diag = max_diag or 1000
    self.mat = torch.range(1, self.max_diag):view(1, -1):repeatTensor(self.max_diag, 1)
    self.diag = torch.eye(self.max_diag):ne(1)
    self.buffer = torch.Tensor()
end

function SqueezeDiag:get_index(mbsz, n)
    assert(n < self.max_diag)
    local mat = self.mat[{{1, n}, {1, n}}]
    local diag = self.diag[{{1, n}, {1, n}}]
    local mask = mat:maskedSelect(diag):long():view(1, n, -1)
    return mask:expand(mbsz, n, n-1)
end


function SqueezeDiag:updateOutput(input)
    -- we squeeze on the last dimension
    assert(input:dim() == 3)
    local T = input:size(2)
    local N = input:size(1)
    assert(input:size(3) == T)
    self.mask = self:get_index(N, T)
    self.output = input:gather(3, self.mask)
    print(self.mask)
    return self.output
end

function SqueezeDiag:backward(input, gradOutput)
    self.buffer:resizeAs(input):zero()
    self.gradInput = self.buffer:scatter(3, self.mask, gradOutput)
    return self.gradInput
end
