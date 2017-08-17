import torch
from torch.autograd import Variable

in1 = torch.rand(128,128)
in2 = torch.rand(128,128)

in1_var = Variable(in1)
in2_var = Variable(in2)

in1 = in1.cuda()
in2 = in2.cuda()
in1_var = in1_var.cuda()
in2_var = in2_var.cuda()

print(type(in1))
print(type(in2))
print(type(in1_var.data))
print(type(in2_var.data))

print(in1.size())
print(in2.size())

print(in1_var.size())
print(in2_var.size())

in1 = in1.view(in1.size()[0],in1.size()[1],1)
in2 = in2.view(in2.size()[0],in2.size()[1],1)

in1_var = in1_var.view(in1_var.size()[0],in1_var.size()[1],1)
in2_var = in2_var.view(in2_var.size()[0],in2_var.size()[1],1)

print(in1.size())
print(in2.size())

print(in1_var.size())
print(in2_var.size())

out = torch.cat([in1,in2],2)
out_var = torch.cat([in1_var,in2_var],2)

print(out.size())
print(out_var.size())
