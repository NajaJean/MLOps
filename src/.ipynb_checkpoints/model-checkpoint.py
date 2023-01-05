from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input layer
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        
        # hidden layer 1
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_hidden)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        
        # hidden layer 2
        self.W_3 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_output), 0))

        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.batchnorm = torch.nn.BatchNorm1d(512)
        
    def forward(self, x):
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)
  #      x = self.batchnorm(x)
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        x = self.dropout(x)
 #       x = self.batchnorm(x)
        x = F.linear(x, self.W_3, self.b_3)
        return F.softmax(x, dim=1)
