import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, lora_r=16, lora_alpha=1, lora_dropout=0.0, bias=True):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_bias_lora = bias

        self.Linear = nn.Linear(in_features, out_features)

        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros((self.rank, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, self.rank)))
            self.scaling = self.lora_alpha / self.rank

            self.Linear.weight.requires_grad = False
            self.Linear.bias.requires_grad = False

            if self.use_bias_lora:
                self.lora_bias = nn.Parameter(torch.zeros(out_features))
                nn.init.zeros_(self.lora_bias)

        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(self.lora_dropout)
        else:
            self.dropout = nn.Identity()

        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.rank > 0:
            lora_weight = self.scaling * self.lora_B @ self.lora_A
            lora_bias = self.lora_bias if self.use_bias_lora else 0.0

            result = F.linear(x, self.Linear.weight + lora_weight, bias=self.Linear.bias + lora_bias)
            result = self.dropout(result)
            return result
        else:
            return self.dropout(self.Linear(x))


class LoRA_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., lora_r=4, lora_alpha=1.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r

        # LoRA weights for fc1
        self.lora_fc1_U = nn.Parameter(torch.zeros(lora_r, in_features))
        self.lora_fc1_V = nn.Parameter(torch.zeros(hidden_features, lora_r))
        # LoRA weights for fc2
        self.lora_fc2_U = nn.Parameter(torch.zeros(lora_r, hidden_features))
        self.lora_fc2_V = nn.Parameter(torch.zeros(in_features, lora_r))

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_fc1_U, a=math.sqrt(5))
        nn.init.zeros_(self.lora_fc1_V)
        nn.init.kaiming_uniform_(self.lora_fc2_U, a=math.sqrt(5))
        nn.init.zeros_(self.lora_fc2_V)

        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Forward through fc1
        fc1_out = self.fc1(x)
        lora_fc1_out = (x @ self.lora_fc1_U.t()) @ self.lora_fc1_V.t()
        fc1_out = fc1_out + self.scaling * lora_fc1_out
        fc1_out = self.act(fc1_out)
        fc1_out = self.drop(fc1_out)

        # Forward through fc2
        fc2_out = self.fc2(fc1_out)
        lora_fc2_out = (fc1_out @ self.lora_fc2_U.t()) @ self.lora_fc2_V.t()
        fc2_out = fc2_out + self.scaling * lora_fc2_out
        fc2_out = self.drop(fc2_out)
        return fc2_out