import torch
import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8,
                               kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8,
                               kernel_size=3, stride=1, padding=1, dilation=1, groups=8)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.diconv4 = nn.Conv3d(in_channels=8, out_channels=16,
                                 kernel_size=3, stride=1, padding=2, dilation=2, groups=8)
        self.diconv5 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=2, dilation=2, groups=4)
        self.diconv6 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=2, dilation=2, groups=2)
        self.pool7 = nn.MaxPool3d(3, 2, 1)

    def forward(self, inputs):
      a1 = self.conv1(inputs)
      a2 = self.conv2(a1)
      a3 = self.pool3(a2)
      a4 = self.diconv4(a3)
      a5 = self.diconv5(a4)
      a6 = self.diconv6(a5)
      a7 = self.pool7(a6)
      return a2, a6, a7


class BRB(torch.nn.Module):
    def __init__(self, num_feats=8):
        super(BRB, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels=num_feats, out_channels=num_feats, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=num_feats//2),
            nn.BatchNorm3d(num_feats),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=num_feats, out_channels=num_feats, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=num_feats//2),
            nn.BatchNorm3d(num_feats),
        )

    def forward(self, inputs):
      a1 = self.layers(inputs)
      a2 = a1 + inputs
      a3 = nn.functional.relu(a2)
      return a3


class GRB(torch.nn.Module):
    def __init__(self, num_feats=8):
        super(GRB, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels=num_feats, out_channels=num_feats, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=num_feats//2),
            nn.BatchNorm3d(num_feats),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=num_feats, out_channels=num_feats, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=num_feats//2),
            nn.BatchNorm3d(num_feats),
        )

    def forward(self, inputs):
      a1 = self.layers(inputs)
      a2 = a1 + inputs + (inputs * torch.tanh(inputs))
      a3 = nn.functional.relu(a2)
      return a3


class CCP(torch.nn.Module):
    def __init__(self):
        super(CCP, self).__init__()

        self.diconv1 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=30, dilation=30, groups=1)
        self.diconv2 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=24, dilation=24, groups=1)
        self.diconv3 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=18, dilation=18, groups=1)
        self.diconv4 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=12, dilation=12, groups=1)
        self.diconv5 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=6, dilation=6, groups=1)
        self.diconv6 = nn.Conv3d(in_channels=16, out_channels=16,
                                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

        self.conv1 = nn.Conv3d(in_channels=16, out_channels=8,
                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=8,
                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=8,
                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=8,
                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=8,
                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv6 = nn.Conv3d(in_channels=16, out_channels=8,
                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1)

        self.brb2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=1,
                      stride=1, padding=0, dilation=1, groups=1),
            BRB())
        self.brb3 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=1,
                      stride=1, padding=0, dilation=1, groups=1),
            BRB())
        self.brb4 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=1,
                      stride=1, padding=0, dilation=1, groups=1),
            BRB())
        self.brb5 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=1,
                      stride=1, padding=0, dilation=1, groups=1),
            BRB())
        self.brb6 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=1,
                      stride=1, padding=0, dilation=1, groups=1),
            BRB())

    def forward(self, inputs):
      x1 = self.conv1(self.diconv1(inputs))
      x2 = self.conv2(self.diconv2(inputs))
      x3 = self.conv3(self.diconv3(inputs))
      x4 = self.conv4(self.diconv4(inputs))
      x5 = self.conv5(self.diconv5(inputs))
      x6 = self.conv6(self.diconv6(inputs))
      out = self.brb6(
          self.brb5(self.brb4(self.brb3(self.brb2(x1 + x2) + x3) + x4) + x5) + x6)
      return out


class GRR(torch.nn.Module):
    def __init__(self):
        super(GRR, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=8, out_channels=16,
                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1)

        self.block2 = nn.Sequential(
            GRB(num_feats=16),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=3,
                               stride=2, padding=1, dilation=1, groups=1, output_padding=1),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=1)
        )

        self.block3 = nn.Sequential(
            GRB(num_feats=16),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=3,
                               stride=2, padding=1, dilation=1, groups=1, output_padding=1),
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=1)
        )

        self.block4 = nn.Sequential(
            GRB(num_feats=8),
            nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=4),
            nn.Conv3d(in_channels=32, out_channels=12, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=4)
        )

    def forward(self, inputs, encoded_feats_1x=None, encoded_feats_2x=None, encoded_feats_4x=None):
      a1 = self.conv1(inputs)

      if encoded_feats_4x is not None:
        a1 = a1 + encoded_feats_4x

      a2 = self.block2(a1)

      if encoded_feats_2x is not None:
        a2 = a2 + encoded_feats_2x

      a3 = self.block3(a2)

      if encoded_feats_1x is not None:
        a3 = a3 + encoded_feats_1x

      a4 = self.block4(a3)
      return a4


class CCPNet(torch.nn.Module):
    def __init__(self):
        super(CCPNet, self).__init__()
        self.encoder = Encoder()
        self.ccp = CCP()
        self.grr = GRR()
        self.init_weights()

    def init_weights(self):
        # ----  weights init
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                #nn.init.xavier_uniform_(m.weight.data) 
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


    def forward(self, inputs):
       #1x, 2x, 4x dowsampled extracted feature volume blocks from original volume
      feats_1x, feats_2x, feats_4x = self.encoder(inputs)

      # ccp layer for multi scale feature aggregation
      mfa = self.ccp(feats_4x)

      # guided refinement module. Fuses lower level features from encoder
      per_class_voxels = self.grr(mfa, feats_1x, feats_2x, feats_4x)

      return per_class_voxels
