import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TeaserGenerator(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=16, res_blocks=3):
        super(TeaserGenerator, self).__init__()

        features = init_features
        self.encoder1 = TeaserGenerator._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  #112
        self.encoder2 = TeaserGenerator._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  #56
        self.encoder3 = TeaserGenerator._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #28
        self.encoder4 = TeaserGenerator._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  #14

        self.bottleneck = TeaserGenerator._block(features * 8, features * 16, name="bottleneck")
        # add multiple (K) resnet blocks as modulelist
        resnet_blocks = []
        for _ in range(res_blocks):
            resnet_blocks.append(
                    ResnetBlock(features * 16, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)
            )
        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )       #28
        self.decoder4 = TeaserGenerator._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )       #56
        self.decoder3 = TeaserGenerator._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )       #112
        self.decoder2 = TeaserGenerator._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )       #224
        self.decoder1 = TeaserGenerator._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        
        self.token_decoder = TokenDecoder()
        
        self.token_zero_conv_4 = nn.Conv2d(256, 256, kernel_size=1)
        self.token_zero_conv_4 = TeaserGenerator.zero_module(self.token_zero_conv_4)  # 权重置零
        self.token_zero_conv_3 = nn.Conv2d(128, 128, kernel_size=1)
        self.token_zero_conv_3 = TeaserGenerator.zero_module(self.token_zero_conv_3)  # 权重置零
        self.token_zero_conv_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.token_zero_conv_2 = TeaserGenerator.zero_module(self.token_zero_conv_2)  # 权重置零
        self.token_zero_conv_1 = nn.Conv2d(32, 32, kernel_size=1)
        self.token_zero_conv_1 = TeaserGenerator.zero_module(self.token_zero_conv_1)  # 权重置零

    def forward(self, x, token_list):

        #if use_mask:
        #    mask = (x[:, 3:] == 0).all(dim=1, keepdim=True).float()

        enc1 = self.encoder1(x,token_list[3])
        enc2 = self.encoder2(self.pool1(enc1),token_list[2])
        enc3 = self.encoder3(self.pool2(enc2),token_list[1])
        enc4 = self.encoder4(self.pool3(enc3),token_list[0])

        bottleneck = self.bottleneck(self.pool4(enc4),token_list[0])
        for resnet_block in self.resnet_blocks:
            bottleneck = resnet_block(bottleneck)
            
            
        #decoder token
        combined_tensor = torch.cat([token_list[3], token_list[2], token_list[1], token_list[0]], dim=1)  #(16,1024)
        dec_token_feature = self.token_decoder(combined_tensor)
    

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4,token_list[0])
        tensor_dec4 = self.token_zero_conv_4(dec_token_feature[0])  #(256,28,28)
        dec4 = dec4 + tensor_dec4
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3,token_list[1])    
        tensor_dec3 = self.token_zero_conv_3(dec_token_feature[1])  #(128,56,56)
        dec3 = dec3 + tensor_dec3
        
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2,token_list[2])
        tensor_dec2 = self.token_zero_conv_2(dec_token_feature[2])  #(64,112,112)
        dec2 = dec2 + tensor_dec2
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1,token_list[3])
        tensor_dec1 = self.token_zero_conv_1(dec_token_feature[3]) #(32,224,224)
        dec1 = dec1 + tensor_dec1

        out = torch.sigmoid(self.conv(dec1))

        # do this better!
        #if use_mask:
        #    out = out[:, :3] * mask + out[:, 3:] * (1 - mask)
        #else:
        #    out = out[:,:3]

        return out
    
    def zero_module(module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module
    
    def _block(in_channels, features, name):
        return AdainBlock(in_channels, features, 256)
    # @staticmethod
    # def _block(in_channels, features, name):
    # @staticmethod
    # def _block(in_channels, features, name):
    #     return nn.Sequential(
    #         OrderedDict(
    #             [
    #                 (
    #                     name + "conv1",
    #                     nn.Conv2d(
    #                         in_channels=in_channels,
    #                         out_channels=features,
    #                         kernel_size=3,
    #                         padding=1,
    #                         bias=False,
    #                     ),
    #                 ),
    #                 (name + "norm1", nn.BatchNorm2d(num_features=features)),
    #                 (name + "relu1", nn.ReLU(inplace=True)),
    #                 (
    #                     name + "conv2",
    #                     nn.Conv2d(
    #                         in_channels=features,
    #                         out_channels=features,
    #                         kernel_size=3,
    #                         padding=1,
    #                         bias=False,
    #                     ),
    #                 ),
    #                 (name + "norm2", nn.BatchNorm2d(num_features=features)),
    #                 (name + "relu2", nn.ReLU(inplace=True)),
    #             ]
    #         )
    #     )

class AdainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, token_len):
        super(AdainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.adain1 = ADAIN(out_channels, token_len)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.adain2 = ADAIN(out_channels, token_len)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, token):
        out = self.conv1(x)
        out = self.norm1(out)
        # print('-----------')
        # print(token.shape)
        out = self.adain1(out, token)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.adain2(out, token)
        out = self.relu2(out)
        return out



class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class TokenDecoder(nn.Module):
    def __init__(self):
        super(TokenDecoder, self).__init__()
        
        self.decoder_1 = nn.Sequential(
            # 从 1024 维向量解码到 (256, 28, 28)
            nn.Linear(1024, 256 * 14 * 14),
            nn.Unflatten(1, (256, 14, 14)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.decoder_2 = nn.Sequential(
            # 从 (256, 28, 28) 解码到 (128,56,56)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.decoder_3 = nn.Sequential(
            # 从 (128, 56, 56) 解码到 (64, 112, 112)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.decoder_4 = nn.Sequential(
            # 从 (64, 112, 112) 解码到 (32, 224, 224)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 最终输出 (32, 224, 224)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = []
        # print('********')
        # print(x.shape)
        # 保存四个主要特征
        features.append(self.decoder_1(x))
        features.append(self.decoder_2(features[-1]))
        features.append(self.decoder_3(features[-1]))
        features.append(self.decoder_4(features[-1]))
        
        return features