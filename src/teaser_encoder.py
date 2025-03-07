import torch
import torch.nn.functional as F
from torch import nn
import timm


def create_backbone(backbone_name, pretrained=True):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim

class PoseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
              
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_small_minimal_100')
        
        self.pose_cam_layers = nn.Sequential(
            nn.Linear(feature_dim, 6)
        )

        self.init_weights()

    def init_weights(self):
        self.pose_cam_layers[-1].weight.data *= 0.001
        self.pose_cam_layers[-1].bias.data *= 0.001

        self.pose_cam_layers[-1].weight.data[3] = 0
        self.pose_cam_layers[-1].bias.data[3] = 7


    def forward(self, img):
        features = self.encoder(img)[-1]  #(bs,576,7,7)  
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)   #(bs,6)

        outputs = {}

        pose_cam = self.pose_cam_layers(features).reshape(img.size(0), -1)
        outputs['pose_params'] = pose_cam[...,:3]  
        outputs['cam'] = pose_cam[...,3:]

        return outputs


class ShapeEncoder(nn.Module):
    def __init__(self, n_shape=300) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')

        self.shape_layers = nn.Sequential(
            nn.Linear(feature_dim, n_shape)
        )

        self.init_weights()


    def init_weights(self):
        self.shape_layers[-1].weight.data *= 0
        self.shape_layers[-1].bias.data *= 0


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        parameters = self.shape_layers(features).reshape(img.size(0), -1)

        return {'shape_params': parameters}


class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')
        
        self.expression_layers = nn.Sequential( 
            nn.Linear(feature_dim, n_exp+2+3) # num expressions + jaw(上下颚) + eyelid（眼睑）
        )

        self.n_exp = n_exp
        self.init_weights()


    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.expression_layers(features).reshape(img.size(0), -1)

        outputs = {}

        outputs['expression_params'] = parameters[...,:self.n_exp]
        outputs['eyelid_params'] = torch.clamp(parameters[...,self.n_exp:self.n_exp+2], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,self.n_exp+2].unsqueeze(-1)), 
                                           torch.clamp(parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)

        return outputs

class TokenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
              
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_small_minimal_100')
        feature_dim_list = [576, 192, 384, 1024]
        self.token_layers = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim_list[i], 256)) for i in range(4)])
        
        self.init_weights()

    def init_weights(self):
        for layer in self.token_layers:
            layer[-1].weight.data *= 0.001
            layer[-1].bias.data *= 0.001

            layer[-1].weight.data[3] = 0
            layer[-1].bias.data[3] = 7


    def forward(self, img):
        # print('-----------')
        # print(self.encoder(img)[-1].shape) #(576,7,7) 576
        # print(self.encoder(img)[-2].shape) #(48,14,14) 192
        # print(self.encoder(img)[-3].shape) #(24,28,28) 384
        # print(self.encoder(img)[-4].shape) #(16,56,56) 1024
        # print(self.encoder(img)[-5].shape) #(16,112,112)
        # print(len(self.encoder(img)))
        token_list = []
        features = self.encoder(img)
        feature = features[-1]  #(bs,576,7,7)  
        feature = F.adaptive_avg_pool2d(feature, (1, 1)).squeeze(-1).squeeze(-1)   #(bs,256)
        token = self.token_layers[0](feature).reshape(img.size(0), -1)
        token_list.append(token)
        for i in range(2,5):
            feature = features[-i]
            # print('------*******------')
            # print(feature.shape)
            feature = F.adaptive_avg_pool2d(feature, (2**(i-1), 2**(i-1))).flatten(start_dim=1)
            # print(i)
            # print(feature.shape)
            token = self.token_layers[i-1](feature).reshape(img.size(0), -1)
            token_list.append(token)
        stacked_tensors = torch.stack(token_list, dim=0)
        return stacked_tensors

class TeaserEncoder(nn.Module):
    def __init__(self, n_exp=50, n_shape=300) -> None:
        super().__init__()

        self.pose_encoder = PoseEncoder()

        self.shape_encoder = ShapeEncoder(n_shape=n_shape)

        self.expression_encoder = ExpressionEncoder(n_exp=n_exp) 
        
        self.token_encoder = TokenEncoder()

    def forward(self, img):
        pose_outputs = self.pose_encoder(img)
        shape_outputs = self.shape_encoder(img)
        expression_outputs = self.expression_encoder(img)
        token_outputs = self.token_encoder(img)

        outputs = {}
        outputs.update(pose_outputs)
        outputs.update(shape_outputs)
        outputs.update(expression_outputs)
        outputs['token'] = token_outputs

        return outputs
