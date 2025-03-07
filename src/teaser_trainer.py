import torch.utils.data
import torch.nn.functional as F
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from src.teaser_encoder import TeaserEncoder
from src.teaser_generator import TeaserGenerator
from src.base_trainer import BaseTrainer 
import numpy as np
import src.utils.utils as utils
import src.utils.masking as masking_utils
import copy
import os.path as osp

def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

class TeaserTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if self.config.arch.enable_fuse_generator:
            self.teaser_generator = TeaserGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5)
            
        self.teaser_encoder = TeaserEncoder(n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        
        self.flame = FLAME(n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)

        self.renderer = Renderer(render_full_head=False)
        self.setup_losses()

        self.templates = utils.load_templates()
            
        lmk203_path = 'datasets/preprocess_scripts/203_landmark_embeding_new.npz'
        if osp.exists(lmk203_path):
            lmk_embeddings_203 = np.load(lmk203_path)
            self.lmk_203_front_indices = lmk_embeddings_203['landmark_front_indices'].tolist()
            self.lmk_203_left_indices  = lmk_embeddings_203['landmark_left_indices'].tolist()
            self.lmk_203_right_indices = lmk_embeddings_203['landmark_right_indices'].tolist()


    def step1(self, batch):
        B, C, H, W = batch['img'].shape

        encoder_output = self.teaser_encoder(batch['img'])
        

        with torch.no_grad():
            base_output = self.base_encoder(batch['img'])

        flame_output = self.flame.forward(encoder_output)
        
                
        renderer_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'],
                                                landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        rendered_img = renderer_output['rendered_img']
        
        
        
        # print(flame_output['landmarks_mp'].shape)
        flame_output.update(renderer_output)
 
        # ---------------- losses ---------------- #
        losses = {}

        img = batch['img']

        # ---------------- landmark based losses ---------------- #
        valid_landmarks = batch['flag_landmarks_fan']
        losses['landmark_loss_fan'] = 0 if torch.sum(valid_landmarks) == 0 else F.mse_loss(flame_output['landmarks_fan'][valid_landmarks,:17], batch['landmarks_fan'][valid_landmarks,:17])
        

        losses['landmark_loss_mp'] = F.mse_loss(flame_output['landmarks_mp'], batch['landmarks_mp'])
        
        #-----------our 203 landmark loss-------------#
        
        out_lst_gt = self.landmark_torch_model(img)
        out_pts_gt = out_lst_gt[2] 
        landmark_gt = out_pts_gt[0].reshape(16,-1, 2)  #（16，203，2）
        landmark_gt = landmark_gt * 2 -1

        
        transformed_203_landmarks = batch_orth_proj(flame_output['landmarks_203'][..., :2],encoder_output['cam']) #(-1，1）
        transformed_203_landmarks[:,:,1] = -1 * transformed_203_landmarks[:,:,1]
        
            
        lm_loss_203 = 0.
        ld_203_with_angle_gt = []
        ld_203_with_angle_flame = []
        for i in range(img.shape[0]):
            if encoder_output['pose_params'][i][1] < -0.05:
                SELECTED_IDS = torch.tensor(self.lmk_203_left_indices).to(self.config.device)
                ld_203_i_flame = torch.index_select(transformed_203_landmarks[i],dim=0,index=SELECTED_IDS)
                ld_203_i_gt = torch.index_select(landmark_gt[i],dim=0,index=torch.tensor(SELECTED_IDS).to(self.config.device))
                lm_loss_203_temp = F.mse_loss(ld_203_i_flame.unsqueeze(0), ld_203_i_gt.unsqueeze(0))
                lm_loss_203 += lm_loss_203_temp
                ld_203_with_angle_gt.append(ld_203_i_gt)
                ld_203_with_angle_flame.append(ld_203_i_flame)
            elif encoder_output['pose_params'][i][1] > 0.05:
                SELECTED_IDS = torch.tensor(self.lmk_203_right_indices).to(self.config.device)
                ld_203_i_flame = torch.index_select(transformed_203_landmarks[i],dim=0,index=SELECTED_IDS)
                ld_203_i_gt = torch.index_select(landmark_gt[i],dim=0,index=torch.tensor(SELECTED_IDS).to(self.config.device))
                lm_loss_203_temp = F.mse_loss(ld_203_i_flame.unsqueeze(0), ld_203_i_gt.unsqueeze(0))
                lm_loss_203 += lm_loss_203_temp
                ld_203_with_angle_gt.append(ld_203_i_gt)
                ld_203_with_angle_flame.append(ld_203_i_flame)
            else:
                SELECTED_IDS = torch.tensor(self.lmk_203_front_indices).to(self.config.device)
                ld_203_i_flame = torch.index_select(transformed_203_landmarks[i],dim=0,index=SELECTED_IDS)
                ld_203_i_gt = torch.index_select(landmark_gt[i],dim=0,index=torch.tensor(SELECTED_IDS).to(self.config.device))
                lm_loss_203_temp = F.mse_loss(ld_203_i_flame.unsqueeze(0), ld_203_i_gt.unsqueeze(0))
                lm_loss_203 += lm_loss_203_temp
                ld_203_with_angle_gt.append(ld_203_i_gt)
                ld_203_with_angle_flame.append(ld_203_i_flame)
                
        losses['landmark_loss_203'] = lm_loss_203 / 16.
        

        #  ---------------- regularization losses ---------------- # 
        if self.config.train.use_base_model_for_regularization:
            with torch.no_grad():
                base_output = self.base_encoder(batch['img'])
        else:
            base_output = {key[0]: torch.zeros(B, key[1]).to(self.config.device) for key in zip(['expression_params', 'shape_params', 'jaw_params'], [self.config.arch.num_expression, self.config.arch.num_shape, 3])}

        losses['expression_regularization'] = torch.mean((encoder_output['expression_params'] - base_output['expression_params'])**2)
        losses['shape_regularization'] = torch.mean((encoder_output['shape_params'] - base_output['shape_params'])**2)
        losses['jaw_regularization'] = torch.mean((encoder_output['jaw_params'] - base_output['jaw_params'])**2)


        if self.config.arch.enable_fuse_generator:
            masks = batch['mask']

            # # mask out face and add random points inside the face
            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
            
            masked_img = masking_utils.masking_face(img, masks, self.config.train.mask_dilation_radius, rendered_mask=rendered_mask)
            reconstructed_img = self.teaser_generator(torch.cat([rendered_img, masked_img], dim=1), encoder_output['token'])
            
            #region loss
            mask = 1 - batch['mask_mouth'] + 1 - batch['mask_eyes']
            losses['region_reconstruction_loss'] = F.l1_loss(reconstructed_img * mask, img * mask, reduction='none').mean()
            
        
            #add token cycle loss
            reconstructed_encoder_output = self.teaser_encoder(reconstructed_img)
            losses['token_cycle_loss_0'] = F.mse_loss(reconstructed_encoder_output['token'][0], encoder_output['token'][0])
            losses['token_cycle_loss_1'] = F.mse_loss(reconstructed_encoder_output['token'][1], encoder_output['token'][1])
            losses['token_cycle_loss_2'] = F.mse_loss(reconstructed_encoder_output['token'][2], encoder_output['token'][2])
            losses['token_cycle_loss_3'] = F.mse_loss(reconstructed_encoder_output['token'][3], encoder_output['token'][3])
            losses['token_cycle_loss'] = losses['token_cycle_loss_0'] + losses['token_cycle_loss_1'] + losses['token_cycle_loss_2'] + losses['token_cycle_loss_3']
            
            # reconstruction loss
            reconstruction_loss = F.l1_loss(reconstructed_img, img , reduction='none')

            # for visualization
            loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
            losses['reconstruction_loss'] = reconstruction_loss.mean()

            # perceptual loss
            losses['perceptual_vgg_loss'] = self.vgg_loss(reconstructed_img, img)

            # perceptual losses
            # perceptual_losses = 0
            if self.config.train.loss_weights['emotion_loss'] > 0:
                # do not let this gradient flow through the generator
                for param in self.teaser_generator.parameters():
                    param.requires_grad_(False)
                self.teaser_generator.eval()
                reconstructed_img_p = self.teaser_generator(torch.cat([rendered_img, masked_img], dim=1), encoder_output['token'])
                for param in self.teaser_generator.parameters():
                    param.requires_grad_(True)
                self.teaser_generator.train()

                losses['emotion_loss'] = self.emotion_loss(reconstructed_img_p, img, metric='l2', use_mean=False)
                losses['emotion_loss'] = losses['emotion_loss'].mean()
            else:
                losses['emotion_loss'] = 0
        else:
            losses['reconstruction_loss'] = 0
            losses['perceptual_vgg_loss'] = 0
            losses['emotion_loss'] = 0

        #mica can be understood as another way to calculate shape parameter
        if self.config.train.loss_weights['mica_loss'] > 0:
            losses['mica_loss'] = self.mica.calculate_mica_shape_loss(encoder_output['shape_params'], batch['img_mica'])
        else:
            losses['mica_loss'] = 0


        shape_losses = losses['shape_regularization'] * self.config.train.loss_weights['shape_regularization'] + \
                                    losses['mica_loss'] * self.config.train.loss_weights['mica_loss']

        expression_losses = losses['expression_regularization'] * self.config.train.loss_weights['expression_regularization'] + \
                            losses['jaw_regularization'] * self.config.train.loss_weights['jaw_regularization']
        
        landmark_losses = losses['landmark_loss_fan'] * self.config.train.loss_weights['landmark_loss'] + \
                            losses['landmark_loss_mp'] * self.config.train.loss_weights['landmark_loss'] + \
                                losses['landmark_loss_203'] * self.config.train.loss_weights['landmark_loss_203']


        fuse_generator_losses = losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + \
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + \
                                losses['emotion_loss'] * self.config.train.loss_weights['emotion_loss'] + \
                                    losses['token_cycle_loss'] * self.config.train.loss_weights['token_cycle_loss'] + \
                                        losses['region_reconstruction_loss'] * self.config.train.loss_weights['region_reconstruction_loss']
               
   
        loss_first_path = (
            (shape_losses if self.config.train.optimize_shape else 0) +
            (expression_losses if self.config.train.optimize_expression else 0) +
            (landmark_losses) +
            (fuse_generator_losses if self.config.arch.enable_fuse_generator else 0)
        )

        for key, value in losses.items():
            try:
                losses[key] = value.item() if isinstance(value, torch.Tensor) else value
            except Exception as e:
                print(key)
                print('Error in invert loss. Trying again...', e)

        # ---------------- create a dictionary of outputs to visualize ---------------- #
        outputs = {}
        outputs['rendered_img'] = rendered_img
        outputs['vertices'] = flame_output['vertices']
        outputs['img'] = img
        outputs['landmarks_fan_gt'] = batch['landmarks_fan']
        outputs['landmarks_fan'] = flame_output['landmarks_fan']
        outputs['landmarks_mp'] = flame_output['landmarks_mp']
        outputs['landmarks_mp_gt'] = batch['landmarks_mp']
        outputs['landmarks_203'] = ld_203_with_angle_flame  #在不同角度下索引值不一样
        outputs['landmarks_mp_gt'] = batch['landmarks_mp']
        outputs['landmarks_203_gt'] = ld_203_with_angle_gt
    
        
        if self.config.arch.enable_fuse_generator:
            outputs['loss_img'] = loss_img
            outputs['reconstructed_img'] = reconstructed_img
            # outputs['reconstructed_img_swap_expression'] = rc_img_swap_expression
            # outputs['masked_1st_path'] = masked_img

        for key in outputs.keys():
            if key != 'landmarks_203' and key != 'landmarks_203_gt':
                outputs[key] = outputs[key].detach().cpu()

        outputs['encoder_output'] = encoder_output

        return outputs, losses, loss_first_path, encoder_output


        
    # ---------------- second path in the paper, recomposite expression with other parameter, and put them into generator, then calculate cycle loss---------------- #
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        B, C, H, W = batch['img'].shape        
        img = batch['img']
        masks = batch['mask']
        
        # number of multiple versions for the second path
        Ke = self.config.train.Ke
        
        # start from the same encoder output and add noise to expression params
        # hard clone flame_feats
        flame_feats = {}
        for k, v in encoder_output.items():
            tmp = v.clone().detach()
            flame_feats[k] = torch.cat(Ke * [tmp], dim=0)

        # split Ke * B into 4 random groups        
        gids = torch.randperm(Ke * B)
        # 4 groups
        gids = [gids[:Ke * B // 4], gids[Ke * B // 4: 2 * Ke * B // 4], gids[2 * Ke * B // 4: 3 * Ke * B // 4], gids[3 * Ke * B // 4:]] 

        feats_dim = flame_feats['expression_params'].size(1)        

        # ---------------- random expression ---------------- #
        # 1 of 4 Ke - random expressions!  
        param_mask = torch.bernoulli(torch.ones((len(gids[0]), feats_dim)) * 0.5).to(self.config.device)
        
        new_expressions = (torch.randn((len(gids[0]), feats_dim)).to(self.config.device)) * (1 + 2 * torch.rand((len(gids[0]), 1)).to(self.config.device)) * param_mask + flame_feats['expression_params'][gids[0]]
        flame_feats['expression_params'][gids[0]] = torch.clamp(new_expressions, -4.0, 4.0) +  (0 + 0.2 * torch.rand((len(gids[0]), 1)).to(self.config.device)) * torch.randn((len(gids[0]), feats_dim)).to(self.config.device)
        
        # ---------------- permutation of expression ---------------- #
        # 2 of 4 Ke - permutation!    + extra noise 
        flame_feats['expression_params'][gids[1]] = (0.25 + 1.25 * torch.rand((len(gids[1]), 1)).to(self.config.device)) * flame_feats['expression_params'][gids[1]][torch.randperm(len(gids[1]))] + \
                                        (0 + 0.2 * torch.rand((len(gids[1]), 1)).to(self.config.device)) *  torch.randn((len(gids[1]), feats_dim)).to(self.config.device)
        
        # ---------------- template injection ---------------- #
        # 3 of 4 Ke - template injection!  + extra noise
        for i in range(len(gids[2])):
            expression = self.load_random_template(num_expressions=self.config.arch.num_expression)
            flame_feats['expression_params'][gids[2][i],:self.config.arch.num_expression] = (0.25 + 1.25 * torch.rand((1, 1)).to(self.config.device)) * torch.Tensor(expression).to(self.config.device)
        flame_feats['expression_params'][gids[2]] += (0 + 0.2 * torch.rand((len(gids[2]), 1)).to(self.config.device)) * torch.randn((len(gids[2]), feats_dim)).to(self.config.device)

        # ---------------- tweak jaw for all paths ---------------- #
        scale_mask = torch.Tensor([1, .1, .1]).to(self.config.device).view(1, 3) * torch.bernoulli(torch.ones(Ke * B) * 0.5).to(self.config.device).view(-1, 1)
        flame_feats['jaw_params'] = flame_feats['jaw_params']  + torch.randn(flame_feats['jaw_params'].size()).to(self.config.device) * 0.2 * scale_mask
        flame_feats['jaw_params'][..., 0] = torch.clamp(flame_feats['jaw_params'][..., 0] , 0.0, 0.5)
        
        # ---------------- tweak eyelids for all paths ---------------- #
        if self.config.arch.use_eyelids:
            flame_feats['eyelid_params'] += (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'].size()).to(self.config.device)) * 0.25
            flame_feats['eyelid_params'] = torch.clamp(flame_feats['eyelid_params'], 0.0, 1.0)

        # ---------------- zero expression ---------------- #
        # 4 of 4 Ke - zero expression!     
        # let the eyelids to move a lot - extra noise
        flame_feats['expression_params'][gids[3]] *= 0.0
        flame_feats['expression_params'][gids[3]] += (0 + 0.2 * torch.rand((len(gids[3]), 1)).to(self.config.device)) * torch.randn((len(gids[3]), flame_feats['expression_params'].size(1))).to(self.config.device)
        
        flame_feats['jaw_params'][gids[3]] *= 0.0
        flame_feats['eyelid_params'][gids[3]] = torch.rand(size=flame_feats['eyelid_params'][gids[3]].size()).to(self.config.device)        

        flame_feats['expression_params'] = flame_feats['expression_params'].detach()
        flame_feats['pose_params'] = flame_feats['pose_params'].detach()
        flame_feats['shape_params'] = flame_feats['shape_params'].detach()
        flame_feats['jaw_params'] = flame_feats['jaw_params'].detach()
        flame_feats['eyelid_params'] = flame_feats['eyelid_params'].detach()

        # after defining param augmentation, we can render the new faces
        with torch.no_grad():
            flame_output = self.flame.forward(encoder_output)
            rendered_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'])
            flame_output.update(rendered_output)
     
            # render the tweaked face
            flame_output_2nd_path = self.flame.forward(flame_feats)
            renderer_output_2nd_path = self.renderer.forward(flame_output_2nd_path['vertices'], encoder_output['cam'])
            rendered_img_2nd_path = renderer_output_2nd_path['rendered_img'].detach()

            
            tmask_ratio = self.config.train.mask_ratio #* 2.0
                    
            rendered_mask = (rendered_img_2nd_path > 0).all(dim=1, keepdim=True).float()
                
        masked_img_2nd_path = masking_utils.masking_face(img.repeat(Ke, 1, 1, 1), masks.repeat(Ke, 1, 1, 1), self.config.train.mask_dilation_radius, rendered_mask=rendered_mask)
        reconstructed_img_2nd_path = self.teaser_generator(torch.cat((rendered_img_2nd_path, masked_img_2nd_path), dim=1).detach(), encoder_output['token'].detach())
        if self.config.train.freeze_generator_in_second_path:
            reconstructed_img_2nd_path = reconstructed_img_2nd_path.detach()

        
        #put the reconstrucred image into teaser_encoder
        recon_feats = self.teaser_encoder(reconstructed_img_2nd_path.view(Ke * B, C, H, W)) 

        flame_output_2nd_path_2 = self.flame.forward(recon_feats)
        rendered_img_2nd_path_2 = self.renderer.forward(flame_output_2nd_path_2['vertices'], recon_feats['cam'])['rendered_img']

        losses = {}
        
        #calculate cycle loss
        cycle_loss = 1.0 * F.mse_loss(recon_feats['expression_params'], flame_feats['expression_params']) + \
                     10.0 * F.mse_loss(recon_feats['jaw_params'], flame_feats['jaw_params']) + \
                         1.0 * F.mse_loss(recon_feats['token'][0], flame_feats['token'][0])
        
        if self.config.arch.use_eyelids:
            cycle_loss += 10.0 * F.mse_loss(recon_feats['eyelid_params'], flame_feats['eyelid_params'])

        if not self.config.train.freeze_generator_in_second_path:                
            cycle_loss += 1.0 * F.mse_loss(recon_feats['shape_params'], flame_feats['shape_params']) 

        losses['cycle_loss']  = cycle_loss
        loss_second_path = losses['cycle_loss'] * self.config.train.loss_weights.cycle_loss

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value


        # ---------------- visualization struct ---------------- #
        
        outputs = {}
        if batch_idx % self.config.train.visualize_every == 0:
            outputs['2nd_path'] = torch.stack([rendered_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                            #  masked_img_2nd_path.detach().cpu().view(Ke, B, C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W),
                                             reconstructed_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                             rendered_img_2nd_path_2.detach().cpu().view(Ke, B, C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W)], dim=1).reshape(-1, C, H, W)
            
        return outputs, losses, loss_second_path

    def freeze_encoder(self):
        utils.freeze_module(self.teaser_encoder.pose_encoder, 'pose encoder')
        utils.freeze_module(self.teaser_encoder.shape_encoder, 'shape encoder')
        utils.freeze_module(self.teaser_encoder.expression_encoder, 'expression encoder')
        
    def unfreeze_encoder(self):
        if self.config.train.optimize_pose:
            utils.unfreeze_module(self.teaser_encoder.pose_encoder, 'pose encoder')
        
        if self.config.train.optimize_shape:
            utils.unfreeze_module(self.teaser_encoder.shape_encoder, 'shape encoder')
            
        if self.config.train.optimize_expression:
            utils.unfreeze_module(self.teaser_encoder.expression_encoder, 'expression encoder')

    def step(self, batch, batch_idx, losses_AM_l1, losses_AM_per, losses_AM_token_cycle, losses_AM_region, losses_AM_203_landmark, phase='train'):
        # ------- set the model to train or eval mode ------- #
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)
                
        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch)

        if phase == 'train':
            self.optimizers_zero_grad()
            loss_first_path.backward()
            self.optimizers_step(step_encoder=True,  step_fuse_generator=True)
        
        #pahse 2     
        if (self.config.train.loss_weights['cycle_loss'] > 0) and (phase == 'train'):
           
            if self.config.train.freeze_encoder_in_second_path:
                self.freeze_encoder()
            if self.config.train.freeze_generator_in_second_path:
                utils.freeze_module(self.teaser_generator, 'fuse generator')
                    
            outputs2, losses2, loss_second_path = self.step2(encoder_output, batch, batch_idx, phase)
            
            self.optimizers_zero_grad()
            loss_second_path.backward()

            # gradient clip for generator - we want only details to be guided 
            if not self.config.train.freeze_generator_in_second_path:
                torch.nn.utils.clip_grad_norm_(self.teaser_generator.parameters(), 0.1)
            
            self.optimizers_step(step_encoder=not self.config.train.freeze_encoder_in_second_path, 
                                 step_fuse_generator=not self.config.train.freeze_generator_in_second_path)

            losses1.update(losses2)
            outputs1.update(outputs2)

            if self.config.train.freeze_encoder_in_second_path:
                self.unfreeze_encoder()
            
            if self.config.train.freeze_generator_in_second_path:
                utils.unfreeze_module(self.teaser_generator, 'fuse generator')
        
        losses = losses1
        losses_AM_l1.update(losses['reconstruction_loss'])
        losses['reconstruction_loss'] = losses_AM_l1.avg
        losses_AM_per.update(losses['perceptual_vgg_loss'])
        losses['perceptual_vgg_loss'] = losses_AM_per.avg
        losses_AM_token_cycle.update(losses['token_cycle_loss'])
        losses['token_cycle_loss'] = losses_AM_token_cycle.avg
        losses_AM_region.update(losses['region_reconstruction_loss'])
        losses['region_reconstruction_loss'] = losses_AM_region.avg
        losses_AM_203_landmark.update(losses['landmark_loss_203'])
        losses['landmark_loss_203'] = losses_AM_203_landmark.avg
        self.logging(batch_idx, losses, phase)
        
        if phase == 'train':
            self.scheduler_step()

        return outputs1

