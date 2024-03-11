from torch import nn
import torch
import torch.nn.functional as F

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class DWConv(nn.Module):
    def __init__(self, dim, H, W):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.H = H
        self.W = W
    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.H, self.W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

    
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class endecoder2(nn.Module):
    def __init__(self, d_model, heads, dropout, activation, flag):
        super(endecoder2, self).__init__()
        self.activition = _get_activation_fn(activation)
        self.flag = flag
        self.heads = heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v1 = nn.Linear(d_model*2, d_model)
        self.linear_v2 = nn.Linear(d_model*2, d_model)
        self.linear1_1 = nn.Linear(d_model, 2 * d_model)
        self.linear1_2 = nn.Linear(2 * d_model, d_model)
        self.linear2_1 = nn.Linear(d_model, 2 * d_model)
        self.linear2_2 = nn.Linear(2 * d_model, d_model)
        self.att_drop = nn.Dropout(0.1)
        self.att_drop_t = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_1 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, heads)  #  batch_first=True会报错
        self.multihead_attn2 = nn.MultiheadAttention(d_model, heads)  #  batch_first=True会报错
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        rgb1, depth1, _ = x
        rediual_r = rgb1
        rediual_d = depth1
        if self.flag == 1:
            rgb1 = self.norm1_1(rgb1)
            depth1 = self.norm1_2(depth1)
        fusion = torch.cat((rgb1, depth1), dim=2)    
        q = self.linear_q(rgb1)
        k = self.linear_k(depth1)
        v1 = self.linear_v1(fusion)
        v2 = self.linear_v2(fusion)
        
        b, n, c = q.shape
        q = q.view(b, n, c//self.heads, self.heads).permute(0, 3, 1, 2) # b heads n c
        k = k.view(b, n, c//self.heads, self.heads).permute(0, 3, 1, 2) # b heads n c
        v1 = v1.view(b, n, c//self.heads, self.heads).permute(0, 3, 1, 2) # b heads n c
        v2 = v2.view(b, n, c//self.heads, self.heads).permute(0, 3, 1, 2) # b heads n c
        
        
        att_o = q @ k.transpose(2, 3) # b heads n n
        # att = att_o.softmax(dim=-1) # RGB one to T all
        att = torch.softmax(att_o, dim=-1)
        
        # att_t = att_o.softmax(dim=-2) # T one to RGB all
        att_t = torch.softmax(att_o, dim=-2)
        att_t = att_t.permute(0, 1, 3, 2) # transpose
        
        att = self.att_drop(att)
        att_t = self.att_drop_t(att_t)  
        
        result1 = att @ v1  # b heads n c
        result1 = result1.permute(0, 2, 3, 1) # b n c heads
        result1 = result1.reshape(b, n, c)
        
        result2 = att_t @ v2  # b heads n c
        result2 = result2.permute(0, 2, 3, 1) # b n c heads
        result2 = result2.reshape(b, n, c)
        
        res_r = rediual_r + self.dropout1_1(result1)
        res1_r = self.norm2_1(res_r)
        res1_r = self.linear1_2(self.dropout1(self.activition(self.linear1_1(res1_r))))
        res1_r = res_r + self.dropout1_2(res1_r)
        
        res_d = rediual_d + self.dropout2_1(result2)
        res1_d = self.norm2_2(res_d)
        res1_d = self.linear2_2(self.dropout2(self.activition(self.linear2_1(res1_d))))
        res1_d = res_d + self.dropout2_2(res1_d)

        return res1_r, res1_d, res1_d

class interactive(nn.Module):
    def __init__(self, n, d_model, heads, dropout, activation, pos_feats, num_pos_feats, ratio):
        super(interactive, self).__init__()
        self.trans = []
        self.pool_rgb_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_t_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_rgb_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_t_w = nn.AdaptiveAvgPool2d((1, None))
        self.fc_h = nn.Linear(d_model*2, d_model*2)
        self.fc_rgb_h = nn.Linear(d_model*2, d_model)
        self.fc_t_h = nn.Linear(d_model*2, d_model)
        
        self.fc_w = nn.Linear(d_model*2, d_model*2)
        self.fc_rgb_w = nn.Linear(d_model*2, d_model)
        self.fc_t_w = nn.Linear(d_model*2, d_model)
        
        self.conv1 = nn.Conv2d(d_model, d_model//ratio, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(d_model, d_model//ratio, kernel_size=(1, 1), stride=(1, 1))
        self.linear_att = nn.Linear(heads, 1)
        self.conv3 = nn.Conv2d(d_model, d_model//ratio, kernel_size=(1, 1), stride=(1, 1))
        self.ratio = ratio
        flag1 = 0
        for i in range(n):
            if flag1 == 0:
                self.trans.append(endecoder2(d_model//ratio, heads, dropout, activation, 0).to(device=0))
                flag1 += 1
            elif flag1 > 0:
                self.trans.append(endecoder2(d_model//ratio, heads, dropout, activation, 1).to(device=0))

        self.transall = nn.Sequential(*self.trans)
        
        self.conv_fusion = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(d_model, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(d_model, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(d_model, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True)
                                     )
        self.conv_rgb = nn.Sequential(nn.Conv2d(d_model, d_model//ratio, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True)
                                     )
        self.conv_t = nn.Sequential(nn.Conv2d(d_model, d_model//ratio, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True)
                                     )
        

    def forward(self, rgb, depth):
        n, c, h, w = rgb.size()
        o1 = rgb
        o2 = depth
        # obtain row and col feature weight for adjusting location
        rgb_h = self.pool_rgb_h(rgb).squeeze(3)  # n c h 
        t_h = self.pool_t_h(depth).squeeze(3)   # n c h
        concat_h = torch.cat((rgb_h, t_h), dim=1)  # n 2c h
        concat_h = concat_h.permute(0, 2, 1)  # n h 2c
        fc_h = self.fc_h(concat_h)   # n h 2c
        fc_rgb_h = self.fc_rgb_h(fc_h).permute(0, 2, 1)   # n c h
        fc_t_h = self.fc_t_h(fc_h).permute(0, 2, 1)   # n c h
        
        rgb_w = self.pool_rgb_w(rgb).squeeze(2)  # n c w
        t_w = self.pool_t_w(depth).squeeze(2)   # n c w
        concat_w = torch.cat((rgb_w, t_w), dim=1)  # n 2c w
        concat_w = concat_w.permute(0, 2, 1)  # n w 2c
        fc_w = self.fc_w(concat_w)   # n w 2c
        fc_rgb_w = self.fc_rgb_w(fc_w).permute(0, 2, 1)   # n c w
        fc_t_w = self.fc_t_w(fc_w).permute(0, 2, 1)   # n c w
        
        
        fc_rgb_h = torch.mean(fc_rgb_h, dim=1, keepdim=True)  # b 1 h
        fc_rgb_h = torch.softmax(fc_rgb_h, dim=-1)  # n c h 
        fc_t_h = torch.mean(fc_t_h, dim=1, keepdim=True)
        fc_t_h = torch.softmax(fc_t_h, dim=-1)  # n c h 
        
        fc_rgb_w = torch.mean(fc_rgb_w, dim=1, keepdim=True)
        fc_rgb_w = torch.softmax(fc_rgb_w, dim=-1)  # n c w 
        fc_t_w = torch.mean(fc_t_w, dim=1, keepdim=True)
        fc_t_w = torch.softmax(fc_t_w, dim=-1)  # n c w 
        
        rgb2 = rgb.clone()
        depth2 = depth.clone()
        for i in range(rgb.shape[3]): # b c h w
            r1 = rgb[:,:,:,i]   # b c h
            r2 = fc_rgb_h * r1 
            rgb2[:,:,:,i] = r2
            
            t1 = depth[:,:,:,i]
            t2 = fc_t_h * t1
            depth2[:,:,:,i] = t2
            
        for j in range(rgb.shape[2]):
            r1 = rgb[:,:,j,:]
            r2 = r1 * fc_rgb_w
            rgb2[:,:,j,:] = r2
            
            t1 = depth[:,:,j,:]
            t2 = t1 * fc_t_w
            depth2[:,:,j,:] = t2
            
        rgb2 = rgb2 + rgb
        depth2 = depth2 + depth
        rgb1 = self.conv_rgb(rgb2)
        depth1 = self.conv_t(depth2)
        rgb1 = torch.flatten(rgb1, start_dim=2, end_dim=3).permute(0, 2, 1)
        depth1 = torch.flatten(depth1, start_dim=2, end_dim=3).permute(0, 2, 1)

        x = self.transall((rgb1, depth1, depth1))
        rgb3, depth3, att = x   # b n c
        
        fusion = torch.cat((rgb3, depth3), dim=2)
        fusion = fusion.permute(0, 2, 1) # b c n
        fusion = fusion.reshape(n, c, h, w)
        fusion = self.conv_fusion(fusion)  # b c h w

        return fusion

class Attention_block(nn.Module):
    def __init__(self, F_g, ratio=2):
        super(Attention_block, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(F_g // ratio, F_g // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(F_g // 16, F_g // ratio, 1, bias=False)

        self.sigmoid1 = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid2 = nn.Sigmoid()
        
        self.ConvBlock = conv_block(F_g // ratio, ch_out=F_g // ratio)

    def forward(self, x):
 
        # ca to attent channel for fusion feature
        avg_concat = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_concat = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        fusion = avg_concat + max_concat
        ca = self.sigmoid1(fusion) * x
        ca_fusion = ca + x
        # sa to attent spatial for fusion feature
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out, _ = torch.max(ca, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa1 = self.conv2(sa)
        sa2 = self.conv3(sa)
        out = sa1 + sa2
        output = self.sigmoid2(out) * ca_fusion + ca_fusion
        
        output = self.ConvBlock(output)
        
        return output
  

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size1=3, kernel_size2=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size1 in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size1 == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size2, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 + x2
        return self.sigmoid(out)
    

class interactive1_2(nn.Module):
    def __init__(self, d_model, ratio):
        super(interactive1_2, self).__init__()
        self.AG = Attention_block(d_model, ratio)

        self.conv1 = nn.Sequential(nn.Conv2d(d_model, d_model//ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(d_model//ratio, d_model//ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(d_model//ratio, d_model//ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(d_model, d_model // 4, kernel_size=(3, 3), padding=1),
                                     nn.BatchNorm2d(d_model // 4, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True))
                                   
    def forward(self, rgb, depth):
        inp = torch.cat((rgb, depth), dim=1)
        output1 = self.conv1(inp)
        fusion = self.AG(output1)

        return fusion
