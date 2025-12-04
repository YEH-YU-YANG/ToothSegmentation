import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.k = k
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)

    def forward(self):
        device = next(self.parameters()).device
        # as the paper does not infer how to construct a 2xkxk position matrix
        # we assume that it's a kxk matrix for deltax,and a kxk matric for deltay.
        # that is, [[[-1,0,1],[-1,0,1],[-1,0,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]] for kernel = 3
        a_range  = torch.arange(-1*(self.k//2),(self.k//2)+1, device=device).view(1,-1)
        x_position = a_range.expand(self.k,a_range.shape[1])
        b_range = torch.arange((self.k//2),-1*(self.k//2)-1,-1, device=device).view(-1,1)
        y_position = b_range.expand(b_range.shape[0],self.k)
        position = torch.cat((x_position.unsqueeze(0),y_position.unsqueeze(0)),0).unsqueeze(0).float()
        out = self.l2(torch.nn.functional.relu(self.l1(position)))
        return out



class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)

    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)

    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map).transpose(2,1).contiguous()   # [N batch , H_out*Wout, C channel * k*k]
        query_map_unfold = self.unfold(query_map).transpose(2,1).contiguous()    # [N batch , H_out*Wout, C channel * k*k]
        key_map_unfold = key_map_unfold.view(key_map.shape[0],-1, key_map.shape[1], key_map_unfold.shape[-1]//key_map.shape[1])
        query_map_unfold = query_map_unfold.view(query_map.shape[0], -1, query_map.shape[1], query_map_unfold.shape[-1]//query_map.shape[1])
        key_map_unfold = key_map_unfold.transpose(2,1).contiguous()
        query_map_unfold = query_map_unfold.transpose(2,1).contiguous()
        return (key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1]).view(key_map_unfold.shape[0],key_map_unfold.shape[1],key_map_unfold.shape[2],k,k)    #[N batch, C channel, (H-k+1)*(W-k+1), k*k]


def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel,dim=-1)


class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, channels, k, stride=1, padding =0,m=None):
        super(LocalRelationalLayer, self).__init__()
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(channels, self.m)
        self.qmap = KeyQueryMap(channels, self.m)
        self.ac = AppearanceComposability(k, self.padding, self.stride)
        self.gp = GeometryPrior(k, channels//self.m)
        self.unfold = torch.nn.Unfold(k, 1, self.padding, self.stride)
        self.final1x1 = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x):   # x = [N,C,H,W]
        km = self.kmap(x)       # [N,C/m,h,w]
        qm = self.qmap(x)       # [N,C/m,h,w]
        ak = self.ac((km, qm))  # [N,C/m,H_out*W_out, k,k]
        gpk = self.gp()    # [1, C/m,k,k]
        ck = combine_prior(ak, gpk.unsqueeze(2))[:, None, :, :, :]  # [N,1,C/m,H_out*W_out, k,k]
        x_unfold = self.unfold(x).transpose(2,1).contiguous().view(x.shape[0], -1, x.shape[1], self.k*self.k).transpose(2,1).contiguous()
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m, -1, self.k, self.k)     # [N, m, C/m, H_out*W_out, k,k]
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1], -1, self.k*self.k)     #  [N, C,HOUT*WOUT, k*k]
        h_out = (x.shape[2] + 2 * self.padding - 1 * self.k )//  self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * self.k )//  self.stride + 1
        pre_output = torch.sum(pre_output, 3).view(x.shape[0], x.shape[1], h_out, w_out)    # [N, C, H_out*W_out]
        return self.final1x1(pre_output)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention

    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, use_pam = True):
        super(PAM_CAM_Layer, self).__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=in_ch),
            nn.PReLU(),
            PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=in_ch),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.GroupNorm(num_groups=16, num_channels=out_ch)#nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2Net(nn.Module):

    def __init__(self,in_ch=3,out_ch=3):
        super().__init__()

        self.stage1 = RSU7(in_ch,32,64)
        # self.local_1=LocalRelationalLayer(channels=64,k=3,stride=1,padding=1, m=8).cuda()
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        # self.conv1=nn.Conv2d(32,64,1)
        # self.pam_attention_1 = PAM_CAM_Layer(32)
        # self.cam_attention_1 = PAM_CAM_Layer(32, False)

        self.stage2 = RSU6(64,32,128)
        # self.local_2=LocalRelationalLayer(channels=128,k=3,stride=1,padding=1, m=8).cuda()
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        # self.conv2=nn.Conv2d(64,128,1)
        # self.pam_attention_2 = PAM_CAM_Layer(64)
        # self.cam_attention_2 = PAM_CAM_Layer(64, False)

        self.stage3 = RSU5(128,64,256)
        # self.local_3=LocalRelationalLayer(channels=256,k=3,stride=1,padding=1, m=8).cuda()
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        # self.conv3=nn.Conv2d(128,256,1)
        # self.pam_attention_3 = PAM_CAM_Layer(128)
        # self.cam_attention_3 = PAM_CAM_Layer(128, False)

        self.stage4 = RSU4(256,128,512)
        # self.local_4=LocalRelationalLayer(channels=512,k=3,stride=1,padding=1, m=8).cuda()
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        # self.conv4=nn.Conv2d(256,512,1)
        # self.pam_attention_4 = PAM_CAM_Layer(256)
        # self.cam_attention_4 = PAM_CAM_Layer(256, False)

        self.stage5 = RSU4F(512,256,512)
        # self.local_5=LocalRelationalLayer(channels=512,k=3,stride=1,padding=1, m=8).cuda()
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        # self.conv5=nn.Conv2d(256,512,1)
        # self.pam_attention_5 = PAM_CAM_Layer(256)
        # self.cam_attention_5 = PAM_CAM_Layer(256, False)

        # self.stage6 = RSU4F(512,256,512)
        self.conv6=nn.Conv2d(256,512,1)
        self.pam_attention_6 = PAM_CAM_Layer(256)
        self.cam_attention_6 = PAM_CAM_Layer(256, False)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)
        self.local_1d=LocalRelationalLayer(channels=64,k=3,stride=1,padding=1, m=8)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(18,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        # hx1=self.local_1(hx1)
        hx = self.pool12(hx1)
        # print("hx5_o=",hx5.shape)
        # hx1_1=self.pam_attention_1(hx1)
        # hx1_2=self.cam_attention_1(hx1)
        # hx1=hx1_1+hx1_2
        # hx1=self.conv1(hx1)
        # print("hx5=",hx5.shape)

        #stage 2
        hx2 = self.stage2(hx)
        # hx2=self.local_2(hx2)
        hx = self.pool23(hx2)
        # print("hx5_o=",hx5.shape)
        # hx2_1=self.pam_attention_2(hx2)
        # hx2_2=self.cam_attention_2(hx2)
        # hx2=hx2_1+hx2_2
        # hx2=self.conv2(hx2)
        # print("hx5=",hx5.shape)

        #stage 3
        hx3 = self.stage3(hx)
        # hx3=self.local_3(hx3)
        hx = self.pool34(hx3)
        # print("hx5_o=",hx5.shape)
        # hx3_1=self.pam_attention_3(hx3)
        # hx3_2=self.cam_attention_3(hx3)
        # hx3=hx3_1+hx3_2
        # hx3=self.conv3(hx3)
        # print("hx5=",hx5.shape)

        #stage 4
        hx4 = self.stage4(hx)
        # hx4=self.local_4(hx4)
        hx = self.pool45(hx4)
        # print("hx5_o=",hx5.shape)
        # hx4_1=self.pam_attention_4(hx4)
        # hx4_2=self.cam_attention_4(hx4)
        # hx4=hx4_1+hx4_2
        # hx4=self.conv4(hx4)
        # print("hx5=",hx5.shape)

        #stage 5
        hx5 = self.stage5(hx)
        # hx5=self.local_5(hx5)
        hx = self.pool56(hx5)
        # print("hx5_o=",hx5.shape)
        # hx5_1=self.pam_attention_5(hx5)
        # hx5_2=self.cam_attention_5(hx5)
        # hx5=hx5_1+hx5_2
        # hx5=self.conv5(hx5)
        # print("hx5=",hx5.shape)

        #stage 6
        # hx6 = self.stage6(hx)
        hx6=hx
        hx6_1=self.pam_attention_6(hx)
        hx6_2=self.cam_attention_6(hx)
        hx6=hx6_1+hx6_2
        hx6=self.conv6(hx6)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        hx1d=self.local_1d(hx1d)


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return d0, d1, d2, d3, d4, d5, d6
        # return F.softmax(d0,dim=1),F.softmax(d1,dim=1),F.softmax(d2,dim=1),F.softmax(d3,dim=1),F.softmax(d4,dim=1),F.softmax(d5,dim=1),F.softmax(d6,dim=1)
        # return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

if __name__ == "__main__":
    from src.dataset import get_loader
    from src.config import Config
    from src.utils import Table

    config = Config()
    config.BATCH_SIZE = 4
    config.FOLD = 1

    loader, _ = get_loader(config)

    model = U2Net(in_ch=1).to(config.DEVICE)

    for images, _, _ in loader:
        images = images.to(config.DEVICE)
        break

    predicts = model(images)

    Table(
        ['Item', 'Shape'],
        ['Input', images.shape],
        ['Output Fusion', predicts[0].shape],
        *[
            [f'Output {i}', predict.shape]
            for i, predict in enumerate(predicts[1:], start=1)
        ]
    ).display()
