import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

#from resnet200 import Res200
#from resnext import ResNeXt
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Function
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        return x.view_as(x)
    def backward(self, grad_output):
        return (grad_output*-self.lambd)
def grad_reverse(x,lambd=1.0):
    return GradReverse(lambd)(x)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)
 

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types                                                                                                                                                   
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm                                                                                                                                                            
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt()+self.eps
        x/=norm.expand_as(x)
        out = self.weight.unsqueeze(0).expand_as(x) * x
        return out
class BaseNet(nn.Module):
    #Model VGG
    def __init__(self):
        super(BaseNet, self).__init__()
        model_ft = models.vgg16(pretrained=True)
        mod = list(model_ft.features.children())
        self.features = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.classifier = nn.Sequential(*mod)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        return x
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = list(model_ft.features.children())
        self.features = model_ft.features#nn.Sequential(*mod)        
        print(self.features[0])
        #mod = list(model_ft.classifier.children())
        #mod.pop()

        #self.classifier = nn.Sequential(*mod)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),9216)
        #x = self.classifier(x)

        return x
class AlexNet_office(nn.Module):
    def __init__(self):
        super(AlexNet_office, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = list(model_ft.features.children())
        self.features = model_ft.features#nn.Sequential(*mod)        
        mod = list(model_ft.classifier.children())
        mod.pop()
        print(mod)
        self.classifier = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),9216)
        x = self.classifier(x)
        #x = F.dropout(F.relu(self.top(x)),training=self.training)

        return x
class AlexMiddle_office(nn.Module):
    def __init__(self):
        super(AlexMiddle_office, self).__init__()
        self.top = nn.Linear(4096,256)        
    def forward(self, x):
        x = F.dropout(F.relu(self.top(x)),training=self.training)
        return x


class AlexClassifier(nn.Module):
    # Classifier for VGG
    def __init__(self, num_classes=12):
        super(AlexClassifier, self).__init__()
        mod = []
        mod.append(nn.Dropout())
        mod.append(nn.Linear(4096,256))
        #mod.append(nn.BatchNorm1d(256,affine=True))
        mod.append(nn.ReLU())
        #mod.append(nn.Linear(256,256))
        mod.append(nn.Dropout())
        #mod.append(nn.ReLU())
        mod.append(nn.Dropout())
        #self.top = nn.Linear(256,256)        
        mod.append(nn.Linear(256,31))
        self.classifier = nn.Sequential(*mod)
    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x



class Classifier(nn.Module):
    # Classifier for VGG
    def __init__(self, num_classes=12):
        super(Classifier, self).__init__()
        model_ft = models.alexnet(pretrained=False)
        mod = list(model_ft.classifier.children())
        mod.pop()
        mod.append(nn.Linear(4096,num_classes))
        self.classifier = nn.Sequential(*mod)

    def forward(self, x):

        x = self.classifier(x)
        return x

class ClassifierMMD(nn.Module):
    def __init__(self, num_classes=12):
        super(ClassifierMMD, self).__init__()
        model_ft = models.vgg16(pretrained=True)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.classifier1 = nn.Sequential(*mod)
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1000,affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            )
        self.last = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.classifier1(x)
        x1 = self.classifier2(x)
        x2 = self.classifier3(x1)
        x3 = self.last(x2)
        return x3,x2,x1
class ResBase(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        mod = list(model_ft.children())
        mod.pop()
        #self.model_ft =model_ft
        self.features = nn.Sequential(*mod)
    def forward(self, x):

        x = self.features(x)
        
        x = x.view(x.size(0), self.dim)
        return x
class ResBasePlus(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBasePlus, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        mod = list(model_ft.children())
        mod.pop()
        #self.model_ft =model_ft
        self.layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000,affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(*mod)
    def forward(self, x):

        x = self.features(x)        
        x = x.view(x.size(0), self.dim)
        x = self.layer(x)
        return x

class ResNet_all(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResNet_all, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        #mod = list(model_ft.children())
        #mod.pop()
        #self.model_ft =model_ft
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.pool = model_ft.avgpool
        self.fc = nn.Linear(2048,12)
    def forward(self, x,layer_return = False,input_mask=False,mask=None,mask2=None):
        if input_mask:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = mask*self.layer1(x)
            fm2 = mask2*self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            return x#,fm1
        else:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = self.layer1(x)
            fm2 = self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            if layer_return:
                return x,fm1,fm2
            else:
                return x

class Mask_Generator(nn.Module):
    def __init__(self):
        super(Mask_Generator, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1,stride=1,padding=0)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=1,stride=1,padding=0)
        self.bn1_2 = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 512, kernel_size=1,stride=1,padding=0)

    def forward(self, x,x2):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.sigmoid(self.conv2(x))
        x2 = F.relu(self.bn1_2(self.conv1_2(x2)))
        x2 = F.sigmoid(self.conv2_2(x2))
        return x,x2


class ResMiddle_office(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResMiddle_office, self).__init__()
        self.dim = 2048
        layers = []
        layers.append(nn.Linear(self.dim,256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout())
        self.bottleneck = nn.Sequential(*layers)
        #self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.bottleneck(x)
        return x


class ResBase_office(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBase_office, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        #mod = list(model_ft.children())
        #mod.pop()
        #self.model_ft =model_ft

        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool

        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.pool = model_ft.avgpool
        #self.bottleneck = nn.Sequential(*layers)
        #self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.pool(self.layer4(fm3))
        x = fm4.view(fm4.size(0), self.dim)
        #x = self.bottleneck(x)
        return x
class ResBase_D(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBase_D, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        #mod = list(model_ft.children())
        #mod.pop()
        #self.model_ft =model_ft
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.drop0 = nn.Dropout2d()
        self.layer1 = model_ft.layer1
        self.drop1 = nn.Dropout2d()
        self.layer2 = model_ft.layer2
        self.drop2 = nn.Dropout2d()
        self.layer3 = model_ft.layer3
        self.drop3 = nn.Dropout2d()
        self.layer4 = model_ft.layer4
        self.drop4 = nn.Dropout2d()
        self.pool = model_ft.avgpool
        #self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.drop0(x)
        conv_x = x
        x = self.maxpool(x)
        fm1 = self.layer1(x)
        x = self.drop1(x)
        fm2 = self.layer2(fm1)
        x = self.drop2(x)
        fm3 = self.layer3(fm2)
        x = self.drop3(x)
        fm4 = self.pool(self.drop4(self.layer4(fm3)))
        x = fm4.view(fm4.size(0), self.dim)
        return x

class ResBasePararrel(nn.Module):
    def __init__(self,option = 'resnet18',pret=True,gpu_ids=[]):
        super(ResBasePararrel, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        mod = list(model_ft.children())
        mod.pop()
        #self.model_ft =model_ft
        self.gpu_ids = [0,1]
        self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = x + Variable(torch.randn(x.size()).cuda())*0.05
        x = nn.parallel.data_parallel(self.features, x, self.gpu_ids)
        #x = self.features(x)
        
        x = x.view(x.size(0), self.dim)
        return x

class ResFreeze(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResFreeze, self).__init__()
        self.dim = 2048*2*2
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.avgpool = model_ft.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x
        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = F.max_pool2d(self.layer4(fm3),kernel_size=3)
        #print(fm1)
        #print(fm2)
        #print(fm3)
        #print(fm4)
        #x = self.avgpool(fm4)
        x = fm4.view(fm4.size(0), self.dim)
        return x


class DenseBase(nn.Module):
    def __init__(self,option = 'densenet201',pret=True):
        super(DenseBase, self).__init__()
        self.dim = 2048
        if option == 'densenet201':
            model_ft = models.densenet201(pretrained=pret)
            self.dim = 1920
        if option == 'densenet161':
            model_ft = models.densenet161(pretrained=pret)
            self.dim = 2208
        mod = list(model_ft.children())
        #mod.pop()

        self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.features(x)
        #print x
        #x = F.avg_pool2d(x,(7,7))
        #x = x.view(x.size(0), self.dim)
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=13,num_layer = 2,num_unit=2048,prob=0.5,middle=1000):
        super(ResClassifier, self).__init__()
        layers = []
        # currently 10000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,middle))
        layers.append(nn.BatchNorm1d(middle,affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer-1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle,num_classes))
        self.classifier = nn.Sequential(*layers)

        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(2048, 1000),
        #    nn.BatchNorm1d(1000,affine=True),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(1000, 1000),
        #    nn.BatchNorm1d(1000,affine=True),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(1000, num_classes),

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x
class ResClassifier_office(nn.Module):
    def __init__(self, num_classes=12,num_layer = 2,num_unit=2048,prob=0.5,middle=256):
        super(ResClassifier_office, self).__init__()
        layers = []
        # currently 10000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,middle))
        layers.append(nn.BatchNorm1d(middle,affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(middle,num_classes))
        self.classifier = nn.Sequential(*layers)
    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x
class DenseClassifier(nn.Module):
    def __init__(self, num_classes=12,num_layer = 2):
        super(DenseClassifier, self).__init__()
        layers = []
        # currently 1000 units
        layers.append(nn.Dropout())
        layers.append(nn.Linear(1000,500))
        layers.append(nn.BatchNorm1d(500,affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer-1):
            layers.append(nn.Dropout())
            layers.append(nn.Linear(500,500))
            layers.append(nn.BatchNorm1d(500,affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(500,num_classes))
        #layers.append(nn.BatchNorm1d(num_classes,affine=True,momentum=0.95))
        self.classifier = nn.Sequential(*layers)
                    
        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(2048, 1000),
        #    nn.BatchNorm1d(1000,affine=True),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(1000, 1000),
        #    nn.BatchNorm1d(1000,affine=True),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(1000, num_classes),

        #)
    def forward(self, x):
        x = self.classifier(x)
        #x = self.classifier(x)
        return x


class AE(nn.Module):
    def __init__(self, num_classes=12,num_layer = 2,ngf=32,norm_layer=nn.BatchNorm2d):
        super(AE, self).__init__()
        layers = []
        layers.append(nn.Dropout())
        layers.append(nn.Linear(512,32*8*8))
        #layers.append(nn.BatchNorm1d(64*8*8,affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(*layers)
        n_downsampling=5
        mult = 2**n_downsampling
        n_blocks = 3
        model2 = [nn.Conv2d(32, ngf*mult, kernel_size=5,
                stride=4, padding=1),
                 norm_layer(ngf * mult, affine=True),
                 nn.ReLU()]

        #model2 = [nn.ConvTranspose2d(64, ngf * mult,
        #                                 kernel_size=3, stride=2,
        #                                 padding=1, output_padding=1),
        #              norm_layer(ngf * mult, affine=True),
        #              nn.ReLU()]                
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=True)]
        #print ngf*mult
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                          padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU()]
            #model2 += [nn.Conv2d(ngf*mult/2, ngf, 
            #                 kernel_size=1, padding=1),
            #       norm_layer(int(ngf), affine=True),
            #       nn.ReLU()]
            #model2 += [nn.Conv2d(int(ngf * mult / 2), 3, kernel_size=, padding=3)]
        model2 += [nn.Conv2d(ngf, 3, kernel_size=11, padding=1)]
        model2 += [nn.Tanh()]

        self.classifier2 = nn.Sequential(*model2)
    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0),32,8,8)
        x = self.classifier2(x)
        return x


class InceptionBase(nn.Module):
    def __init__(self):
        super(InceptionBase, self).__init__()
        model_ft = models.inception_v3(pretrained=True)
        #mod = list(model_ft.children())
        #mod.pop()
        self.features = model_ft#nn.Sequential(*mod)
    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), 2048)
        return x
class InceptionClassifier(nn.Module):
    def __init__(self, num_classes=12,num_layer = 2):
        super(InceptionClassifier, self).__init__()
        layers = []
        layers.append(nn.Dropout())
        layers.append(nn.Linear(1000,1000))
        layers.append(nn.BatchNorm1d(1000,affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer-1):
            layers.append(nn.Dropout())
            layers.append(nn.Linear(1000,1000))
            layers.append(nn.BatchNorm1d(1000,affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(1000,num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x)
        #x = self.classifier(x)
        return x


class ResClassifierMMD(nn.Module):
    def __init__(self, num_classes=12):
        super(ResClassifierMMD, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 1000),
            nn.BatchNorm1d(1000,affine=True),
            #nn.Dropout(),
            nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256,affine=True),
            #nn.Dropout(),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(256, num_classes)
    def forward(self, x):
        x1 = self.classifier(x)
        x2 = self.classifier2(x1)
        x3 = self.last(x2)
        return x3,x2,x1
class BaseShallow(nn.Module):
    def __init__(self,num_classes=12):
        super(BaseShallow, self).__init__()
        layers = []
        nc = 3
        ndf = 64
        self.features = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 7, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

        self.last = nn.Sequential(
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x1 = self.features(x)
        x1 = x1.view(-1,100)
        #x2 = self.last(x1)
        #x3 = self.last(x2)
        return x1

class ClassifierShallow(nn.Module):

    def __init__(self, num_classes=12):
        super(ClassifierShallow, self).__init__()
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(100, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000,affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes),
        )
    def forward(self, x):
        #x = self.classifier1(x)
        x = self.classifier2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes=12):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 100),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(100,affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )
    def forward(self, x):
        #x = self.classifier1(x)
        #print x
        x = self.classifier(x)
        return x


class EClassifier(nn.Module):

    def __init__(self, num_classes=12):
        super(EClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216+12, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000,affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(num_classes,12),
        )
    def forward(self, x1,x2):
        x = torch.cat([x1,x2],1)
        x = self.classifier(x)
        x_source = self.classifier2(x)
        return x,x_source

class Resbridge(nn.Module):
    def __init__(self, num_classes=12,num_layer = 2,num_unit=2048,prob=0.5):
        super(Resbridge, self).__init__()
        layers = []
        # currently 1000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,500))
        layers.append(nn.BatchNorm1d(500,affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.classifier1 = nn.Sequential(*layers)
        layers2 = []
        layers2.append(nn.Dropout(p=prob))
        layers2.append(nn.Linear(1000,500))
        layers2.append(nn.BatchNorm1d(500,affine=True))
        layers2.append(nn.ReLU(inplace=True))
        for i in range(num_layer-1):
            layers2.append(nn.Dropout(p=prob))
            layers2.append(nn.Linear(500,500))
            layers2.append(nn.BatchNorm1d(500,affine=True))
            layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.Linear(500,num_classes))
        self.classifier2 = nn.Sequential(*layers2)


        layers3 = []
        # currently 1000 units
        layers3.append(nn.Dropout(p=prob))
        layers3.append(nn.Linear(num_unit,500))
        layers3.append(nn.BatchNorm1d(500,affine=True))
        layers3.append(nn.ReLU(inplace=True))
        self.classifier3 = nn.Sequential(*layers3)
        layers4 = []
        layers4.append(nn.Dropout(p=prob))
        layers4.append(nn.Linear(1000,500))
        layers4.append(nn.BatchNorm1d(500,affine=True))
        layers4.append(nn.ReLU(inplace=True))
        for i in range(num_layer-1):
            layers4.append(nn.Dropout(p=prob))
            layers4.append(nn.Linear(500,500))
            layers4.append(nn.BatchNorm1d(500,affine=True))
            layers4.append(nn.ReLU(inplace=True))
        layers4.append(nn.Linear(500,num_classes))
        self.classifier4 = nn.Sequential(*layers4)

    def forward(self, x):
        x1 = self.classifier1(x)
        x3 = self.classifier3(x)
        x2 = torch.cat((x1,x3),1)
        x2 = self.classifier2(x2)
        x4 = torch.cat((x3,x1),1)
        x4 = self.classifier4(x4)

        return x2,x4
