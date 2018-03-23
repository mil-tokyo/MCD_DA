import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from models import extended_resnet


class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode="bilinear")
        x = self.conv1(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):
    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(.1)

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)

        return out


class Fusion2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Fusion2, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.1)

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.dropout(self.relu(out))

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = self._conv_bn_relu_dropout(2048)
        self.l2 = self._conv_bn_relu_dropout(256)
        self.l3 = self._conv_bn_relu_dropout(32)
        self.fc1 = nn.Linear(2048, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 2)

    def _conv_bn_relu_dropout(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 8),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
        )

    def forward(self, x):
        h = self.l1(x)  # [1, 2048, 16, 32] -> [1, 256, 16, 32]
        h = self.l2(h)  # [1, 256, 16, 32] -> [1, 32, 16, 32]
        h = self.l3(h)  # [1, 32, 16, 32] -> [1, 4, 16, 32]
        h = h.view(h.size(0), -1)  # [1, 4, 16, 32] -> [1, 2048]
        out = F.relu(self.bn1(self.fc1(h)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        return out


class ResFCN(nn.Module):
    """
    img_size: torch.Size([512, 1024])
    conv_x: torch.Size([1, 64, 256, 512])
    pool_x: torch.Size([1, 64, 128, 256])
    fm2: torch.Size([1, 512, 64, 128])
    fm3: torch.Size([1, 1024, 32, 64])
    fm4: torch.Size([1, 2048, 16, 32])
    """

    def __init__(self, num_classes, layer='50', input_ch=3):
        super(ResFCN, self).__init__()

        self.num_classes = num_classes
        print ('resnet' + layer)

        if layer == '18':
            resnet = extended_resnet.resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '34':
            resnet = extended_resnet.resnet34(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            resnet = extended_resnet.resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            resnet = extended_resnet.resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            resnet = extended_resnet.resnet152(pretrained=True, input_ch=input_ch)
        else:
            NotImplementedError

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def forward(self, x):
        input_size = x.size()
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, input_size[2:])

        out = self.out5(fsfm5)

        return out

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )


class ResBase(nn.Module):
    def __init__(self, num_classes, layer='50', input_ch=3):
        super(ResBase, self).__init__()

        self.num_classes = num_classes
        print ('resnet' + layer)

        if layer == '18':
            resnet = extended_resnet.resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            resnet = extended_resnet.resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            resnet = extended_resnet.resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            resnet = extended_resnet.resnet152(pretrained=True, input_ch=input_ch)
        else:
            NotImplementedError

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        img_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        out_dic = {
            "img_size": img_size,
            "conv_x": conv_x,
            "pool_x": pool_x,
            "fm2": fm2,
            "fm3": fm3,
            "fm4": fm4
        }

        return out_dic


class ResClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResClassifier, self).__init__()

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)
        self.out5 = self._classifier(32)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            #nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, gen_out_dic):
        gen_out_dic = edict(gen_out_dic)
        fsfm1 = self.fs1(gen_out_dic.fm3, self.upsample1(gen_out_dic.fm4, gen_out_dic.fm3.size()[2:]))
        fsfm2 = self.fs2(gen_out_dic.fm2, self.upsample2(fsfm1, gen_out_dic.fm2.size()[2:]))
        fsfm3 = self.fs4(gen_out_dic.pool_x, self.upsample3(fsfm2, gen_out_dic.pool_x.size()[2:]))
        fsfm4 = self.fs5(gen_out_dic.conv_x, self.upsample4(fsfm3, gen_out_dic.conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, gen_out_dic.img_size)
        out = self.out5(fsfm5)
        return out


class zzResBase(nn.Module):
    def __init__(self, num_classes, layer='50', input_ch=3):
        super(zzResBase, self).__init__()

        self.num_classes = num_classes
        print ('resnet' + layer)

        if layer == '18':
            resnet = extended_resnet.resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            resnet = extended_resnet.resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            resnet = extended_resnet.resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            resnet = extended_resnet.resnet152(pretrained=True, input_ch=input_ch)
        else:
            NotImplementedError

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        img_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        return conv_x, pool_x, fm1, fm2, fm3, fm4


class zzResClassifier(nn.Module):
    def __init__(self, num_classes):
        super(zzResClassifier, self).__init__()

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        # self.out0 = self._classifier(2048)
        # self.out1 = self._classifier(1024)
        # self.out2 = self._classifier(512)
        # self.out_e = self._classifier(256)
        # self.out3 = self._classifier(64)
        # self.out4 = self._classifier(64)

        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, x, conv_x, pool_x, fm1, fm2, fm3, fm4):
        input = x

        # out32 = self.out0(fm4)

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        # out16 = self.out1(fsfm1)

        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        # out8 = self.out2(fsfm2)

        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        # print(fsfm3.size())
        # out4 = self.out3(fsfm3)

        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        # out2 = self.out4(fsfm4)

        fsfm5 = self.upsample5(fsfm4, input.size()[2:])
        out = self.out5(fsfm5)

        return out


class MFResClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MFResClassifier, self).__init__()

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        self.out5 = self._classifier(32)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, gen_out_dic1, gen_out_dic2):
        gen_out_dic1 = edict(gen_out_dic1)
        gen_out_dic2 = edict(gen_out_dic2)

        assert gen_out_dic1.img_size == gen_out_dic2.img_size
        img_size = gen_out_dic1.img_size

        conv_x = gen_out_dic1.conv_x + gen_out_dic2.conv_x
        pool_x = gen_out_dic1.pool_x + gen_out_dic2.pool_x

        fm2 = gen_out_dic1.fm2 + gen_out_dic2.fm2
        fm3 = gen_out_dic1.fm3 + gen_out_dic2.fm3
        fm4 = gen_out_dic1.fm4 + gen_out_dic2.fm4

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, img_size)
        out = self.out5(fsfm5)
        return out


class MFResClassifier2(nn.Module):
    def __init__(self, num_classes):
        super(MFResClassifier2, self).__init__()
        self.num_classes = num_classes

        self.upsample1 = Upsample(2048 * 2, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion2(1024 * 2, 1024)
        self.fs2 = Fusion2(512 * 2, 512)
        self.fs3 = Fusion2(256 * 2, 256)
        self.fs4 = Fusion2(64 * 2, 64)
        self.fs5 = Fusion2(64 * 2, 64)

        self.out5 = self._classifier(32)

    def _classifier(self, inplanes):
        if inplanes == int(32):
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, gen_out_dic1, gen_out_dic2):
        gen_out_dic1 = edict(gen_out_dic1)
        gen_out_dic2 = edict(gen_out_dic2)

        assert gen_out_dic1.img_size == gen_out_dic2.img_size
        img_size = gen_out_dic1.img_size

        conv_x = torch.cat([gen_out_dic1.conv_x, gen_out_dic2.conv_x], 1)
        pool_x = torch.cat([gen_out_dic1.pool_x, gen_out_dic2.pool_x], 1)

        fm2 = torch.cat([gen_out_dic1.fm2, gen_out_dic2.fm2], 1)
        fm3 = torch.cat([gen_out_dic1.fm3, gen_out_dic2.fm3], 1)
        fm4 = torch.cat([gen_out_dic1.fm4, gen_out_dic2.fm4], 1)

        # print (gen_out_dic1.conv_x.size())
        # print (conv_x.size())
        # print (fm2.size())
        # print (fm3.size())
        # print (fm4.size())

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, img_size)
        out = self.out5(fsfm5)

        return out


class ResClassifier_P(nn.Module):
    def __init__(self, num_classes):
        super(ResClassifier_P, self).__init__()

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        self.out0 = self._classifier(2048)
        # self.out1 = self._classifier(1024)
        # self.out2 = self._classifier(512)
        # self.out_e = self._classifier(256)
        # self.out3 = self._classifier(64)
        # self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, x, conv_x, pool_x, fm1, fm2, fm3, fm4):
        input = x

        out32 = self.out0(fm4)

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))

        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))

        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        # print(fsfm3.size())

        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))

        fsfm5 = self.upsample5(fsfm4, input.size()[2:])
        out = self.out5(fsfm5)

        return out  # ,out32


class ResBaseUP(nn.Module):
    def __init__(self, num_classes, layer='50'):
        super(ResBaseUP, self).__init__()

        self.num_classes = num_classes
        if layer == '50':
            print 'resnet' + layer
            resnet = extended_resnet.resnet50(pretrained=True)
        if layer == '101':
            print 'resnet' + layer
            resnet = extended_resnet.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)
        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, input.size()[2:])

        return fsfm5


class ResClassifierUP(nn.Module):
    def __init__(self, num_classes):
        super(ResClassifierUP, self).__init__()
        self.num_classes = num_classes
        self.out5 = self._classifier(32)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, fsfm5):
        out = self.out5(fsfm5)

        return out
