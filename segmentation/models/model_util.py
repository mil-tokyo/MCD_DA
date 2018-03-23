import torch


def get_full_model(net, res, n_class, input_ch):
    if net == "fcn":
        from models.fcn import ResFCN
        return torch.nn.DataParallel(ResFCN(n_class, res, input_ch))
    elif net == "fcnvgg":
        from models.vgg_fcn import FCN8s
        return torch.nn.DataParallel(FCN8s(n_class))

    elif "drn" in net:
        from models.dilated_fcn import DRNSeg
        assert net in ["drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22", "drn_d_38", "drn_d_54", "drn_d_105"]
        return torch.nn.DataParallel(DRNSeg(net, n_class, input_ch=input_ch))
    else:
        raise NotImplementedError("Only FCN, DRN are supported!")


def get_models(net_name, input_ch, n_class, res="50", method="MCD", uses_one_classifier=False, use_ae=False,
               is_data_parallel=False):
    def get_MCD_model_list():
        if net_name == "fcn":
            from models.fcn import ResBase, ResClassifier
            model_g = ResBase(n_class, layer=res, input_ch=input_ch)
            model_f1 = ResClassifier(n_class)
            model_f2 = ResClassifier(n_class)
        elif net_name == "fcnvgg":
            from models.vgg_fcn import FCN8sBase, FCN8sClassifier
            model_g = FCN8sBase(n_class)
            model_f1 = FCN8sClassifier(n_class)
            model_f2 = FCN8sClassifier(n_class)
        elif "drn" in net_name:
            from models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier_ADR
            if uses_one_classifier:
                model_g = DRNSegBase(model_name=net_name, n_class=n_class, input_ch=input_ch)
                model_f1 = DRNSegPixelClassifier_ADR(n_class=n_class)
                model_f2 = DRNSegPixelClassifier_ADR(n_class=n_class)
            else:
                from models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier
                model_g = DRNSegBase(model_name=net_name, n_class=n_class, input_ch=input_ch)
                model_f1 = DRNSegPixelClassifier(n_class=n_class)
                model_f2 = DRNSegPixelClassifier(n_class=n_class)

        else:
            raise NotImplementedError("Only FCN (Including Dilated FCN), SegNet, PSPNetare supported!")

        return model_g, model_f1, model_f2

    if method == "MCD":
        model_list = get_MCD_model_list()
    else:
        return NotImplementedError("Sorry... Only MCD is supported!")

    if is_data_parallel:
        return [torch.nn.DataParallel(x) for x in model_list]
    else:
        return model_list


def get_optimizer(model_parameters, opt, lr, momentum, weight_decay):
    if opt == "sgd":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model_parameters), lr=lr, momentum=momentum,
                               weight_decay=weight_decay)

    elif opt == "adadelta":
        return torch.optim.Adadelta(filter(lambda p: p.requires_grad, model_parameters), lr=lr,
                                    weight_decay=weight_decay)

    elif opt == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model_parameters), lr=lr, betas=[0.5, 0.999],
                                weight_decay=weight_decay)
    else:
        raise NotImplementedError("Only (Momentum) SGD, Adadelta, Adam are supported!")



def check_training(model):
    print (type(model))
    print (model.training)
    for module in model.children():
        check_training(module)
