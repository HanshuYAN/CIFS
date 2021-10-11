import os
import sys
import pathlib

import torch
import torch.nn.functional as F

import advertorch
# from advertorch.attacks.utils import multiple_mini_batch_attack
# from advertorch.utils import predict_from_logits

from tqdm import tqdm
####################

def topk_dataset_accuracy(predict, test_loader, num_batch=None, device='cuda', topk=1):

    clncorrect = 0
    idx_batch = 0
    num_examples = 0
    
    lst_label = []
    lst_pred = []
    
    for clndata, target in tqdm(test_loader):
        clndata, target = clndata.to(device), target.to(device)
        # with torch.no_grad():
        output = predict(clndata)
        pred = predict_from_logits_topk(output, topk=topk)
        
        lst_label.append(target)
        lst_pred.append(pred)
        
        num_examples += clndata.shape[0]
        idx_batch += 1
        if idx_batch == num_batch:
            break
    
    label = torch.cat(lst_label).view(-1, 1)
    pred = torch.cat(lst_pred).view(-1, topk)
    num = label.size(0)
    accuracy = (label == pred).sum().item() / num

    message = '***** Test set acc: {}/{} ({:.2f}%)'.format(
                clncorrect, 
                num_examples,
                100. * accuracy)
    return accuracy, message


def topk_defense_success_rate(predict_for_test, predict_for_atk, loader, attack_class, attack_kwargs,
                    device=torch.device("cuda:0"), num_batch=None, topk=1):
    
    adversary = attack_class(predict_for_atk, **attack_kwargs)
    accuracy, defense_success_rate, dist, num = attack_mini_batches(predict_for_test, adversary, loader, device=device, norm=None, num_batch=num_batch, topk=topk)
    
    # message returned
    message = '***** Test set acc: {:.2f}%, adv: {:.2f}%.'.format(
            accuracy * 100., 
            defense_success_rate * 100.)
    rval = _generate_basic_benchmark_str(attack_class, attack_kwargs, num, accuracy, defense_success_rate)
    return accuracy, defense_success_rate, message, rval



def predict_from_logits_topk(logits, dim=1, topk=1):
    return logits.topk(topk, dim)[1]

def attack_mini_batches(myPredict, 
                adversary, loader, device="cuda",
                norm=None, num_batch=None, topk=1):
    lst_label = []
    lst_pred = []
    lst_advpred = []
    lst_dist = []

    _norm_convert_dict = {"Linf": "inf", "L2": 2, "L1": 1}
    if norm in _norm_convert_dict:
        norm = _norm_convert_dict[norm]

    if norm == "inf":
        def dist_func(x, y):
            return (x - y).view(x.size(0), -1).max(dim=1)[0]
    elif norm == 1 or norm == 2:
        from advertorch.utils import _get_norm_batch
        def dist_func(x, y):
            return _get_norm_batch(x - y, norm)
    else:
        assert norm is None

    idx_batch = 0
    from tqdm import tqdm
    for data, label in tqdm(loader):
        data, label = data.to(device), label.to(device)
        adv = adversary.perturb(data, label)
        
        adv_logits = myPredict(adv); advpred = predict_from_logits_topk(adv_logits, topk=topk)
        nat_logits = myPredict(data); pred = predict_from_logits_topk(nat_logits, topk=topk)

        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        
        if norm is not None:
            lst_dist.append(dist_func(data, adv))
            
        idx_batch += 1
        if idx_batch == num_batch:
            break
    
    label = torch.cat(lst_label).view(-1, 1)
    pred = torch.cat(lst_pred).view(-1, topk)
    advpred = torch.cat(lst_advpred).view(-1, topk)
    dist = torch.cat(lst_dist) if norm is not None else None
    
    num = label.size(0)
    accuracy = (label == pred).sum().item() / num
    defense_success_rate = (label == advpred).sum().item() / num
    dist = None if dist is None else dist[(label != advpred) & (label == pred)]
    
    return accuracy, defense_success_rate, dist, num




##################

def dataset_accuracy(predict, test_loader, num_batch=None, device=torch.device("cuda")):

    clncorrect = 0
    idx_batch = 0
    num_examples = 0
    
    for clndata, target in tqdm(test_loader):
        clndata, target = clndata.to(device), target.to(device)
        output = predict(clndata)
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()
        
        num_examples += clndata.shape[0]
        idx_batch += 1
        if idx_batch == num_batch:
            break
        
    accuracy = clncorrect / num_examples
    message = '***** Test set acc: {}/{} ({:.2f}%)'.format(
                clncorrect, 
                num_examples,
                100. * accuracy)
    return accuracy, message

        
        
def benchmark_defense_success_rate(predict_for_test, predict_for_atk, loader, attack_class, attack_kwargs,
                    device=torch.device("cuda:0"), num_batch=None):
    
    adversary = attack_class(predict_for_atk, **attack_kwargs)
    label, pred, advpred, dist = multiple_mini_batch_attack(predict_for_test, adversary, loader, device=device, norm=None, num_batch=num_batch)
    
    accuracy = (label == pred).sum().item() / len(label)
    defense_success_rate = (label == advpred).sum().item() / len(label)
    dist = None if dist is None else dist[(label != advpred) & (label == pred)]
    num = len(label)
    
    # message returned
    message = '***** Test set acc: {:.2f}%, adv: {:.2f}%.'.format(
            accuracy * 100., 
            defense_success_rate * 100.)
    rval = _generate_basic_benchmark_str(attack_class, attack_kwargs, num, accuracy, defense_success_rate)
    return accuracy, defense_success_rate, message, rval



def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def multiple_mini_batch_attack(myPredict, 
                adversary, loader, device="cuda",
                norm=None, num_batch=None):
    lst_label = []
    lst_pred = []
    lst_advpred = []
    lst_dist = []

    _norm_convert_dict = {"Linf": "inf", "L2": 2, "L1": 1}
    if norm in _norm_convert_dict:
        norm = _norm_convert_dict[norm]

    if norm == "inf":
        def dist_func(x, y):
            return (x - y).view(x.size(0), -1).max(dim=1)[0]
    elif norm == 1 or norm == 2:
        from advertorch.utils import _get_norm_batch
        def dist_func(x, y):
            return _get_norm_batch(x - y, norm)
    else:
        assert norm is None

    idx_batch = 0
    from tqdm import tqdm
    for data, label in tqdm(loader):
        data, label = data.to(device), label.to(device)
        adv = adversary.perturb(data, label)
        
        adv_logits = myPredict(adv); advpred = predict_from_logits(adv_logits)
        nat_logits = myPredict(data); pred = predict_from_logits(nat_logits)
            
        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        
        if norm is not None:
            lst_dist.append(dist_func(data, adv))
            
        idx_batch += 1
        if idx_batch == num_batch:
            break

    return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred), \
        torch.cat(lst_dist) if norm is not None else None
        


def _generate_basic_benchmark_str(attack_class, attack_kwargs, num, accuracy,
        defense_success_rate):
    rval = ""
    rval += "# attack type: {}\n".format(attack_class.__name__)
    prefix = "# attack kwargs: "
    # count = 0
    # for key in attack_kwargs:
    #     this_prefix = prefix if count == 0 else " " * len(prefix)
    #     count += 1
    #     rval += "{}{}={}\n".format(this_prefix, key, attack_kwargs[key])
        
    rval += prefix
    for key in attack_kwargs:
        rval += "{}={}, ".format(key, attack_kwargs[key])
    rval += '\n#\n'
    
    rval += "# accuracy: {:.2f}%\n".format(accuracy * 100.)
    rval += "# defending rate: {:.2f}%\n".format(defense_success_rate * 100.)
    return rval






















def _calculate_benchmark_results(
        model, loader, attack_class, attack_kwargs, norm, device, num_batch, num_pred_output):
    
    adversary = attack_class(model, **attack_kwargs)
    
    label, pred, advpred, dist = multiple_mini_batch_attack(
        adversary, loader, device=device, norm=norm, num_batch=num_batch, num_pred_output=num_pred_output)
    
    accuracy = (label == pred).sum().item() / len(label)
    defense_success_rate = (label == advpred).sum().item() / len(label)
    
    dist = None if dist is None else dist[(label != advpred) & (label == pred)]
    return len(label), accuracy, defense_success_rate, dist





def benchmark_margin(
        model, loader, attack_class, attack_kwargs, norm,
        device="cuda", num_batch=None):

    num, accuracy, attack_success_rate, dist = _calculate_benchmark_results(
        model, loader, attack_class, attack_kwargs, norm, device, num_batch)
    rval = _generate_basic_benchmark_str(
        model, loader, attack_class, attack_kwargs, num, accuracy,
        attack_success_rate)

    rval += "# Among successful attacks ({} norm) ".format(norm) + \
        "on correctly classified examples:\n"
    rval += "#    minimum distance: {:.4}\n".format(dist.min().item())
    rval += "#    median distance: {:.4}\n".format(dist.median().item())
    rval += "#    maximum distance: {:.4}\n".format(dist.max().item())
    rval += "#    average distance: {:.4}\n".format(dist.mean().item())
    rval += "#    distance standard deviation: {:.4}\n".format(
        dist.std().item())

    return rval


