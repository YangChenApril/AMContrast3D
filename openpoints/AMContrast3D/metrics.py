import torch
import torch.nn.functional as F
from openpoints.cpp.pointops.functions import pointops
from openpoints.AMContrast3D.AEF.ambiguity import ambiguity_function
import numpy as np
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious


def ambiguity_summary(num_classes, ambiguity_vs_accuracy_list, ambiguity_vs_count_list, ambiguity_vs_accuracy_lowsemihigh_list, ambiguity_vs_cls_list, ambiguity_cm_miou, ambiguity_cm_macc, ambiguity_cm_oa, ambiguity_cm_count):
    cls_per_ai = {}
    for c in range(num_classes):
        append_c =[]
        for l in ambiguity_vs_cls_list:
            if c in l.keys():
                append_c.append(l[c])
        cls_per_ai[c] = np.around(np.mean(append_c, axis=0), decimals=3)
        print('count per cls: ', c, cls_per_ai[c])
    
    print('count per a_i: 0, low=(0,0.5), semi=0.5, high=(0.5,1), 1:', np.around(np.mean(ambiguity_vs_count_list, axis=0), decimals=3))
    print('acc per a_i: 0, low=(0,0.5), semi=0.5, high=(0.5,1), 1:', np.around(np.mean(ambiguity_vs_accuracy_lowsemihigh_list, axis=0), decimals=3))

    print('miou per ambiguity:', np.around(np.mean(ambiguity_cm_miou, axis=0), decimals=2))
    print('macc per ambiguity:', np.around(np.mean(ambiguity_cm_macc, axis=0), decimals=2))
    print('oa per ambiguity:', np.around(np.mean(ambiguity_cm_oa, axis=0), decimals=2))        
    print('count-0:   ', np.around(np.mean(ambiguity_cm_count, axis=0)[0], decimals=0))
    print('count-low: ', np.around(np.mean(ambiguity_cm_count, axis=0)[1], decimals=0))
    print('count-semi:', np.around(np.mean(ambiguity_cm_count, axis=0)[2], decimals=0))
    print('count-high:', np.around(np.mean(ambiguity_cm_count, axis=0)[3], decimals=0))
    print('count-1:   ', np.around(np.mean(ambiguity_cm_count, axis=0)[4], decimals=0))



def ambiguity_metrics(p, label, pred, posmask_test, nsample_test, neighbor_idx_test, cctype, ccbeta, vis, cm_0, cm_low, cm_semi, cm_high, cm_1, nu):

    '''---[ambiguity]---'''
    ambiguity_soft, ambiguity_count = ambiguity_function(p, posmask_test, nsample_test, neighbor_idx_test, cctype, ccbeta, vis, nu)

    '''# scannet (Select 0~19)
    if 20 in torch.unique(label):
        remove_idx = label == 20
        remain_idx = ~remove_idx
        label = torch.index_select(label, 0, torch.flatten(torch.nonzero(remain_idx.int())))
        pred = torch.index_select(pred, 0, torch.flatten(torch.nonzero(remain_idx.int())))
        ambiguity_soft = torch.index_select(ambiguity_soft, 0, torch.flatten(torch.nonzero(remain_idx.int())))'''

    '''---[mapping]---: 
    [0, 0.1, 0.2, ... , 1] -> [1, 2, ..., 11]
    nu=0.5 (S3DIS); nu=0.6 (ScanNet)'''
    mapping = torch.floor(ambiguity_soft * 10 + 1) 
    total_unqiue = torch.unique(mapping, return_counts=True) 

    a_0 = ambiguity_soft == 0
    a_low = torch.logical_and(0 < ambiguity_soft, ambiguity_soft < nu) 
    a_semi = ambiguity_soft == nu
    a_high = torch.logical_and(nu < ambiguity_soft, ambiguity_soft < 1)
    a_1 = ambiguity_soft == 1

    nu_m = nu * 10 + 1

    mapping_0 = mapping == 1
    mapping_low = torch.logical_and(1 < mapping, mapping < nu_m) # nu = 6 or 7
    mapping_semi = mapping == nu_m
    mapping_high = torch.logical_and(nu_m < mapping, mapping < 11)
    mapping_1 = mapping == 11


    '''---[miou, macc, oa, count]---'''

    pred_0 = torch.index_select(pred, 0, torch.flatten(torch.nonzero(mapping_0.int())))
    pred_low = torch.index_select(pred, 0, torch.flatten(torch.nonzero(mapping_low.int())))
    pred_semi = torch.index_select(pred, 0, torch.flatten(torch.nonzero(mapping_semi.int())))
    pred_high = torch.index_select(pred, 0, torch.flatten(torch.nonzero(mapping_high.int())))
    pred_1 = torch.index_select(pred, 0, torch.flatten(torch.nonzero(mapping_1.int())))


    label_0 = torch.index_select(label, 0, torch.flatten(torch.nonzero(mapping_0.int())))
    label_low = torch.index_select(label, 0, torch.flatten(torch.nonzero(mapping_low.int())))
    label_semi = torch.index_select(label, 0, torch.flatten(torch.nonzero(mapping_semi.int())))
    label_high = torch.index_select(label, 0, torch.flatten(torch.nonzero(mapping_high.int())))
    label_1 = torch.index_select(label, 0, torch.flatten(torch.nonzero(mapping_1.int())))

    cm_0.update(pred_0, label_0)
    cm_low .update(pred_low , label_low )
    cm_semi.update(pred_semi, label_semi)
    cm_high.update(pred_high, label_high)
    cm_1.update(pred_1, label_1)

    tp_0, union_0, count_0 = cm_0.tp, cm_0.union, cm_0.count
    miou_0, macc_0, oa_0, _, _ = get_mious(tp_0, union_0, count_0)
    tp_low, union_low, count_low = cm_low.tp, cm_low.union, cm_low.count
    miou_low, macc_low, oa_low, _, _ = get_mious(tp_low, union_low, count_low)
    tp_semi, union_semi, count_semi = cm_semi.tp, cm_semi.union, cm_semi.count
    miou_semi, macc_semi, oa_semi, _, _ = get_mious(tp_semi, union_semi, count_semi)
    tp_high, union_high, count_high = cm_high.tp, cm_high.union, cm_high.count
    miou_high, macc_high, oa_high, _, _ = get_mious(tp_high, union_high, count_high)
    tp_1, union_1, count_1 = cm_1.tp, cm_1.union, cm_1.count
    miou_1, macc_1, oa_1, _, _ = get_mious(tp_1, union_1, count_1)

    cm_miou_list = [round(miou_0,2), round(miou_low,2), round(miou_semi,2), round(miou_high,2), round(miou_1,2)]
    cm_macc_list = [round(macc_0,2), round(macc_low,2), round(macc_semi,2), round(macc_high,2), round(macc_1,2)]
    cm_oa_list = [round(oa_0,2), round(oa_low,2), round(oa_semi,2), round(oa_high,2), round(oa_1,2)]
    cm_count_list = [count_0.tolist(), count_low.tolist(), count_semi.tolist(), count_high.tolist(), count_1.tolist()] # num_class

    print('miou per ambiguity:', cm_miou_list)
    print('macc per ambiguity:', cm_macc_list)
    print('oa per ambiguity:', cm_oa_list)
    print('count per ambiguity:', cm_count_list)

    '''accuracy'''
    true_mask = pred == label 
    true_unqiue = torch.unique(mapping * true_mask.int(), return_counts=True) 
    
    ratio = {}
    for i in range(len(total_unqiue[0])):
        key = total_unqiue[0][i].item()
        down = total_unqiue[1][i].item()
        if key in true_unqiue[0]:
            j = torch.nonzero(true_unqiue[0]==key).squeeze()
            up = true_unqiue[1][j].item()
        else:
            up = 0
        if up/down > 1:
            print('error')
        ratio[key] = up/down
        
    ratio_low_semi_high = [1.0, 1.0, 1.0, 1.0, 1.0]

    '''count'''
    count_0 = round(torch.sum(mapping_0).item() / len(mapping) * 100, 2)
    count_low = round(torch.sum(mapping_low).item() / len(mapping) * 100, 2)
    count_semi = round(torch.sum(mapping_semi).item() / len(mapping) * 100, 2)
    count_high = round(torch.sum(mapping_high).item() / len(mapping) * 100, 2)
    count_1 = round(torch.sum(mapping_1).item() / len(mapping) * 100, 2)

    cls = {}
    for c in torch.unique(label):
        mask_c = label == c
        mapping_c = mapping * mask_c

        mapping_c_0 = mapping_c == 1
        mapping_c_low = torch.logical_and(1 < mapping_c, mapping_c < 6)
        mapping_c_semi = mapping_c == 6
        mapping_c_high = torch.logical_and(6 < mapping_c, mapping_c < 11)
        mapping_c_1 = mapping_c == 11

        count_c_0 = round(torch.sum(mapping_c_0).item() / torch.sum(mask_c).item() * 100, 2)
        count_c_low = round(torch.sum(mapping_c_low).item() / torch.sum(mask_c).item() * 100, 2)
        count_c_semi = round(torch.sum(mapping_c_semi).item() / torch.sum(mask_c).item() * 100, 2)
        count_c_high = round(torch.sum(mapping_c_high).item() / torch.sum(mask_c).item() * 100, 2)
        count_c_1 = round(torch.sum(mapping_c_1).item() / torch.sum(mask_c).item() * 100, 2)

        cls[c.item()] = [count_c_0, count_c_low, count_c_semi, count_c_high, count_c_1]

    print('(%) count per cls: 0, low=(0,0.5), semi=0.5, high=(0.5,1), 1:', cls)

    return ambiguity_soft, ratio, ambiguity_count, ratio_low_semi_high, cls, cm_miou_list, cm_macc_list, cm_oa_list, cm_count_list
    


def posmask_searching(xyz, target, nsample, num_classes, ignore_index):
    '''print(num_classes, ignore_index) 
    ===> S3DIS: 13, None
    ===> ScanNet: 20, -100'''
    if ignore_index is not None: ### [ScanNet]
        num_classes = num_classes + 1 
        if (target == ignore_index).sum() > 0: ### The num. of labeled (-100) points
            target = target.clone()            ### To Solve [Need Inplace Operation Error]
            target[target == ignore_index] = num_classes -1 ### From labeled (-100) to labeled (20), others still 0,1,2,...,19

    x = F.one_hot(target, num_classes)  

    labels = x.float() 
    o = torch.IntTensor([xyz.shape[0]]).cuda()   
    neighbor_idx, _ = pointops.knnquery(nsample, xyz, xyz, o, o)

    '''exclude self-loop'''
    nsample -= 1
    neighbor_idx = neighbor_idx[..., 1:].contiguous() 
    m = neighbor_idx.shape[0]
    neighbor_label = labels[neighbor_idx.view(-1).long(), :].view(m, nsample, labels.shape[1])
    labels = torch.argmax(torch.unsqueeze(labels, -2), -1)  # [m, 1]
    neighbor_label = torch.argmax(neighbor_label, -1)  # [m, nsample]
    posmask = labels == neighbor_label  # [m, nsample]
    return posmask, neighbor_idx


def vis_tsne(stageACE_list_all):
    from sklearn.manifold import TSNE
    from numpy import reshape
    import seaborn as sns
    import pandas as pd 
    import matplotlib.pyplot as plt

    x = []
    y = []
    for i in range(len(stageACE_list_all)):
        x.extend(stageACE_list_all[i]['up'][0]['f_out'].cpu().detach().numpy()) # [:1000] ### [120000, 64]
        y.extend(stageACE_list_all[i]['inputs']['y'].flatten().cpu().detach().numpy()) # [:1000] ### [120000]

    x = x[60000:180000]
    y = y[60000:180000]
    print(len(x))
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x) 

    classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter']

    df = pd.DataFrame()
    df["y"] = y
    df['name'] = [classes[y_i] for y_i in y]
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.name.tolist(),
                    palette=sns.color_palette("Paired", 13),
                    data=df)

    fig = sns_plot.get_figure()
    num = 8884 # 3407
    fig.savefig('./plot/output_' + str(num) + '.png')

                