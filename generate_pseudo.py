import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist

def obtain_label(args):
    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)

    target_loader = torch.utils.data.DataLoader(
        ImageListIndex(open(args.target_list).readlines(), transform=transform_target, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    model = torch.load("art_RW.pth")
    ad_net = torch.load("ar_ad_net")
    transformer = model.transformer
    head = model.head
    
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in tqdm(range(len(loader))):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            # feas = netB(netF(inputs))
            # outputs = netC(feas)
            feas, _, _, _ = transformer(inputs, ad_net)
            feas = feas[:, 0]
            outputs = head(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - ent / np.log(65)
    _, predict = torch.max(all_output, 1)

    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # print(accuracy)
    # if args.distance == 'cosine':
    # print('1',all_fea.shape)   #[4357, 768]
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    # print('2', all_fea.shape)   #[4357, 769]
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    # print(K)
    # print(predict.shape)  #[4537]
    # print(np.eye(K)[predict].shape)  #[4537, 65]
    cls_count = np.eye(K)[predict].sum(axis=0)
    # print(cls_count.shape)  #[65]
    # labelset = np.where(cls_count>args.threshold)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]
    # print(labelset)
    # print(initc.shape)   #[65, 769]
    # print(labelset.shape)
    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    # acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    # print(acc)
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    # print(log_str+'\n')
    # print(pred_label.shape)
    return torch.tensor(pred_label.astype('int'))

if __name__ == '__main__':
    pseudo_label = obtain_label()