import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate

from tqdm import tqdm
from loguru import logger
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.SiamNet as SiamNet


def valid(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):
    num_classes, att_size, feat_size = args.num_classes, 1, 2048
    model = SiamNet.siamnet(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                            device=args.device, pretrained=True)

    model.to(args.device)

    model.load_state_dict(torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/model.pth'),
                          strict=False)
    model.eval()
    # print(model.state_dict()['W_G'])
    w_g_abs = torch.abs(model.state_dict()['W_G'])
    for i in range(12):
        values, index = torch.sort(w_g_abs[i])
        values = values.cuda()
        for j in range(len(w_g_abs[i])):
            if w_g_abs[0][j] < values[127]:
                model.state_dict()['W_G'][i][j] = 0.0
    print(model.state_dict()['W_G'])
    query_code = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/query_code.pth')
    query_code = query_code.to(args.device)
    query_dataloader.dataset.get_onehot_targets = torch.load(
        './checkpoints/' + args.info + '/' + str(code_length) + '/query_targets.pth')
    B = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_code.pth')
    B = B.to(args.device)
    retrieval_targets = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_targets.pth')
    retrieval_targets = retrieval_targets.to(args.device)

    mAP = evaluate.mean_average_precision(
        query_code.to(args.device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(args.device),
        retrieval_targets,
        args.device,
        args.topk,
    )
    print("Code_Length: " + str(code_length), end="; ")
    print('[mAP:{:.5f}]'.format(mAP))
