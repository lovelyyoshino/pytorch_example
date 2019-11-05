import fire
import json
import torch
import os
import numpy as np
import torch.optim as optim

from ipdb import set_trace
from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from evaluate import return_report

import dataset
import models
import utils

from config import opt


def collate_fn(batch):
    """Function:用来对DataModel打包的npy数据进行解包,其对应打包的数据
    使用说明: 替换下方的x1, x2, x3
    """
    x1, x2, x3 = zip(*batch)
    return x1, x2, x3


def setup_seed(seed):
    """Funciton:固定随机种子，使得模型每次结果唯一
    使用说明:不用改变
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def go(**keward):

    opt.parse(keward)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    setup_seed(opt.seed)


    DataModel = getattr(dataset, 'DataModel')
    train_data = DataModel(opt, case='train')
    train_data_loader = DataLoader(train_data, opt.train_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    dev_data = DataModel(opt, case='dev')
    dev_data_loader = DataLoader(dev_data, opt.dev_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_data = DataModel(opt, case='test')
    test_data_loader = DataLoader(test_data, opt.test_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    print("*****************************************************")
    print("*****************************************************")
    print("train data size:{}; dev data size:{}; test data size:{}".format(len(train_data), len(dev_data), len(test_data)))
    print("*****************************************************")
    print("*****************************************************")
    model = getattr(models, opt.model)(opt)
    if opt.use_gpu:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1+0.95*epoch))
    train_steps = (len(train_data) + opt.train_batch_size - 1) // opt.train_batch_size
    dev_step = (len(dev_data) + opt.dev_batch_size - 1) // opt.dev_batch_size
    test_step = (len(test_data) + opt.test_batch_size - 1) // opt.test_batch_size

    """
    更改res_dict为自己对应指标
    """
    res_dict = {
        'model': opt.print_opt,
        'epoch_loss': [],
        'epoch_train_f1': [],
        'epoch_dev_f1': [],
        'epoch_test_f1': [],
    }
    best_dev_f1 = 0

    print("start training...")
    for epoch in range(opt.num_epochs):
        print("{}; epoch:{}/{}; training....".format(utils.now(), epoch, opt.num_epochs))
        loss = train(model, train_data_loader, scheduler, optimizer, train_steps, opt)

        print("{}; epoch:{}/{}; judge.....".format(utils.now(), epoch, opt.num_epochs))

        print("{}; epoch:{}/{}; train data".format(utils.now(), epoch, opt.num_epochs))
        train_f1 = predict(model, train_data_loader, train_steps, opt, train_data.data_processer.id2label, case=0)
        print(" ".join(train_report))

        print("{}; epoch:{}/{}; dev data".format(utils.now(), epoch, opt.num_epochs))
        dev_f1 = predict(model, dev_data_loader, dev_step, opt, dev_data.data_processer.id2label, case=1)
        print(" ".join(dev_report))

        print("{}; epoch:{}/{}; test data".format(utils.now(), epoch, opt.num_epochs))
        test_f1 = predict(model, test_data_loader, test_step, opt, dev_data.data_processer.id2label, case=2)
        print(" ".join(test_report))

        res_dict['epoch_loss'].append(loss)
        res_dict['epoch_train_f1'].append(train_f1)
        res_dict['epoch_dev_f1'].append(dev_f1)
        res_dict['epoch_test_f1'].append(test_f1)

        if best_dev_f1 < dev_f1:
            best_dev_f1 = dev_f1
            cores_test_f1 = test_f1
            os.system("mv {} {}".format(opt.dev_out_path, opt.dev_out_path + '_best'))
            os.system("mv {} {}".format(opt.test_out_path, opt.test_out_path + '_best'))
            print("*****************************************************************")
            print("*****************************************************************")
            print("epoch:{}; Find New Best Dev F1:{}; And Correspodding Test F1:{}".format(epoch, dev_f1, test_f1))
            print("*****************************************************************")
            print("*****************************************************************")

    print("*****************************************************************")
    print("*****************************************************************")
    print("Best Dev F1:{}; And Correspodding Test F1:{}".format(best_dev_f1, cores_test_f1))
    print("*****************************************************************")
    print("*****************************************************************")


    with open(opt.plot_data_path, 'a') as f:
        f.write(json.dumps(res_dict)+'\n')

def train(model, dataLoader, scheduler, optimizer, steps, opt):
    model.train()
    dataIterator = enumerate(dataLoader)
    lossAll = utils.RunningAverage()
    t = trange(steps)

    for i in t:
        idx, data = next(dataIterator)
        loss = model(data)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.clip_grad)
        optimizer.step()
        lossAll.update(loss.item())
        t.set_postfix(allLoss='{:05.5f}/{:05.5f}'.format(lossAll(), loss.item()))
    scheduler.step()
    return lossAll()

def predict(model, dataLoader, steps, opt, id2label, case=0):
    model.eval()
    if case == 0:
        out_path = opt.train_out_path
    elif case == 1:
        out_path = opt.dev_out_path
    else:
        out_path = opt.test_out_path
    f = open(out_path, 'w')
    with torch.no_grad():
        data_interator = enumerate(dataLoader)
        t = trange(steps)
        for i in t:
            idx, data = next(data_interator)
            out = model(data, train=False)
            f.write(something)
        evaluate(out_path)
    return something


if __name__ == '__main__':
    fire.Fire()
