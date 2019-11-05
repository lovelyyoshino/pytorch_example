from texttable import Texttable
from ipdb import set_trace

class DefaultConfig(object):
    """
    user can set default hyperparamter here
    hint: don't use 【parse】 as the name
    """

    model = 'BertCrfNer'
    # 路径相关
    path = 'xxx'

    seed = 9979

    use_gpu = 1
    gpu_id = 1

    # 优化器:
    lr = 2e-5
    clip_grad = 2
    warmup = 0.1
    grad_acc_steps = 1

    # 输入相关
    max_length = 128
    train_batch_size = 16
    dev_batch_size = 64
    test_batch_size = 64

    num_epochs = 30

    # 模型区分参数
    print_opt = 'DEF'

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        """
        一些依赖于其他超参数的超参数设置
        """
        setattr(self, 'print_opt', "model_{}_lr_{}_bs_{}".format(self.model, self.lr, self.train_batch_size))
        path = ['train_out_path', 'dev_out_path', 'test_out_path']
        for each in path:
            setattr(self, each, getattr(self, each) + '_'+ self.print_opt)

        """
        print the information of hyperparater
        """
        t = Texttable()
        t.add_row(["Parameter", "Value"])
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not (k.startswith('__') or k == 'parse'):
                t.add_row(["P:" + str(k), "V:" + str(getattr(self, k))])
        print(t.draw())


opt = DefaultConfig()
