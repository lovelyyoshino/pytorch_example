import utils
import json
import sys
from ipdb import set_trace

if __name__ == '__main__':
    if(len(sys.argv)) != 2:
        print("usage python3 draw.py res_path")
        exit(1)

    best_res = 0
    best_model = ""
    res_path = sys.argv[1]
    all_res = []

    with open(res_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            """
            在此处修改代码，设置最终评价【epoch_dev_f1】为指标选择模型
            """
            temp = max(line['epoch_dev_f1'])
            if temp > best_res:
                best_model = line['model']
                best_res = temp
            utils.plot_epoch_for_performance_and_loss('./picture', line)
    print("*****************************************************************************")
    print("*****************************************************************************")
    print("最优的模型为:{}; 验证集指标为:{}".format(best_model, best_res))
    print("*****************************************************************************")
    print("*****************************************************************************")
