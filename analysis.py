"""
用于分析错误报告，如将最终将文件转换为MarkDown形式
"""
import re
import sys
from ipdb import set_trace


def anaysis_for_md(predict_file, out_file):
    pass

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python evaluate.py predict_file anaysis_result_file")
        exit(1)
    predict_file = sys.argv[1]
    anaysis_result_file = sys.argv[3]
    anaysis_for_md(predict_file, anaysis_result_file)

