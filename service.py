import threading
import time
import datetime

from flask import Flask, request
from infer import infer
from concurrent.futures import ThreadPoolExecutor
from api import FileApi, TaskApi
from PIL import Image
import numpy as np
import zipfile
import os
import random
from config import Config


app = Flask(__name__)

fileApi = FileApi(Config['file_server'])
taskApi = TaskApi(Config['mqtt_host'],
                  Config['mqtt_port'],
                  Config['mqtt_username'],
                  Config['mqtt_password'])

# infer return code:
# 'invalid': parameter invalid
# 'busy': alg is running
# 'submitted': alg submitted
# 'failed': alg failed
# 'done': alg done

TASK_STATUS_INVALID = 'invalid'
TASK_STATUS_BUSY = 'busy'
TASK_STATUS_SUBMITTED = 'submitted'
TASK_STATUS_DONE = 'done'
TASK_STATUS_FAILED = 'failed'

def infer_task(a):
    while a.running:
        time.sleep(0.1)
        if alg.pending:

            fid, task_id, task_type = a.target
            # demo task execution
            if fid == 'data_test' and task_id == 'infer_demo':
                fid = Config['file_demo']

            # 如文件不存在，返回失败
            # TODO：comment out if for offline test
            if not fileApi.isFileExist(fid):
                taskApi.update_task(task_id,
                                    task_type,
                                    TASK_STATUS_FAILED,
                                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    '',
                                    { # result params
                                        'error': 'file not exist'
                                    }
                                    )
                a.target = ()
                a.pending = False
                continue
            fileApi.download(file_name=fid, saved_file_path=tmp_file_name)
            # #

            tmp_file_name = 'vid.mp4'

            labels = infer(tmp_file_name)

            # format time as yyyy-mm-dd hh:mm:ss
            finish_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('finished: ', finish_time)

            # generate result file
            output_file_name = 'output_{}'.format(task_id) + '.dat'
            with open(output_file_name, 'w') as f:
                np.savetxt(f, np.array(labels).astype(int), format('%d'))
                f.close()

            while not os.path.exists(output_file_name):
                time.sleep(0.5)

            """这里不做压缩"""
            response = fileApi.upload(output_file_name)
            result_file_name = response['msg']['data']['new_name']
            taskApi.update_task(task_id,
                                task_type,
                                TASK_STATUS_DONE,
                                finish_time,
                                result_file_name,
                                { # result params
                                }
                                )

            # 清理临时文件
            # os.remove(tmp_file_name)
            # 清空任务
            a.target = ()
            a.pending = False


"""
算法调用类, 采用异步线程，每次只能提交一个任务
"""
class Alg:
    def __init__(self):
        self.running = True
        self.target = ()
        self.infer_task = infer_task
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = False
        self.pool.submit(self.infer_task, self)

    def submit(self, data_file_name, task_id, task_type):
        ret = 0
        if self.pending:
            # 如果当前有算法任务在执行，丢弃该次请求，返回-1
            ret = -1
        else:
            self.target = (data_file_name, task_id, task_type)
            # 这里可能有同步问题，尽量不要采用高并发的模式
            self.pending = True
            ret = 0

        return ret

    def shutdown(self):
        self.running = False
        self.pool.shutdown(wait=False)

alg = Alg()

alg_type = 'crowd_anomaly_dl' # crowd anomaly detection based on deep learning

"""
算法信息
"""
@app.route('/api/about', methods=['GET'])
def about():
    return {
        'name': 'hir_alg_crowd_count',
        'type': alg_type,
        'version': '1.0',
        'description': 'a crowd count algorithm based on vgg19 and bayesian loss',
        'mode': infer.device.type # 算法运行模式：cpu / cuda
    }


"""
算法调用主入口
"""
@app.route('/api/infer', methods=['POST'])
def request_infer():
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('infer request at: ', t)

    task_id = request.json.get('task_id')
    task_type = request.json.get('task_type')
    data_file_name = request.json.get('data_file_name')
    data_time = request.json.get('data_time')
    data_coord = request.json.get('data_coord')

    # data validation
    if not (data_file_name and task_id and data_time and data_coord):
        return {
            'task_id': task_id,
            'code': TASK_STATUS_INVALID
        }

    if task_type != alg_type:
        return {
            'task_id': task_id,
            'code': TASK_STATUS_INVALID
        }

    # submit task
    ret = alg.submit(data_file_name, task_id, task_type)

    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('infer request result', ret, ' at ', t)

    if ret == -1:
        return {
            'task_id': task_id,
            'code': TASK_STATUS_BUSY
        }
    else:
        # return anyway
        return {
            'task_id': task_id,
            'code': TASK_STATUS_SUBMITTED
        }

@app.route('/api/status', methods=['GET'])
def status():
    return {
        'code': 'ok',
        'status': alg.pending,
    }

@app.route('/api/infer/demo', methods=['POST'])
def infer_demo():
    res = alg.submit('data_test', 'infer_demo', alg_type)
    if res == -1:
        return {
            'task_id': 'infer_demo',
            'code': TASK_STATUS_BUSY
        }
    else:
        return {
            'task_id': 'infer_demo',
            'code': TASK_STATUS_SUBMITTED
        }


"""算法结果样例访问接口"""
@app.route('/api/infer/result/demo', methods=['GET'])
def infer_result_demo():
    return {
        'task_id': 'task_id',
        'data_file_name': 'data_file_name',
        'data_time': '2019-01-01 00:00:00',
        'submit_time': '2019-01-01 00:00:00',
        'finish_time': '2019-01-01 00:00:00',
        'type': alg_type,
        'status': TASK_STATUS_DONE,
        'result_file_name': 'result_file_name',
        'result_params': {
            'count': 700,
        }
    }

app.run(host='0.0.0.0', port=10516)