import urllib3
import urllib
import json
from .base_api import BaseApi

http = urllib3.PoolManager()


"""Api for self-algorithm testing"""
class AlgApi(BaseApi):
    def __init__(self, base_url):
        super().__init__(base_url)
        self.root_url = '/api'

    def infer(self, file_id, task_id):
        url = '/infer'
        params = {
            'file_id': file_id,
            'task_id': task_id
        }
        return self._post_req(url, params)

    def infer_demo(self):
        url = '/infer/demo'
        return self._get_req(url, {})

    def about(self):
        url = '/about'
        return self._get_req(url, {})