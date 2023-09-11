import urllib3
import urllib
import json

http = urllib3.PoolManager()

class BaseApi:
    def __init__(self, base_url):
        self.base_url = base_url
        self.root_url = ''

    def _get_req(self, url, params):
        param_str = urllib.parse.urlencode(params)
        if (len(params) > 0):
            url = url + '?'+ param_str
        response = http.request('GET', self.base_url + self.root_url + url,
                                headers={'Content-Type': 'application/json'})

        if (response.status == 200):
            return {'result': 'ok',
                    'msg': json.loads(response.data)}
        else:
            print('err', response.status, 'on fetch', 'url')
            return {'result': 'err',
                    'msg': None}
    def _download(self, url, file_name, saved_file_path):
        res = urllib.request.urlretrieve(self.base_url + self.root_url + url+'/'+file_name, saved_file_path)
        return res

    def _upload_multipart(self, url, data, headers=None):

        if headers is None:
            headers = {}
        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'

        with open(data['file'], 'rb') as f:
            file_content = f.read()
            f.close()
            fields = {'file': (data['file'], file_content, 'multipart/form-data')}
            response = http.request('POST', self.base_url + self.root_url + url, fields=fields, headers=headers)
            if response.status == 200:
                return {'result': 'ok',
                        'msg': json.loads(response.data)}
            else:
                print('err', response.status, 'on post', 'url')
                return {'result': 'err',
                        'msg': None}
    def _post_req(self, url, body, headers=None):
        if headers is None:
            headers = {}
        headers['Content-Type'] = 'application/json'
        response = http.request('POST', self.base_url + self.root_url + url, body=json.dumps(body).encode('utf-8'),
                                headers= headers)
        if (response.status == 200):
            return {'result': 'ok',
                    'msg': json.loads(response.data),
                    'headers': response.headers}
        else:
            print('err', response.status, 'on post', 'url')
            return {'result': 'err',
                    'msg': None}