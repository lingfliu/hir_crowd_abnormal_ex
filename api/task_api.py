import json
import paho.mqtt.client as mqtt_client
import random

class TaskApi:
    def __init__(self, ip, port, username, password):
        self.status = 'disconnected'
        self.topic = '/hir/alg/crowd'
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.client = None

    def connect(self):

        def on_publish(client, userdata, mid):
            print ("message published")

        # TODO: the on_connect callback not working
        def on_connect(client, userdata, flags, rc):
            print('on connect event')
            if rc == 0:
                print("Connected to MQTT Broker!")
                self.status = 'connected'
            else:
                print("Failed to connect, return code %d\n", rc)
                self.status = 'disconnected'

        client_id = f'hir-alg-mqtt-{random.randint(0, 100000)}'
        self.client = mqtt_client.Client(client_id)
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = on_connect
        self.client.on_publish = on_publish
        self.client.connect(host=self.ip, port=self.port, keepalive=45)
        return self.client

    def update_task(self, id, type,  status, finish_time, result_file_name, result_params):
        # 每次调用重新发起连接，防止上一次连接失效
        self.connect()
        params = {
            'task_id': id,
            'status': status,
            'type': type,
            'finish_time': finish_time,
            'result_file_name': result_file_name,
            'result_params': json.dumps(result_params)
        }
        f = json.dumps(params)
        print('publishing message')
        self.client.publish(self.topic, f)