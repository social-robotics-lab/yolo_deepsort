import json
import stomp

class AmqClient(object):
    def __init__(self, ip='localhost', port='61613'):
        self.ip    = ip
        self.port  = port
        self.conns = []
        self.conn4pub = None

    def subscribe(self, topic, listener):
        conn = stomp.Connection([(self.ip, self.port)])
        conn.set_listener('', listener)
        conn.connect(wait=True)
        conn.subscribe(destination='/topic/' + topic, id=topic)
        self.conns.append(conn)

    def publish(self, topic, data=dict()):
        if not self.conn4pub:
            self.conn4pub = stomp.Connection([(self.ip, self.port)])
            self.conn4pub.connect(wait=True)
        self.conn4pub.send('/topic/{}'.format(topic), json.dumps(data))

    def close(self):
        for conn in self.conns: conn.disconnect()
        if self.conn4pub: self.conn4pub.disconnect()
        self.conss = []
        self.conn4pub = None


class Listener(stomp.ConnectionListener):
    def __init__(self, q):
        super().__init__()
        self.q = q
    
    def on_error(self, frame):
        print(f'received an error {frame.body}')

    def on_message(self, frame):
        timestamp = frame.headers['timestamp']
        topic = frame.headers['destination'].split('/')[-1]
        packet = dict(timestamp=timestamp, event=topic, data=json.loads(frame.body))
        self.q.put(packet)