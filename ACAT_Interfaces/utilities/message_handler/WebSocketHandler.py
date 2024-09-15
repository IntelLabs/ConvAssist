import warnings
from websocket import create_connection
from message_handler import MessageHandler


class InsecureWarning(Warning):
    pass

def insecure_class_warning(cls):
    class Wrapped(cls):
        def __init__(self, *args, **kwargs):
            warnings.warn(f"{cls.__name__} is insecure and should be used with caution.", InsecureWarning, stacklevel=2)
            super().__init__(*args, **kwargs)
    return Wrapped

@insecure_class_warning
class WebSocketHandler(MessageHandler):
    def __init__(self, config):
        self.config = config

    def connect(self) -> bool:
        self.ws = create_connection(self.config['url'])
        return True

    def send_message(self, message: str) -> None:
        self.ws.send(message)

    def receive_message(self) -> str:
        return str(self.ws.recv())

    def disconnect(self) -> None:
        self.ws.close()