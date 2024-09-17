
from abc import ABC, abstractmethod

class MessageHandler(ABC):
    @abstractmethod
    def send_message(self, message: str) -> None:
        pass

    @abstractmethod
    def receive_message(self) -> str:
        pass

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def create_connection(self) -> None:
        pass

    @staticmethod
    def getMessageHandler(config):
        if config['type'] == "win32":
            from message_handler.Win32PipeHandler import Win32PipeMessageHandler
            return Win32PipeMessageHandler(config['pipe_name'])
        elif config['type'] == "WebSocket":
            from message_handler.WebSocketHandler import WebSocketHandler
            return WebSocketHandler(config['url'])
        
        else:
            raise Exception("Invalid message handler configuration")

