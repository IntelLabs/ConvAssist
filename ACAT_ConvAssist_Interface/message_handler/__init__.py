
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

    @staticmethod
    def getMessageHandler(config):
        # if config['type'] == "win32":
        from ACAT_ConvAssist_Interface.message_handler.Win32PipeHandler import Win32PipeMessageHandler
        return Win32PipeMessageHandler(config['pipe_name'])
        # else:
        #     raise Exception("Invalid message handler configuration")

