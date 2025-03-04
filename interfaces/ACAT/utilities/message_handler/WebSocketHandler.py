# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings
import websockets.sync.server

from MessageHandler import MessageHandler, MessageHandlerException

class WebSocketHandler(MessageHandler):
    def __init__(self):
        pass

    def _handler(self, ws, message_processor):

        try:
            while True:
                message = ws.recv()
                response = message_processor(message)
                ws.send(response)
        
        except Exception as e:
            print(f"Error: {e}")

    def startMessageHandler(self) -> None:
        warnings.warn(
            f"{self.__class__.__name__} is being used in an insecure manner, and should be used with caution.",
            RuntimeWarning,
            stacklevel=2,
        )

        # Make sure the config is set
        if not self.config:
            error = ValueError("Config not set")
            raise MessageHandlerException(str(error))
        
        elif "port" not in self.config:
            error = ValueError("Port not set in config")
            raise MessageHandlerException(str(error))
        
        elif "message_processor" not in self.config:
            error = ValueError("Message Processor not set in config")
            raise MessageHandlerException(str(error))
        
        elif not callable(self.config["message_processor"]):
            error = ValueError("Message Processor is not callable")
            raise MessageHandlerException(str(error))

        self.server = websockets.sync.server.serve(self._handler, "localhost", self.config["port"])
        with self.server:
            print (f"WebSocket Server running on ws://localhost:{self.config["port"]}")
            print ("Waiting for clients to connect...")
            self.server.serve_forever()
