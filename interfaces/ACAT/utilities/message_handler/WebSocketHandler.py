# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import threading
import warnings
import websockets.sync.server

from interfaces.ACAT.utilities.message_handler.MessageHandler import MessageHandler, MessageHandlerException

class WebSocketHandler(MessageHandler):

    server: websockets.sync.server.serve = None
    messageprocessor: callable = None
    config: dict[str, object] = {}

    def _handler(self, ws, process_message):

        try:
            while True:
                message = ws.recv()
                response = process_message(message)
                ws.send(response)
        
        except Exception as e:
            print(f"Error: {e}")

    def start_server(self):
        def wrapper_handler(ws):
            self._handler(ws, self.messageprocessor)

        self.server = websockets.sync.server.serve(wrapper_handler, "localhost", self.config["port"])
        with self.server:
            print (f"WebSocket Server running on ws://localhost:{self.config["port"]}")
            print ("Waiting for clients to connect...")
            self.server.serve_forever()


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
        
        self.messageprocessor = self.config["message_processor"]

        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()

    def stopMessageHandler(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server = None