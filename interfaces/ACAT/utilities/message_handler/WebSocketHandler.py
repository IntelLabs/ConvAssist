# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

from websockets.server import serve
from websockets.sync.client import connect

from .MessageHandler import MessageHandler


class WebSocketHandler(MessageHandler):
    def __init__(self, config):
        self.config = config

    def connect(self) -> tuple[bool, str]:
        if not self.config["use_tls"]:
            warnings.warn(
                f"{self.__class__.__name__} is being used in an insecure manner, and should be used with caution.",
                RuntimeWarning,
                stacklevel=2,
            )

        else:
            # TODO Implement TLS Support
            pass

        self.ws = connect(self.config["url"])
        return True, ""

    def send_message(self, message: str) -> None:
        self.ws.send(message)

    def receive_message(self) -> str:
        return str(self.ws.recv())

    def disconnect(self) -> None:
        self.ws.close()

    def create_connection(self) -> None:
        return super().create_connection()
