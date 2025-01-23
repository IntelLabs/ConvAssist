# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from typing import Any

if sys.platform == "win32":
    import win32file
    import win32pipe
    import pywintypes
    import winerror

import json

from .MessageHandler import MessageHandler

if not sys.platform == "win32":

    class Win32PipeMessageHandler(MessageHandler):
        def __init__(self, pipe_name: str):
            raise Exception("Win32PipeMessageHandler is only supported on Windows")

        def connect(self):
            raise Exception("Win32PipeMessageHandler is only supported on Windows")

        def disconnect(self) -> None:
            raise Exception("Win32PipeMessageHandler is only supported on Windows")

        def send_message(self, message: str) -> None:
            raise Exception("Win32PipeMessageHandler is only supported on Windows")

        def receive_message(self) -> str:
            raise Exception("Win32PipeMessageHandler is only supported on Windows")

        def create_connection(self) -> None:
            raise Exception("Win32PipeMessageHandler is only supported on Windows")

else:

    class Win32PipeMessageHandler(MessageHandler):
        def __init__(self, pipe_name: str):
            self.pipe_name = pipe_name
            self.pipe_handle = None

        def connect(self) -> tuple[bool, str]:
            return self._ConnectToNamedPipe()

        def disconnect(self) -> None:
            self._DisconnectNamedPipe()

        def send_message(self, message: str) -> None:
            self._send_message(message)

        def receive_message(self) -> str:
            return self._get_incoming_message()

        def create_connection(self) -> None:
            pass

        def _get_incoming_message(self) -> Any:

            assert self.pipe_handle is not None
            buffer_size = 1024  # 1KB at a time
            data = ""

            try:
                while True:
                    result, tmp = win32file.ReadFile(self.pipe_handle.handle, buffer_size)
                    if tmp:
                        data += tmp.decode("utf-8")  # type: ignore

                    if result is not winerror.ERROR_MORE_DATA:
                        break

            except pywintypes.error as e:
                raise BrokenPipeError(f"{e.strerror}") from e

            finally:
                # Load the data as json and return
                if data:
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError as e:
                        #TODO handle this properly, but for now just continue
                        pass

            return data

        def _send_message(self, message: Any) -> None:
            """
            Send the message to the pipe

            :param Pipehandle: Handle of the pipe
            :param message: message to send
            :return: void
            """
            try:
                win32file.WriteFile(self.pipe_handle, message.encode("utf-8"))  # type: ignore
            except pywintypes.error as e:
                raise BrokenPipeError(f"Error sending message to named pipe: {e}") from e

        def _ConnectToNamedPipe(self) -> tuple[bool, str]:
            """
            Set the Pipe as Client

            :param PipeServerName: Name of the pipe
            :return: tuple of clientConnected and handle
            """

            pipeName = rf"\\.\pipe\{self.pipe_name}"

            clientConnected = False
            statusMessage = "Unable to connect."

            try:
                self.pipe_handle = win32file.CreateFile(
                    pipeName,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )

                if self.pipe_handle:
                    win32pipe.SetNamedPipeHandleState(
                        self.pipe_handle.handle, win32pipe.PIPE_READMODE_MESSAGE, None, None
                    )

                    clientConnected = True
                    statusMessage = "Connection successful."
            except pywintypes.error as e:
                if e.args[0] == winerror.ERROR_PIPE_BUSY:
                    statusMessage = "Pipe busy."
                elif e.args[0] == winerror.ERROR_FILE_NOT_FOUND:
                    statusMessage = "Pipe not found."
                else:
                    statusMessage = f"Error connecting to named pipe: {e}"

            return clientConnected, statusMessage

        def _DisconnectNamedPipe(self) -> None:
            """
            Disconnect the named pipe

            :param handle: Handle of the pipe
            :return: void
            """
            try:  # nosec B110
                if self.pipe_handle:
                    win32file.FlushFileBuffers(self.pipe_handle.handle)
                    win32file.CloseHandle(self.pipe_handle.handle)

            except Exception as e:  # nosec B110
                # raise Exception(f"Error disconnecting named pipe: {e}") from e
                pass  # nosec B110
