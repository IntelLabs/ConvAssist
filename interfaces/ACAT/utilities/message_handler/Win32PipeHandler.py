# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import time
from typing import Any

if sys.platform == "win32":
    import win32file
    import win32pipe
    import win32event
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

        def _is_pipe_overlapped(self) -> bool:
            """
            Check if the pipe is overlapped

            :param pipe_handle: Handle of the pipe
            :return: True if overlapped, False otherwise
            """
            try:
                assert self.pipe_handle is not None
                _, _, pipe_mode, _ = win32pipe.GetNamedPipeInfo(self.pipe_handle.handle)
                if pipe_mode & win32pipe.PIPE_NOWAIT:
                    return True
                else:
                    return False

            except pywintypes.error as e:
                raise Exception(f"Error checking if pipe is overlapped: {e}") from e

        def _get_incoming_message(self) -> Any:
            """
            Get the incoming message from the pipe

            :param Pipehandle: Handle of the pipe
            :return: message received
            """
            try:
                assert self.pipe_handle is not None

                buffer_size = 4096  # For example, 4 KB
                if self._is_pipe_overlapped():
                    # Allocate a buffer for the data
                    read_buffer = win32file.AllocateReadBuffer(buffer_size)

                    # Create an OVERLAPPED structure
                    overlapped = pywintypes.OVERLAPPED()

                    # Create an event for the OVERLAPPED structure
                    overlapped.hEvent = win32event.CreateEvent(None, True, False, None)

                    # Start the overlapped read operation
                    err_code, _ = win32file.ReadFile(self.pipe_handle.handle, read_buffer, overlapped)

                    # If the operation is pending, wait for it to complete
                    if err_code == winerror.ERROR_IO_PENDING:
                        timeout = 100  # milliseconds
                        win32event.WaitForSingleObject(overlapped.hEvent, timeout)

                        # Get the result of the overlapped operation
                        n_bytes_read = win32file.GetOverlappedResult(
                            self.pipe_handle.handle, overlapped, True
                        )

                        # Extract the data from the buffer
                        data = bytes(read_buffer[:n_bytes_read]).decode("utf-8")  # type: ignore

                    elif err_code == win32event.WAIT_TIMEOUT:
                        raise TimeoutError("Timeout waiting for overlapped operation to complete")

                else:
                    # Read the data from the pipe
                    _, data = win32file.ReadFile(self.pipe_handle.handle, buffer_size)

                # Load the data as json and return
                return json.loads(data)

            except pywintypes.error as e:
                raise BrokenPipeError(f"Error receiving message from named pipe: {e}") from e

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
                    win32file.FILE_FLAG_OVERLAPPED,
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
            try:
                if self.pipe_handle:
                    win32file.FlushFileBuffers(self.pipe_handle.handle)
                    win32file.CloseHandle(self.pipe_handle.handle)

            except Exception as e:
                # raise Exception(f"Error disconnecting named pipe: {e}") from e
                pass
