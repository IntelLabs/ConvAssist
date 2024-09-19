# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# CHATGpt Generated Code
import sys

if not sys.platform == "win32":

    def main():
        raise RuntimeError("This script is only supported on Windows.")

else:
    import glob
    import logging
    import os
    import queue
    import shutil
    import sys
    import tempfile
    import threading
    import time
    import tkinter as tk
    from tkinter import BOTH, END, LEFT, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
    from tkinter.ttk import Button

    import psutil
    import pystray
    import sv_ttk
    from ..acat_interface.ACATConvAssistInterface import (
        ACATConvAssistInterface,
    )
    from PIL import Image
    from pystray import MenuItem as item

    from convassist.utilities.logging_utility import LoggingUtility

    license_text_string = "Copyright (C) 2024 Intel Corporation\n"
    license_text_string += "SPDX-License-Identifier: GPL 3.0\n\n"
    license_text_string += "Portions of ConvAssist were ported from the Pressage\n"
    license_text_string += "project, which is licensed under the GPL 3.0 license.\n"

    working_dir = os.path.dirname(os.path.realpath(__file__))

    def findProcessIdByName(process_name):
        """
        Get a list of all the PIDs of all the running process whose name contains
        the given string processName

        :param process_name: Name of process to look
        :return: True if process is running
        """
        listOfProcessObjects = []
        # Iterate over the all the running process
        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict(attrs=["pid", "name", "create_time"])
                # Check if process name contains the given name string.
                if process_name.lower() in pinfo["name"].lower():
                    listOfProcessObjects.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if len(listOfProcessObjects) > 2:
            return True
        return False

    def deleteOldPyinstallerFolders(time_threshold=100):
        """
        Deletes the Temp folders created by Pyinstaller if they were not closed correctly
        :param time_threshold: in seconds
        :return: void
        """
        try:
            if not getattr(sys, "frozen", False):
                return

            temp_path = tempfile.gettempdir()

            # Search all MEIPASS folders...
            mei_folders = glob.glob(os.path.join(temp_path, "_MEI*"))
            count_list = len(mei_folders)
            for item in mei_folders:  # noqa: F402
                try:
                    if (
                        time.time() - os.path.getctime(item)
                    ) > time_threshold:  # and item != base_path:
                        if os.path.isdir(item):
                            pass
                            shutil.rmtree(item)
                except Exception as es:
                    raise es
        except Exception as es:
            raise es

    class ConvAssistWindow(tk.Tk):
        """Main application window."""

        def __init__(self):
            super().__init__()

            self._tray_icon: SysTrayIcon

            self.title("ConvAssist")
            self.iconbitmap(os.path.join(working_dir, "Assets", "icon_tray.ico"))

            # Set up logging
            self.logger = LoggingUtility().get_logger(
                name="CONVASSISTUI", log_level=logging.DEBUG, queue_handler=True
            )
            self.logger.info("Application started")

            # Set up the GUI components
            self.configure()

        @property
        def tray_icon(self):
            return self._tray_icon

        @tray_icon.setter
        def tray_icon(self, value):
            self._tray_icon = value
            self._tray_icon.run_detached()

        @tray_icon.deleter
        def tray_icon(self):
            self._tray_icon.stop()
            del self._tray_icon

        def mainloop(self, n: int = 0) -> None:
            # Start the log update loop
            self.update_log_window()

            # Start the exit check loop
            self.app_quit_event = threading.Event()
            self.check_for_exit()

            # Start ACATConvAssistInterface
            self.thread = ACATConvAssistInterface(self.app_quit_event, queue_handler=True)
            self.thread.start()

            return super().mainloop(n)

        def configure(self):
            self.log_widget = ScrolledText(self, height=10)
            self.log_widget.pack(fill=BOTH, expand=True, pady=10, padx=10)

            self.create_buttons()

            # self.withdraw()  # Hide the main window at startup

            sv_ttk.set_theme("light")

        def create_buttons(self):
            button_frame = ttk.Frame(self, height=50)

            # self.clear_image = ttk.PhotoImage(file=os.path.join(working_dir, "Assets", "button_clear.png"))
            # self.license_image = ttk.PhotoImage(file=os.path.join(working_dir, "Assets", "button_license.png"))
            # self.close_image = ttk.PhotoImage(file=os.path.join(working_dir, "Assets", "button_exit.png"))

            # clear_button = ttk.Button(button_frame, image=self.clear_image, command=self.clear_action)
            # license_button = ttk.Button(button_frame, image=self.license_image, command=self.license_action)
            # close_button = ttk.Button(button_frame, image=self.close_image, command=self.close_action)

            clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_action)
            license_button = ttk.Button(button_frame, text="About", command=self.about_action)
            close_button = ttk.Button(button_frame, text="Close", command=self.close_action)

            # Pack the buttons into the frame
            clear_button.pack(side="right", expand=False, padx=5)
            license_button.pack(side="right", expand=False, padx=5)
            close_button.pack(side="right", expand=False, padx=5)
            button_frame.pack(side="bottom", fill="x", expand=False, padx=10, pady=10)

        def clear_action(self):
            self.log_widget.delete(1.0, END)

        def about_action(self):
            messagebox.showinfo("About", license_text_string)

        def close_action(self):
            self.withdraw()

        def update_log_window(self):
            """Check the log queue for new messages and update the log window."""
            try:
                while True:
                    message = LoggingUtility().central_log_queue.get_nowait()
                    self.log_widget.insert(tk.END, message + "\n")
                    self.log_widget.see(tk.END)  # Auto-scroll to the end
            except queue.Empty:
                pass

            # Schedule the next check
            self.after(100, self.update_log_window)

        def check_for_exit(self):
            """Check if the application should exit."""
            if self.app_quit_event.is_set():
                self.on_closing()

            self.after(100, self.check_for_exit)

        def destroy(self):
            """Handle window close event."""
            # Override default close behavior to hide the window instead
            if not self.app_quit_event.is_set():
                self.hide_window()
            else:
                if self.tray_icon:
                    del self.tray_icon
                super().destroy()

        def hide_window(self):
            """Hide the Tkinter window."""
            self.withdraw()

        def show_window(self):
            """Show the Tkinter window."""
            self.deiconify()

        def on_closing(self):
            """Handle the window closing event."""
            self.app_quit_event.set()
            self.thread.join()
            self.destroy()

    class SysTrayIcon(pystray.Icon):
        """System tray icon."""

        def __init__(self, tk_window: ConvAssistWindow, *args, **kwargs):
            self.tk_window = tk_window
            super().__init__(*args, **kwargs)

            self.icon = self.create_image()

            self.menu = (
                item("Show", self.show_window),
                item("About", self.about_message),
                item("Quit", self.on_quit),
            )

        def check_for_exit(self):
            """Check if the application should exit."""
            if self.tk_window.app_quit_event.is_set():
                self.on_quit(self.icon, None)

            self.tk_window.after(100, self.check_for_exit)

        @staticmethod
        def create_image():
            """Create an image for the systray icon."""
            image = Image.open(
                os.path.join(
                    r"C:\Users\mbeale\source\repos\ConvAssist\ACAT_ConvAssist_Interface\ConvAssistCPApp\Assets\icon_tray.png"
                )
            )
            return image

        def on_quit(self, icon, item):
            """Quit the application."""
            self.tk_window.app_quit_event.set()

        def show_window(self, icon, item):
            """Show the Tkinter window."""
            self.tk_window.show_window()

        def about_message(self, icon, item):
            """Hide the Tkinter window."""
            self.tk_window.about_action()

    def main():
        # Create the main window
        tk_window = ConvAssistWindow()

        # Create the system tray icon
        systray_icon = SysTrayIcon(tk_window, "Tkinter App")

        # Set the tray icon
        tk_window.tray_icon = systray_icon

        # Start the system tray icon
        # systray_icon.run_detached()

        # Start the Tkinter main loop
        tk_window.mainloop()


if __name__ == "__main__":
    main()
