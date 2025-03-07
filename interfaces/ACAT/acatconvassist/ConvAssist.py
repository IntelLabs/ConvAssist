# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# CHATGpt Generated Code
import sys
import os
import logging
import tempfile
from filelock import FileLock, Timeout
LOCK_FILE = os.path.join(tempfile.gettempdir(), 'convassist.lock')

working_dir = os.path.dirname(os.path.realpath(__file__))

LOG_LEVEL = logging.DEBUG

if not sys.platform == "win32":

    def main():
        raise RuntimeError("This script is only supported on Windows.")

else:
    import glob
    import queue
    import shutil
    import sys
    import threading
    import time
    import tkinter as tk
    from tkinter import BOTH, END, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    import pystray
    import sv_ttk
    from PIL import Image
    from pystray import MenuItem

    from interfaces.ACAT.acatconvassist.preferences import Preferences
    from convassist.utilities.logging_utility import LoggingUtility

    license_text_string = "Copyright (C) 2024 Intel Corporation\n"
    license_text_string += "SPDX-License-Identifier: GPL 3.0\n\n"
    license_text_string += "Portions of ConvAssist were ported from the Pressage\n"
    license_text_string += "project, which is licensed under the GPL 3.0 license.\n"


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

            # Load preferences
            self.preferences = Preferences("ACATConvAssist")

            # Set up logging
            log_location = self.preferences.load("pathlog", f"{self.preferences.get_config_dir()}/logs")
            self.logutil = LoggingUtility()
            self.logutil.set_log_location(log_location)
            self.logger = LoggingUtility().get_logger(
                name="CONVASSIST", log_level=LOG_LEVEL, 
                log_file=True,
                queue_handler=True
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

            self.logger.debug("Starting main loop")
            # Start the log update loop
            self.update_log_window()

            # Start the exit check loop
            self.app_quit_event = threading.Event()
            self.check_for_exit()

            self.logger.debug("Loading ACATConvAssistInterface")
            # Time how long it takes to load the ACATConvAssistInterface
            start_time = time.time()

            # Start ACATConvAssistInterface
            from interfaces.ACAT.acatconvassist.acatconvassist import ACATConvAssistInterface
            self.thread = ACATConvAssistInterface(self.app_quit_event, queue_handler=True, log_level = LOG_LEVEL)
            self.logger.debug("ACATConvAssistInterface loaded in %s seconds", time.time() - start_time)
            self.thread.start()

            return super().mainloop(n)

        def configure(self):
            self.withdraw()  # Hide the main window at startup

            self.log_widget = ScrolledText(self, height=10)
            self.log_widget.pack(fill=BOTH, expand=True, pady=10, padx=10)

            self.create_buttons()

            sv_ttk.set_theme("light")

        def create_buttons(self):
            button_frame = ttk.Frame(self, height=50)

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
                    message = self.logutil.central_log_queue.get_nowait()
                    self.log_widget.insert(tk.END, message + "\n")
                    self.log_widget.see(tk.END)  # Auto-scroll to the end
            except queue.Empty:
                pass

            # Schedule the next check
            self.after(50, self.update_log_window)

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
                MenuItem("Show", self.show_window),
                MenuItem("About", self.about_message),
                MenuItem("Quit", self.on_quit),
            )

        def __call__(self):
            self.tk_window.show_window()
            # return super().__call__()

        def check_for_exit(self):
            """Check if the application should exit."""
            if self.tk_window.app_quit_event.is_set():
                self.on_quit(self.icon, None)

            self.tk_window.after(100, self.check_for_exit)

        @staticmethod
        def create_image():
            """Create an image for the systray icon."""
            image = Image.open(os.path.join(working_dir, "Assets", "icon_tray.ico"))
            return image
        
        def on_clicked(self, icon, item):
            """Handle the systray icon click event."""
            self.tk_window.show_window()

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

        # Start the Tkinter main loop
        tk_window.mainloop()


if __name__ == "__main__":
    lock = FileLock(LOCK_FILE)

    try:
        with lock.acquire(timeout=1):
            main()
    except Timeout:
        print("Another instance of ConvAssist is already running.")
        sys.exit()
