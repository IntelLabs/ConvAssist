import os
import PIL.Image
import pystray
from pystray import MenuItem as item
from tkinter import END,  BOTH, VERTICAL, \
    Button, Tk, ttk, Frame, Label, Scrollbar, PhotoImage, Text, messagebox

from tkinter.scrolledtext import ScrolledText
import logging

license_text_string = "Copyright (C) 2023 Intel Corporation\n"
license_text_string += "SPDX-License-Identifier: Apache-2.0\n"
license_text_string += "This is a demo application for the ConvAssist\n"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


# Custom logging handler that redirects log messages to a Tkinter ScrolledText widget
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(END, msg + '\n')
        self.text_widget.configure(state='disabled')
        # Autoscroll to the bottom
        self.text_widget.yview(END)
        self.text_widget.update_idletasks()

class ConvAssistWindow(Tk):
    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)

        # self.geometry("600x350")
        self.title("ConvAssist")
        taskbar_icon = os.path.join(SCRIPT_DIR, "Assets", "icon_tray.ico")
        self.iconbitmap(taskbar_icon, default=taskbar_icon)

        self.configure(bg='grey')

        self.logger = logger

        self.log_widget = ScrolledText(self, state='disabled', height=10)
        self.log_widget.pack(fill=BOTH, expand=True)

        # Create the custom handler and set the formatter
        text_handler = TextHandler(self.log_widget)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        text_handler.setFormatter(formatter)
        self.logger.addHandler(text_handler)

        self.clear_image = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "button_clear.png"))
        self.license_image = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "button_license.png"))
        self.close_image = PhotoImage(file=os.path.join(SCRIPT_DIR, "Assets", "button_exit.png"))

        self.create_buttons()

        self.withdraw()

    def create_buttons(self):
        button_frame = Frame(self, bg='grey', pady=10)

        style = ttk.Style()
        style.configure('TButton', borderwidth=0, highlightthickness=0,)
        style.map('TButton',
           background=[('pressed', 'grey'), ('active', 'grey'), ])

        clear_button = ttk.Button(button_frame, image=self.clear_image, command=self.clear_action, style='TButton')
        license_button = ttk.Button(button_frame, image=self.license_image, command=self.license_action, style='TButton')
        close_button = ttk.Button(button_frame, image=self.close_image, command=self.deiconify, style='TButton')

         # Pack the buttons into the frame
        clear_button.pack(side='left', fill='both', expand=True)
        license_button.pack(side='left', fill='both', expand=True)
        close_button.pack(side='left', fill='both', expand=True)
        button_frame.pack(side='bottom', fill='x', expand=False)

    def clear_action(self):
        self.log_widget.delete(1.0, END)

    def license_action(self):
        messagebox.showinfo("License", license_text_string)

    # def close_action(self):
    #     self.withdraw()
    #     # self.destroy()

    # def show_action(self):
    #     self.deiconify()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    app = ConvAssistWindow(logger)
    app.mainloop()


    
# licesnce_text_string2 = "Copyright (c) 2013-2017 Intel Corporation\n" \
#                 "Licensed under the Apache License, Version 2.0 (the License);\n" \
#                 "you may not use this file except in compliance with the License.\n" \
#                 "You may obtain a copy of the License at\n" \
#                 "->  http://www.apache.org/licenses/LICENSE-2.0\n" \
#                 "Unless required by applicable law or agreed to in writing, software\n" \
#                 "distributed under the License is distributed on an AS IS BASIS,\n" \
#                 "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n" \
#                 "See the License for the specific language governing permissions and\n" \
#                 "limitations under the License.\n" \
#                 "ConvAssist is built on Pressagio, that is a library that predicts text \n" \
#                 "based on n-gram models (https://pressagio.readthedocs.io, https://github.com/Poio-NLP/pressagio).  \n "\
#                 "Pressagio is a pure Python port of the presage library: https://presage.sourceforge.io \n "\
#                 "and is part of the Poio project: https://www.poio.eu. \n"\
#                 " \n" \
#                 + version_ConvAssist


# licesnce_text_string = "Presage, an extensible predictive text entry system\n" \
#                 "Copyright (C) 2008  Matteo Vescovi <matteo.vescovi@yahoo.co.uk>\n" \
#                 "This program is free software; you can redistribute it and/or modify\n" \
#                 "it under the terms of the GNU General Public License as published by\n" \
#                 "the Free Software Foundation; either version 2 of the License, or\n" \
#                 "(at your option) any later version.\n" \
#                 "This program is distributed in the hope that it will be useful,\n" \
#                 "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" \
#                 "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n" \
#                 "GNU General Public License for more details.\n" \
#                 "You should have received a copy of the GNU General Public License along\n" \
#                 "with this program; if not, write to the Free Software Foundation, Inc.,\n" \
#                 "51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.\n" \
#                 " \n" \
#                 "v 1.0.0.A"