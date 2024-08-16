"""
 Copyright (C) 2023 Intel Corporation

 SPDX-License-Identifier: Apache-2.0

"""
from tkinter import *


class MessagesWindow:

    def __init__(self):
        self.ws = Tk()
        self.ws.title('Messages Window')
        self.ws.geometry("600x350")
        self.ws.overrideredirect(1)
        self.ws.wm_attributes("-transparentcolor", "grey")
        self.frame_photo = PhotoImage(file='frame.png')
        self.frame_label = Label(self.ws, border=0, bg='grey', image=self.frame_photo)
        self.frame_label.pack(fill=BOTH, expand=True)
        self.ws.resizable(False, False)

        self.label = Label(self.ws, text="ConvAssist - Messages Window", fg="black", bg='#64b4ff', font=("Verdana", 14))
        self.label.place(x=75, y=12.5)
        self.label2 = Label(self.ws, text="Messages from Pipe COM", fg="black", bg='#FFFFFF', font=("Arial", 10))
        self.label2.place(x=25, y=57.5)

        self.text_box = Text(self.ws, font=("Arial", 12))
        self.text_box.place(x=25, y=85, width=526, height=200)
        self.sb = Scrollbar(self.ws, orient=VERTICAL)
        self.sb.place(x=551, y=85, width=25, height=200)
        self.text_box.config(yscrollcommand=self.sb.set)
        self.sb.config(command=self.text_box.yview)

        self.button_image = PhotoImage(file='button.png')
        self.clear_button = Label(self.ws, image=self.button_image, border=0, bg='#FFFFFF', text="Clear")
        self.clear_button.place(x=25, y=300)
        self.clear_button.bind("<Button>", lambda e: self.text_box.delete(1.0, END))

        self.exit_button = Label(self.ws, image=self.button_image, border=0, bg='#FFFFFF', text="Clear")
        self.exit_button.place(x=160, y=300)
        self.exit_button.bind("<Button>", lambda e: self.quit())




