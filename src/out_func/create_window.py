import tkinter as tk

# ラジオボタンを作成
def create_radio_button(window, text, variable, value, padx=10, pady=10, side=tk.LEFT):
    radio_button = tk.Radiobutton(window, text=text, variable=variable, value=value)
    radio_button.pack(padx=padx, pady=pady, side=side)

# ボタンを作成
def create_button(window, text, command, padx=10, pady=10, side=tk.LEFT):
    button = tk.Button(window, text=text, command=command)
    button.pack(padx=padx, pady=pady, side=side)

# ラベル付きのスピンボックスを作成
def create_spinbox(window, text, variable, from_, to, increment, padx=10, pady=10):
    # フレームを作成し、ウィンドウに配置
    frame = tk.Frame(window)
    frame.pack(padx=padx)

    # ラベルを作成
    label = tk.Label(frame, text=text)
    label.pack(side=tk.LEFT)
    
    # 数値を変更可能なスピンボックスを作成
    spinbox = tk.Spinbox(frame, from_=from_, to=to, increment=increment, textvariable=variable)
    spinbox.pack(side=tk.LEFT)