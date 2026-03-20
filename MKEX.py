import serial
import struct
import threading
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib

class SerialPlotter:
    def __init__(self, master, port='COM7', baud=115200):
        # 1) 配置 Matplotlib 中文及负号正常显示
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        self.master = master
        self.master.title("TSL1401 传感器数据可视化")
        self.port = port
        self.baud = baud
        self.running = False

        # 2) 创建绘图 Canvas
        self.fig, self.ax = plt.subplots(figsize=(12,5))
        self.line, = self.ax.plot([0]*128)     # 初始绘图曲线
        self.ax.set_ylim(0, 1023)              # ADC 范围
        self.ax.set_xlim(0, 126)               
        self.ax.set_title("CCD 线阵数据")
        self.ax.set_xlabel("像元")
        self.ax.set_ylabel("模拟值")

        # 把 Matplotlib Figure 嵌入 Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()

        # 3) 控制按钮：开始 / 停止
        self.btn_start = tk.Button(master, text="开始", command=self.start)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop  = tk.Button(master, text="停止", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT)

        # 串口和线程句柄
        self.serial = None
        self.thread = None

    def start(self):
        if not self.running:
            try:
                # 打开串口
                self.serial = serial.Serial(self.port, self.baud, timeout=1)
            except Exception as e:
                print(f"串口打开失败：{e}")
                return
            self.running = True
            # 启动后台线程读串口
            self.thread = threading.Thread(target=self.read_serial)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.running = False
        if self.serial:
            self.serial.close()
            self.serial = None

    def read_serial(self):
        while self.running:
            try:
                # 1) 读取 260 字节（130×2 Byte）
                data = self.serial.read(260)
                if len(data) == 260:
                    shorts = struct.unpack('<130h', data)
                    self.update_plot(shorts[:128])
            except Exception as e:
                print(f"串口读取错误：{e}")
                self.stop()

    def update_plot(self, data):
        # 将曲线数据设置为最新的 CCD 读数
        self.line.set_ydata(data)
        # 请求重新绘制
        self.canvas.draw_idle()

    def on_close(self):
        self.stop()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    plotter = SerialPlotter(root, port='COM7', baud=115200)  # 请根据实际端口修改
    # 捕捉窗口关闭事件，先关闭串口再退出
    root.protocol("WM_DELETE_WINDOW", plotter.on_close)
    root.mainloop()