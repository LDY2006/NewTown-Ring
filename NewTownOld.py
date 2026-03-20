# Python 上位机代码（Tkinter + Matplotlib 实时显示和处理CCD数据）
import serial
import threading
import numpy as np
import tkinter as tk
from scipy.signal import find_peaks
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# 波长 (单位: 像素->m 可能需要校准，示例假设单位不变)
lambda_nm = 632.8e-9  # 光波长 (米)
pixel_pitch = 0.028e-3  # CCD像素间距 (米)，根据模块规格设定，如0.028 mm
lambda_pixel = lambda_nm / pixel_pitch  # 等效的波长单位为像素^2*R计算

class NewtonRingsApp:
    def __init__(self, master, port='COM7', baud=115200):
        self.master = master
        master.title("牛顿环曲率测量")
        self.ser = None
        self.running = False
        
        # 创建Matplotlib图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8,4))
        self.line_raw, = self.ax1.plot(np.zeros(128), label='原始信号')
        self.line_filt, = self.ax1.plot(np.zeros(128), label='滤波后信号')
        self.ax1.set_ylim(0, 260)
        self.ax1.set_xlabel('像素序号')
        self.ax1.set_ylabel('灰度值')
        self.ax1.set_title('CCD灰度分布')
        self.ax1.legend(loc='upper right')
        self.scatter = self.ax2.scatter([], [], color='C2', label='r^2 vs k')
        self.fit_line, = self.ax2.plot([], [], color='C3', label='拟合直线')
        self.ax2.set_xlabel('干涉级数 k')
        self.ax2.set_ylabel(r'$r^2$ (像素$^2$)')
        self.ax2.set_title('线性拟合')
        self.ax2.legend(loc='upper left')
        
        # 嵌入Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 按钮控制
        frame = tk.Frame(master)
        frame.pack(side=tk.BOTTOM)
        tk.Button(frame, text='开始', command=self.start).pack(side=tk.LEFT)
        tk.Button(frame, text='停止', command=self.stop).pack(side=tk.LEFT)
        self.labelR = tk.Label(frame, text="曲率半径 R = 0")
        self.labelR.pack(side=tk.LEFT)
        
        self.ser_port = port
        self.baud = baud

    def start(self):
        if not self.running:
            try:
                self.ser = serial.Serial(self.ser_port, self.baud, timeout=1)
            except Exception as e:
                print(f"串口打开失败: {e}")
                return
            self.running = True
            threading.Thread(target=self.read_serial, daemon=True).start()

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()

    def read_serial(self):
        while self.running:
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line: 
                continue
            # 解析串口数据为整数列表
            try:
                data = list(map(int, line.split(',')))
            except:
                continue
            if len(data) != 128:
                continue
            gray = np.array(data)
            
            # FFT降噪：简单低通滤波
            fft = np.fft.fft(gray)
            # 保留低频分量，零掉高频噪声
            cutoff = 10  # 可调整截断点
            fft[cutoff:-cutoff] = 0
            gray_filt = np.real(np.fft.ifft(fft))
            
            # 峰值检测
            peaks, _ = find_peaks(gray_filt, height=np.mean(gray_filt))
            # 找中心点（假设暗点为全局最小值）
            center_idx = np.argmin(gray_filt)
            # 计算每个峰到中心的距离，确定半径r
            r_values = np.abs(peaks - center_idx)
            k_values = np.arange(1, len(r_values)+1)  # 假设峰按顺序对应1,2,...
            r2 = r_values**2
            
            # 线性拟合 r^2 = a * k + b
            if len(r_values) >= 2:
                coeff = np.polyfit(k_values, r2, 1)
                a, b = coeff
                R_pixel = a / lambda_pixel  # 曲率半径 (单位: 像素^2 / 像素 -> 像素，换算成米需乘pixel_pitch^2)
                R_m = R_pixel * (pixel_pitch**2)  # 转换为米
                # 更新界面显示
                self.labelR.config(text=f"曲率半径 R ≈ {R_m*1e3:.2f} mm")
                # 绘制拟合直线
                k_line = np.array([1, k_values[-1]])
                fit_line = a * k_line + b
                # 更新散点与拟合线图
                self.ax2.cla()
                self.ax2.scatter(k_values, r2, color='C2', label='r^2 vs k')
                self.ax2.plot(k_line, fit_line, color='C3', label='拟合直线')
                self.ax2.set_xlabel('干涉级数 k')
                self.ax2.set_ylabel(r'$r^2$ (像素$^2$)')
                self.ax2.legend(loc='upper left')
            
            # 更新灰度分布图
            self.line_raw.set_ydata(gray)
            self.line_filt.set_ydata(gray_filt)
            self.ax1.draw_artist(self.ax1.patch)
            self.ax1.draw_artist(self.line_raw)
            self.ax1.draw_artist(self.line_filt)
            self.canvas.draw_idle()

        self.canvas.draw()

if __name__ == '__main__':
    root = tk.Tk()
    app = NewtonRingsApp(root, port='COM7', baud=115200)  # 修改为实际串口
    root.mainloop()
