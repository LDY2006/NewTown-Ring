# pip install pyserial matplotlib numpy scipy
import serial
import struct
import threading
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:
    raise ImportError("Please install scipy: pip install scipy")

class SerialPlotter:
    def __init__(self, master, port='COM7', baud=115200):
        # 配置 matplotlib 使用中文字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
        
        self.master = master
        self.master.title("牛顿环曲率半径测量")
        self.port = port
        self.baud = baud
        self.running = False
        
        # 参数设置
        self.lambda_mm = 632.8e-6      # 波长 (mm)，可根据光源修改 (例如 632.8 nm)
        self.pixel_size_mm = 0.005     # CCD 像素间距 (mm)，根据传感器规格设置
        self.fft_cutoff = 10          # FFT 滤波：保留前 cutoff 个低频
        self.min_peak_dist = 5        # 峰值检测：最小峰值间距
        self.gap = 5                  # 差分法：环级间隔
        
        # 创建 matplotlib 图形和子图 (2x2)
        self.fig, ((self.ax_raw, self.ax_filtered),
                   (self.ax_peaks, self.ax_fit)) = plt.subplots(2, 2, figsize=(10, 8))
        
        # 原始灰度数据子图
        self.ax_raw.set_title("原始灰度数据")
        self.ax_raw.set_xlabel("像素")
        self.ax_raw.set_ylabel("灰度值")
        self.line_raw, = self.ax_raw.plot([0]*128, color='C0')
        self.ax_raw.set_xlim(0, 128)
        self.ax_raw.set_ylim(0, 1023)
        
        # 滤波后数据子图
        self.ax_filtered.set_title("滤波后灰度数据")
        self.ax_filtered.set_xlabel("像素")
        self.ax_filtered.set_ylabel("灰度值")
        self.line_filtered, = self.ax_filtered.plot([0]*128, color='C1')
        self.ax_filtered.set_xlim(0, 128)
        self.ax_filtered.set_ylim(0, 1023)
        
        # 峰值检测子图 (滤波后数据 + 峰值点)
        self.ax_peaks.set_title("峰值检测")
        self.ax_peaks.set_xlabel("像素")
        self.ax_peaks.set_ylabel("灰度值")
        self.line_peaks_line, = self.ax_peaks.plot([0]*128, color='C1', label='滤波后')
        self.scatter_peaks, = self.ax_peaks.plot([], [], 'ro', label='峰值')
        self.ax_peaks.set_xlim(0, 128)
        self.ax_peaks.set_ylim(0, 1023)
        self.ax_peaks.legend()
        
        # 拟合结果子图 (r^2 vs k)
        self.ax_fit.set_title("拟合: $r^2$ vs 干涉级数 k")
        self.ax_fit.set_xlabel("干涉级数 k")
        self.ax_fit.set_ylabel("半径平方 $r^2$ ($\mathrm{mm}^2$)")
        self.line_fit_points, = self.ax_fit.plot([], [], 'bo', label='实验数据')
        self.line_fit_line, = self.ax_fit.plot([], [], 'r-', label='线性拟合')
        self.ax_fit.legend()
        
        self.fig.tight_layout()
        
        # 将 matplotlib 图形嵌入 Tkinter 界面
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()
        
        # 控制按钮
        self.btn_start = tk.Button(master, text="开始采集", command=self.start)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(master, text="停止采集", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT)
        
        # 结果显示标签
        self.label_info = tk.Label(master, text="拟合法: R = N/A\n差分法: R = N/A", justify=tk.LEFT)
        self.label_info.pack(side=tk.LEFT)
        
        self.serial = None
        self.thread = None
    
    def start(self):
        if not self.running:
            try:
                self.serial = serial.Serial(self.port, self.baud, timeout=1)
            except Exception as e:
                print(f"串口打开失败: {e}")
                return
            self.running = True
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
                # 读取 260 字节 (130 个 short)，使用前 128 个数据
                data = self.serial.read(260)
                if len(data) == 260:
                    shorts = struct.unpack('<130h', data)
                    raw_data = np.array(shorts[:128], dtype=float)
                    # 在主线程更新图形和计算
                    self.master.after(0, self.update_plot, raw_data)
            except Exception as e:
                print(f"串口读取出错: {e}")
                self.stop()
    
    def update_plot(self, data):
        # 更新原始灰度曲线
        self.line_raw.set_ydata(data)
        
        # FFT 频域滤波
        fft_data = np.fft.fft(data)
        N = len(data)
        cutoff = self.fft_cutoff
        if cutoff < N // 2:
            fft_data[cutoff:N-cutoff] = 0
        filtered = np.real(np.fft.ifft(fft_data))
        self.line_filtered.set_ydata(filtered)
        
        # 峰值检测 (滤波后曲线)
        peaks, _ = find_peaks(filtered, distance=self.min_peak_dist)
        # 忽略中心 (0) 可能出现的峰
        peaks = np.array([p for p in peaks if p != 0])
        
        # 更新峰值检测图: 滤波曲线与峰值点
        self.line_peaks_line.set_ydata(filtered)
        peak_vals = filtered[peaks] if len(peaks) > 0 else []
        self.scatter_peaks.set_data(peaks, peak_vals)
        self.ax_peaks.set_xlim(0, N)
        ymax = np.max(filtered) if len(filtered) > 0 else 1
        self.ax_peaks.set_ylim(0, ymax * 1.1)
        
        R_fit_val = None
        R_diff_mean = None
        R_diff_std = None
        
        # 如果检测到多个峰值，执行拟合和差分计算
        if len(peaks) > 1:
            # 计算亮环半径 (假设中心在像素 0 处)
            r = peaks * self.pixel_size_mm  # 单位 mm
            r2 = r**2
            k = np.arange(1, len(r2) + 1)  # 干涉级数
            
            # 拟合法: r^2 = k * λ * R
            slope, intercept = np.polyfit(k, r2, 1)
            R_fit_val = slope / self.lambda_mm
            
            # 差分法 (非相邻环级)
            R_vals = []
            for i in range(len(r2) - self.gap):
                R_i = (r2[i + self.gap] - r2[i]) / (self.lambda_mm * self.gap)
                R_vals.append(R_i)
            R_vals = np.array(R_vals)
            if R_vals.size > 0:
                R_diff_mean = np.mean(R_vals)
                if R_vals.size > 1:
                    R_diff_std = np.std(R_vals, ddof=1)
                else:
                    R_diff_std = 0.0
            
            # 更新拟合图: 数据点与拟合直线
            self.line_fit_points.set_data(k, r2)
            x_fit = np.array([0, k[-1] + 1])
            y_fit = slope * x_fit + intercept
            self.line_fit_line.set_data(x_fit, y_fit)
            self.ax_fit.set_xlim(0, k[-1] + 1)
            self.ax_fit.set_ylim(0, np.max(r2) * 1.1)
        else:
            # 若峰值数量不足，不显示拟合结果
            self.line_fit_points.set_data([], [])
            self.line_fit_line.set_data([], [])
        
        # 更新标签显示结果
        info_text = ""
        if R_fit_val is not None:
            info_text += f"拟合法: R = {R_fit_val:.3f} mm\n"
        else:
            info_text += "拟合法: R = N/A\n"
        if R_diff_mean is not None:
            info_text += f"差分法: R = {R_diff_mean:.3f} ± {R_diff_std:.3f} mm"
        else:
            info_text += "差分法: R = N/A"
        self.label_info.config(text=info_text)
        
        # 刷新画布
        self.canvas.draw_idle()
    
    def on_close(self):
        # 关闭前停止串口
        self.stop()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    plotter = SerialPlotter(root, port='COM7', baud=115200)  # 修改串口号及波特率
    root.protocol("WM_DELETE_WINDOW", plotter.on_close)
    root.mainloop()
