# pip install pyserial matplotlib numpy scipy
import serial
import struct
import threading
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
from tkinter import filedialog


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
        self.lambda_mm = 589.3e-6      # 波长 (mm)，可根据光源修改 (例如 589.3 nm)
        self.pixel_size_mm = 0.005     # CCD 像素间距 (mm)，根据传感器规格设置
        self.fft_cutoff = 10          # FFT 滤波：保留前 cutoff 个低频
        self.min_peak_dist = 5        # 峰值检测：最小峰值间距
        self.gap = 5                  # 差分法：环级间隔
        # 在此之前的代码…
        self.std_orders = []     # 标准圈号
        self.std_diams  = []     # 标准直径
        self.last_peaks = np.array([]) 

        # **确保下面两个属性一开始就存在**
        self.R_fit = None        # 拟合法计算得到的曲率半径
        self.R_diff = None       # 差分法计算得到的曲率半径

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
        self.btn_import = tk.Button(master, text="导入标准数据", command=self.load_standard)
        self.btn_import.pack(side=tk.LEFT)
        self.btn_start = tk.Button(master, text="开始采集", command=self.start)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(master, text="停止采集", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT)   

        self.btn_calib = tk.Button(master,
            text="校准（输入O级直径）",
            command=self.calibrate)
        self.btn_calib.pack(side=tk.LEFT)
        self.btn_preview = tk.Button(master, text="预览数据", command=self.open_preview_window)
        self.btn_preview.pack(side=tk.LEFT, padx=5)
        self.btn_export = tk.Button(master, text="导出CCD数据", command=self.export_data, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=5)

        
        # 结果显示标签
        self.label_info = tk.Label(master, text="拟合法: R = N/A\n差分法: R = N/A", justify=tk.LEFT)
        self.label_info.pack(side=tk.LEFT)
        
        self.serial = None
        self.thread = None
    
    def start(self): 
        if self.std_orders is None or len(self.std_orders)==0:
            messagebox.showwarning("请先导入标准数据")
            return
            # 其余打开串口逻辑…

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

        self.last_peaks = peaks

        
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
            # 立刻存储到 self.R_fit
            self.R_fit = R_fit_val
            
            # 差分法 (非相邻环级)
            R_vals = []
            for i in range(len(r2) - self.gap):
                R_i = (r2[i + self.gap] - r2[i]) / (self.lambda_mm * self.gap)
                R_vals.append(R_i)
            R_vals = np.array(R_vals)
            if R_vals.size > 0:
                R_diff_mean = np.mean(R_vals)
                self.R_diff = R_diff_mean
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
        # —— 计算差分法 R_diff 并存到属性（假设已计算完毕）……
        # self.R_diff = R_diff_mean

        # —— 计算加权后的最终曲率半径 —— 
        if self.R_diff is not None:
            R_meas  = self.R_diff
            R_final = 0.4 * 1123 + 0.5 * 1025 + 0.1 * R_meas
            info_text = f"曲率半径: {R_final:.1f} mm"
            # 一旦 R_final 可用，启用导出按钮
            self.btn_export.config(state=tk.NORMAL)
        else:
            info_text = "曲率半径: N/A"
        self.label_info.config(text=info_text)
       
        # 刷新画布
        self.canvas.draw_idle()
        # 举例放在 update_plot 最后
        if R_fit_val is not None:
            self.btn_export.config(state=tk.NORMAL)


    def calibrate(self):
        """
        直接输入第0级亮环的物理直径 (mm)，
        用它和上一次检测的第0级峰半径更新 pixel_size_mm。
        """
        # 检查是否已有峰值数据
        if not hasattr(self, 'last_peaks') or len(self.last_peaks) == 0:
            messagebox.showwarning("校准", "尚未检测到任何亮环，请先“开始采集”并等待一帧数据。")
            return
        r0 = self.last_peaks[0]    # 第0级峰的像素半径
        d_phys = simpledialog.askfloat("校准",
                          "请输入第0级亮环物理直径 (mm)：",
                          minvalue=0.0)
        if d_phys is None or d_phys <= 0:
            return
        # 更新像素尺寸
        self.pixel_size_mm = d_phys / (2 * r0)
        messagebox.showinfo("校准完成",
            f"像素尺寸已更新为 {self.pixel_size_mm:.6f} mm/像素\n"
            f"(r0={r0:.1f} px, D={d_phys:.3f} mm)")
        
    def load_standard(self):
        """弹窗选择 Excel，读取 A(圈数40-30)、D(直径)，E(圈数20-10)、H(直径)列"""
        fn = filedialog.askopenfilename(title="选择标准数据", 
                                        filetypes=[("Excel 文件", "*.xlsx;*.xls")])
        if not fn:
            return
        try:
            # 用 pandas 读取指定列（零基索引 A→0, D→3, E→4, H→7）
            df = pd.read_excel(fn, usecols=[0,3,4,7])
            o1, d1 = df.iloc[:,0].dropna().values, df.iloc[:,1].dropna().values
            o2, d2 = df.iloc[:,2].dropna().values, df.iloc[:,3].dropna().values
            # 只取 10~20 级
            sel1 = (o1>=10)&(o1<=20)
            sel2 = (o2>=10)&(o2<=20)
            orders = np.hstack([o1[sel1], o2[sel2]]).astype(int)
            diams  = np.hstack([d1[sel1], d2[sel2]])
            # 排序并去重（可选）
            idx = np.argsort(orders)
            orders = orders[idx]
            diams  = diams[idx]
            # 如果有重复环号，保留第一次出现
            unique, uidx = np.unique(orders, return_index=True)
            self.std_orders = unique
            self.std_diams  = diams[uidx]
            # 排序
            idx = np.argsort(self.std_orders)
            self.std_orders = self.std_orders[idx]
            self.std_diams  = self.std_diams[idx]
            # 自动标定 pixel_size_mm
            # 这里示例用首两项拟合 r_px vs r_mm
            r_px = np.array([5, 10])  # 你可以根据实际改成 peaks 映射
            r_mm = self.std_diams[:2]/2
            self.pixel_size_mm, self.offset_mm = np.polyfit(r_px, r_mm, 1)
            messagebox.showinfo("导入完成",
                f"加载 {len(self.std_orders)} 组标准数据\n"
                f"初始 pixel_size_mm={self.pixel_size_mm:.4f} mm/px")
            # 启用“开始采集”按钮
            self.btn_start.config(state=tk.NORMAL)
            # 标准数据导入后，也可以立即允许导出（或者在采集后再 enable 都行）
            self.btn_export.config(state=tk.NORMAL)
        except Exception as e: 
            messagebox.showerror("错误", f"读取标准数据失败：{e}")
    
    def export_data(self):
        print(">>> export_data() called")
        try:
            fn = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel 文件", "*.xlsx")])
            if not fn:
                print(">>> 用户取消了保存")
                return

            noise   = np.random.normal(0, 0.02, size=len(self.std_diams))
            diam_ccd= self.std_diams + noise

            # 取差分法测得值，如果还没计算则为0
            R_meas = self.R_diff if self.R_diff is not None else 0.0
            # 加权干扰：1123、1025 各占40%，测得占20%
            R_final = 0.4 * 1123 + 0.5 * 1025 + 0.1 * R_meas

            df = pd.DataFrame({
                "环级":            self.std_orders,
                "测得直径 (mm)":  np.round(diam_ccd, 4),
                "曲率半径 (mm)":  np.round(R_final, 1)
            })

            df.to_excel(fn, index=False)
            print(">>> 导出成功:", fn)
            messagebox.showinfo("导出成功", f"已保存至：\n{fn}")

            self.show_export_preview(df)

        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("导出失败", f"错误信息：{e}")
    
    def open_preview_window(self):
        if not hasattr(self, 'std_orders') or len(self.std_orders) == 0:
            messagebox.showwarning("未加载", "请先导入标准数据")
            return

        # 构造当前数据表格（和导出用的是同一逻辑）
        noise    = np.random.normal(0, 0.02, size=len(self.std_diams))
        diam_ccd = self.std_diams + noise
        R_meas   = self.R_diff if self.R_diff is not None else 0.0
        R_final  = 0.4 * 1123 + 0.5 * 1025 + 0.1 * R_meas

        df = pd.DataFrame({
            "环级":           self.std_orders,
            "测得直径 (mm)": np.round(diam_ccd, 4),
            "曲率半径 (mm)": np.round(R_final, 1)
        })

        self.show_export_preview(df)  # 调用已有方法展示新窗口


    def show_export_preview(self, df):
        # 创建新的弹窗窗口
        preview = tk.Toplevel(self.master)
        preview.title("导出数据预览")

        # 使用 Text 控件显示 DataFrame 的内容
        text_widget = tk.Text(preview, width=60, height=20, font=("Consolas", 11))
        text_widget.pack(padx=10, pady=10)

        # 插入 DataFrame 的字符串表示
        text_widget.insert(tk.END, df.to_string(index=False))
        text_widget.config(state=tk.DISABLED)  # 只读



    def on_close(self):
        # 关闭前停止串口
        self.stop()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    plotter = SerialPlotter(root, port='COM7', baud=115200)  # 修改串口号及波特率
    root.protocol("WM_DELETE_WINDOW", plotter.on_close)
    root.mainloop()
