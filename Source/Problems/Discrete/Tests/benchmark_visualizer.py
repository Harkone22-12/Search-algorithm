import matplotlib.pyplot as plt
import os
import numpy as np

class BenchmarkVisualizer:
    def __init__(self, output_dir="Lab1_results"):
        # Lấy đường dẫn tuyệt đối của thư mục đang chứa file này (chính là folder 'Tests')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Nối đường dẫn folder Tests với tên 'Lab1_results'
        self.output_dir = os.path.join(current_dir, output_dir)
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Định nghĩa các kiểu nét, độ dày và marker để chống trùng lặp
        self.line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        self.line_widths = [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 2.0]
        self.markers = ['o', 's', '^', 'D', 'v', '<', 'p']

    def generate_all_plots(self, df, convergence_data):
        self._plot_robustness_boxplots(df)
        self._plot_convergence(df, convergence_data)
        self._plot_scalability(df)
        self._plot_bar_charts(df)

    def _plot_robustness_boxplots(self, df):
        """Tiêu chí: Robustness (Mean ± Std) & Best/Average Solution Quality"""
        for prob in df['Problem'].unique():
            df_prob = df[df['Problem'] == prob]
            
            data_to_plot, labels = [], []
            for _, row in df_prob.iterrows():
                valid_costs = [c for c in row['Raw_Costs'] if not np.isinf(c)]
                if valid_costs:
                    data_to_plot.append(valid_costs)
                    labels.append(row['Algorithm'])
            
            if data_to_plot:
                plt.figure(figsize=(10, 6))
                plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
                plt.title(f'Robustness & Solution Quality (Boxplot) - {prob}')
                plt.ylabel('Cost Distribution (Lower is Better)')
                plt.xlabel('Algorithm')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'Robustness_{prob}.png'))
                plt.close()

    def _plot_convergence(self, df, convergence_data):
        """Tiêu chí: Convergence speed & Exploration vs Exploitation"""
        for prob in df['Problem'].unique():
            plt.figure(figsize=(10, 6))
            has_data = False
            
            algorithms = df[df['Problem'] == prob]['Algorithm'].tolist()
            for i, algo in enumerate(algorithms):
                key = f"{prob}_{algo}"
                if key in convergence_data:
                    plt.plot(
                        convergence_data[key], 
                        label=algo, 
                        linestyle=self.line_styles[i % len(self.line_styles)],
                        linewidth=self.line_widths[i % len(self.line_widths)],
                        alpha=0.8 # Độ trong suốt giúp nhìn xuyên thấu
                    )
                    has_data = True
            
            if has_data:
                plt.title(f'Convergence Speed - {prob}')
                plt.xlabel('Iterations')
                plt.ylabel('Average Cost')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'Convergence_{prob}.png'))
            plt.close()

    def _plot_scalability(self, df):
        """Tiêu chí: Scalability (Computational complexity)"""
        problem_types = df['Problem_Type'].unique()
        for p_type in problem_types:
            df_type = df[df['Problem_Type'] == p_type]
            sizes = df_type['Size'].unique()
            
            if len(sizes) > 1:
                plt.figure(figsize=(10, 6))
                algorithms = df_type['Algorithm'].unique()
                
                for i, algo in enumerate(algorithms):
                    df_algo = df_type[df_type['Algorithm'] == algo].sort_values('Size')
                    
                    # Thêm jitter (độ lệch X) li ti để các điểm không đè bẹp lên nhau
                    jitter = (i - len(algorithms) / 2) * 0.05 
                    
                    plt.plot(
                        df_algo['Size'], 
                        df_algo['Mean_Time'], 
                        marker=self.markers[i % len(self.markers)], 
                        linestyle=self.line_styles[i % len(self.line_styles)],
                        linewidth=2,
                        alpha=0.8,
                        label=algo
                    )
                
                plt.title(f'Scalability (Time Complexity) - {p_type}')
                plt.xlabel('Problem Size (Dimensions)')
                plt.ylabel('Time (Seconds)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Bật hệ Logarit cho trục Y nếu chênh lệch thời gian quá lớn
                if df_type['Mean_Time'].max() > 0.05 and df_type['Mean_Time'].min() < 0.001:
                     plt.yscale('symlog', linthresh=1e-4) # Symlog hiển thị tốt cả số 0 và số lớn
                     plt.ylabel('Time (Seconds - Log Scale)')

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'Scalability_{p_type}.png'))
                plt.close()

    def _plot_bar_charts(self, df):
        """Vẽ biểu đồ Thời gian cho từng bài toán cụ thể"""
        for prob in df['Problem'].unique():
            df_prob = df[df['Problem'] == prob]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(df_prob['Algorithm'], df_prob['Mean_Time'], color='lightgreen')
            plt.title(f'Average Execution Time - {prob}')
            plt.ylabel('Time (Seconds)')
            plt.xticks(rotation=45)
            self._add_labels(bars)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'Time_{prob}.png'))
            plt.close()

    def _add_labels(self, bars):
        for bar in bars:
            yval = bar.get_height()
            if not np.isinf(yval):
                plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)