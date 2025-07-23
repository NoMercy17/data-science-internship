import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class DataCleaningVisualizer:
    def __init__(self, output_dir='/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/cleaned'):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Define cleaning steps and their corresponding files
        self.cleaning_steps = [
            ('raw', 'Raw Data'),
            ('01_missing_values_cleaned.csv', 'Missing Values Cleaned'),
            ('02_duplicates_cleaned.csv', 'Duplicates Removed'),
            ('03_statistical_outliers_cleaned.csv', 'Statistical Outliers'),
            ('04_data_errors_cleaned.csv', 'Data Errors Fixed'),
            ('05_infrequent_values_cleaned.csv', 'Infrequent Values'),
            ('06_context_outliers_cleaned.csv', 'Context Outliers'),
            ('07_dtypes_cleaned.pkl', 'Data Types Fixed'),
            ('08_target_leakage_cleaned.pkl', 'Target Leakage Removed'),
            ('09_multicollinearity_cleaned.pkl', 'Multicollinearity Fixed')
        ]
        
        self.raw_data_path = '/home/antonios/Desktop/Practica_de_vara/data-science-internship/data/raw/hotel_booking_cancellation_prediction.csv'
    
    def load_data(self, step):
        """Load data for a specific cleaning step"""
        filename, _ = step
        
        if filename == 'raw':
            return pd.read_csv(self.raw_data_path)
        elif filename.endswith('.pkl'):
            return pd.read_pickle(os.path.join(self.output_dir, filename))
        else:
            return pd.read_csv(os.path.join(self.output_dir, filename))
    
    def plot_data_shape_progression(self):
        """Plot how data shape changes through cleaning steps"""
        shapes = []
        step_names = []
        
        for filename, step_name in self.cleaning_steps:
            try:
                data = self.load_data((filename, step_name))
                shapes.append((data.shape[0], data.shape[1]))
                step_names.append(step_name)
                print(f"Loaded {step_name}: {data.shape}")
            except FileNotFoundError:
                print(f"File not found for {step_name}, skipping...")
                continue
            except Exception as e:
                print(f"Error loading {step_name}: {e}")
                continue
        
        if not shapes:
            print("No data files found!")
            return
        
        rows = [s[0] for s in shapes]
        cols = [s[1] for s in shapes]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot rows progression
        ax1.plot(range(len(step_names)), rows, marker='o', linewidth=3, markersize=8, color='#e74c3c')
        ax1.set_title('Number of Rows Through Cleaning Pipeline', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Number of Rows', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(len(step_names)))
        ax1.set_xticklabels(step_names, rotation=45, ha='right')
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(range(len(step_names)), rows)):
            ax1.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot columns progression
        ax2.plot(range(len(step_names)), cols, marker='s', linewidth=3, markersize=8, color='#3498db')
        ax2.set_title('Number of Columns Through Cleaning Pipeline', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Number of Columns', fontsize=12)
        ax2.set_xlabel('Cleaning Steps', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(len(step_names)))
        ax2.set_xticklabels(step_names, rotation=45, ha='right')
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(range(len(step_names)), cols)):
            ax2.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'data_shape_progression.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_missing_values_heatmap(self):
        """Plot missing values heatmap for raw data"""
        try:
            raw_data = self.load_data(('raw', 'Raw Data'))
            
            # Calculate missing values percentage
            missing_pct = (raw_data.isnull().sum() / len(raw_data)) * 100
            missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
            
            if len(missing_pct) == 0:
                print("No missing values found in raw data")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Missing values bar chart
            missing_pct.plot(kind='bar', ax=ax1, color='#e74c3c')
            ax1.set_title('Missing Values Percentage by Column', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Missing Percentage (%)', fontsize=12)
            ax1.set_xlabel('Columns', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(missing_pct.values):
                ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
            
            # Missing values heatmap (sample of data)
            sample_data = raw_data.sample(min(1000, len(raw_data)))
            missing_matrix = sample_data.isnull()
            
            sns.heatmap(missing_matrix, yticklabels=False, cbar=True, cmap='RdYlBu_r', ax=ax2)
            ax2.set_title('Missing Values Pattern (Sample)', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Columns', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'missing_values_analysis.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating missing values plot: {e}")
    
    def plot_outliers_before_after(self):
        """Plot outliers before and after statistical outlier cleaning for columns that actually had changes"""
        try:
            before_data = self.load_data(('02_duplicates_cleaned.csv', 'Before Outliers'))
            after_data = self.load_data(('03_statistical_outliers_cleaned.csv', 'After Outliers'))
            
            numeric_cols = before_data.select_dtypes(include=[np.number]).columns
            
            # Find columns that actually had outliers removed
            columns_with_changes = []
            for col in numeric_cols:
                if col in after_data.columns:
                    # Calculate outliers before and after
                    before_q75 = before_data[col].quantile(0.75)
                    before_q25 = before_data[col].quantile(0.25)
                    before_iqr = before_q75 - before_q25
                    before_outliers = len(before_data[(before_data[col] < before_q25 - 1.5*before_iqr) | 
                                                     (before_data[col] > before_q75 + 1.5*before_iqr)])
                    
                    after_q75 = after_data[col].quantile(0.75)
                    after_q25 = after_data[col].quantile(0.25)
                    after_iqr = after_q75 - after_q25
                    after_outliers = len(after_data[(after_data[col] < after_q25 - 1.5*after_iqr) | 
                                                   (after_data[col] > after_q75 + 1.5*after_iqr)])
                    
                    # Check if there was actually a change
                    if before_outliers != after_outliers or len(before_data[col].dropna()) != len(after_data[col].dropna()):
                        columns_with_changes.append((col, before_outliers, after_outliers, 
                                                   len(before_data[col].dropna()), len(after_data[col].dropna())))
            
            # Sort by most outliers removed and take top 6 (or all if less than 6)
            columns_with_changes.sort(key=lambda x: x[1] - x[2], reverse=True)
            selected_cols = columns_with_changes[:6]
            
            if not selected_cols:
                print("No columns with outlier changes found")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (col, before_outliers, after_outliers, before_count, after_count) in enumerate(selected_cols):
                if i >= 6:
                    break
                    
                # Before outlier removal
                axes[i].boxplot([before_data[col].dropna(), after_data[col].dropna()], 
                               labels=['Before', 'After'], patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='red', linewidth=2))
                
                axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                
                # Add comprehensive statistics text
                stats_text = f'Before: {before_outliers} outliers \n'
                stats_text += f'After: {after_outliers} outliers \n'
                
                axes[i].text(0.02, 0.98, stats_text, 
                            transform=axes[i].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                            fontsize=9)
            
            # Hide unused subplots
            for j in range(len(selected_cols), 6):
                axes[j].set_visible(False)
            
            plt.suptitle('Statistical Outliers: Before vs After Cleaning ', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'outliers_before_after.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print summary of changes
            print("Outlier cleaning summary:")
            for col, before_outliers, after_outliers, before_count, after_count in selected_cols:
                print(f"{col}: {before_count - after_count} values removed "
                      f"({before_outliers} -> {after_outliers} outliers)")
            
        except Exception as e:
            print(f"Error creating outliers plot: {e}")
    
    def plot_data_types_distribution(self):
        """Plot data types distribution before and after dtype cleaning"""
        try:
            before_data = self.load_data(('06_context_outliers_cleaned.csv', 'Before Dtypes'))
            after_data = self.load_data(('07_dtypes_cleaned.pkl', 'After Dtypes'))
            
            # Count data types
            before_types = before_data.dtypes.astype(str).value_counts()
            after_types = after_data.dtypes.astype(str).value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Before
            wedges1, texts1, autotexts1 = ax1.pie(before_types.values, labels=before_types.index, 
                                                  autopct='%1.1f%%', startangle=90)
            ax1.set_title('Data Types Distribution - Before Cleaning', fontsize=14, fontweight='bold')
            
            # After
            wedges2, texts2, autotexts2 = ax2.pie(after_types.values, labels=after_types.index, 
                                                  autopct='%1.1f%%', startangle=90)
            ax2.set_title('Data Types Distribution - After Cleaning', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'data_types_distribution.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating data types plot: {e}")
    
    def plot_target_correlation_analysis(self):
        """Plot correlation analysis for the actual remaining features after target leakage removal"""
        try:
            data = self.load_data(('08_target_leakage_cleaned.pkl', 'After Target Leakage'))
            
            if 'is_canceled' not in data.columns:
                print("Target variable 'is_canceled' not found")
                return
            
            print(f"Available columns: {list(data.columns)}")
            print(f"Data shape: {data.shape}")
            
            # Identify different column types
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            

            
            # Remove target from numeric columns for analysis
            if 'is_canceled' in numeric_cols:
                numeric_cols.remove('is_canceled')
            
            print(f"Numeric columns: {numeric_cols}")
            print(f"Categorical columns: {categorical_cols}")
            
            # Create figure with 2 subplots for numeric and categorical features
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # 1. Numeric Features Correlation
            if numeric_cols:
                numeric_data = data[numeric_cols + ['is_canceled']]
                target_corr = numeric_data.corr()['is_canceled'].drop('is_canceled')
                
                # Plot numeric correlations
                colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in target_corr.values]
                bars = ax1.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
                ax1.set_yticks(range(len(target_corr)))
                ax1.set_yticklabels(target_corr.index, fontsize=12)
                ax1.set_xlabel('Correlation with Target', fontsize=12)
                ax1.set_title('Numeric Features Correlation with Target', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')
                ax1.axvline(x=0, color='black', linewidth=1)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, target_corr.values)):
                    ax1.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                            ha='left' if value >= 0 else 'right', va='center', fontsize=10, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No numeric features\nfound for correlation', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Numeric Features Correlation', fontsize=14, fontweight='bold')
            
            # 2. Categorical Features Analysis
            if categorical_cols:
                # Calculate association with target for categorical variables (using Chi-square or Cramér's V)
                from scipy.stats import chi2_contingency
                
                categorical_associations = {}
                for col in categorical_cols:
                    if col != 'is_canceled':
                        try:
                            # Create contingency table
                            contingency = pd.crosstab(data[col], data['is_canceled'])
                            chi2, p_value, dof, expected = chi2_contingency(contingency)
                            
                            # Calculate Cramér's V (association strength)
                            n = contingency.sum().sum()
                            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                            categorical_associations[col] = cramers_v
                        except:
                            categorical_associations[col] = 0
                
                if categorical_associations:
                    # Sort by association strength
                    sorted_assoc = dict(sorted(categorical_associations.items(), 
                                            key=lambda x: x[1], reverse=True))
                    
                    # Plot categorical associations
                    colors_cat = plt.cm.viridis(np.linspace(0, 1, len(sorted_assoc)))
                    bars_cat = ax2.barh(range(len(sorted_assoc)), list(sorted_assoc.values()), 
                                    color=colors_cat, alpha=0.7)
                    ax2.set_yticks(range(len(sorted_assoc)))
                    ax2.set_yticklabels(list(sorted_assoc.keys()), fontsize=12)
                    ax2.set_xlabel("Association with Target", fontsize=12)
                    ax2.set_title('Categorical Features Association with Target', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, value) in enumerate(zip(bars_cat, sorted_assoc.values())):
                        ax2.text(value + 0.01, i, f'{value:.3f}', 
                                ha='left', va='center', fontsize=10, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No categorical features\nfound for analysis', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Categorical Features Association', fontsize=14, fontweight='bold')

            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'target_correlation_analysis.png'), 
                    dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print detailed summary
            print("\n=== FEATURE ANALYSIS SUMMARY ===")
            print(f"Dataset shape: {data.shape}")
            print("Target variable: is_canceled")
            print(f"Cancellation rate: {data['is_canceled'].mean()*100:.1f}%")
            
            if numeric_cols:
                print("\nNumeric Features Correlation:")
                for feature, corr in target_corr.sort_values(key=abs, ascending=False).items():
                    print(f"  {feature}: {corr:.3f} ({abs(corr)*100:.1f}% strength)")
            
            if categorical_cols and categorical_associations:
                print("\nCategorical Features Association (Cramér's V):")
                for feature, assoc in sorted(categorical_associations.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {feature}: {assoc:.3f} ({assoc*100:.1f}% strength)")
            
        except Exception as e:
            print(f"Error creating target correlation analysis: {e}")
            import traceback
            traceback.print_exc()

    def plot_multicollinearity_analysis(self):
        """Plot correlation heatmap before and after multicollinearity removal"""
        try:
            before_data = self.load_data(('08_target_leakage_cleaned.pkl', 'Before Multicollinearity'))
            after_data = self.load_data(('09_multicollinearity_cleaned.pkl', 'After Multicollinearity'))
            
            # Remove target for correlation analysis
            if 'is_canceled' in before_data.columns:
                before_numeric = before_data.drop('is_canceled', axis=1).select_dtypes(include=[np.number])
            else:
                before_numeric = before_data.select_dtypes(include=[np.number])
                
            if 'is_canceled' in after_data.columns:
                after_numeric = after_data.drop('is_canceled', axis=1).select_dtypes(include=[np.number])
            else:
                after_numeric = after_data.select_dtypes(include=[np.number])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Before multicollinearity removal
            before_corr = before_numeric.corr()
            mask1 = np.triu(np.ones_like(before_corr, dtype=bool))
            sns.heatmap(before_corr, mask=mask1, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, fmt='.2f', ax=ax1, cbar_kws={"shrink": 0.8})
            ax1.set_title('Feature Correlation - Before Multicollinearity Removal', fontsize=14, fontweight='bold')
            
            # After multicollinearity removal
            after_corr = after_numeric.corr()
            mask2 = np.triu(np.ones_like(after_corr, dtype=bool))
            sns.heatmap(after_corr, mask=mask2, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, fmt='.2f', ax=ax2, cbar_kws={"shrink": 0.8})
            ax2.set_title('Feature Correlation - After Multicollinearity Removal', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'multicollinearity_analysis.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating multicollinearity plot: {e}")
    
    def plot_cleaning_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        try:
            # Load data
            raw_data = self.load_data(('raw', 'Raw Data'))
            final_data = self.load_data(('09_multicollinearity_cleaned.pkl', 'Final Data'))
            
            # 1. Data shape comparison
            ax1 = fig.add_subplot(gs[0, 0])
            categories = ['Rows', 'Columns']
            raw_values = [raw_data.shape[0], raw_data.shape[1]]
            final_values = [final_data.shape[0], final_data.shape[1]]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, raw_values, width, label='Raw Data', color='#e74c3c')
            bars2 = ax1.bar(x + width/2, final_values, width, label='Final Data', color='#2ecc71')
            
            ax1.set_xlabel('Data Dimensions')
            ax1.set_ylabel('Count')
            ax1.set_title('Data Shape: Raw vs Final')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            # 2. Missing values summary
            ax2 = fig.add_subplot(gs[0, 1])
            raw_missing = raw_data.isnull().sum().sum()
            final_missing = final_data.isnull().sum().sum()
            
            ax2.bar(['Raw Data', 'Final Data'], [raw_missing, final_missing], 
                   color=['#e74c3c', '#2ecc71'])
            ax2.set_title('Total Missing Values')
            ax2.set_ylabel('Count')
            ax2.grid(True, alpha=0.3)
            
            for i, v in enumerate([raw_missing, final_missing]):
                ax2.text(i, v + max(raw_missing, final_missing) * 0.01, str(v), 
                        ha='center', va='bottom')
            
            # 3. Data types distribution
            ax3 = fig.add_subplot(gs[0, 2])
            final_types = final_data.dtypes.astype(str).value_counts()
            ax3.pie(final_types.values, labels=final_types.index, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Final Data Types Distribution')
            
            # 4. Numeric columns distributions (sample)
            numeric_cols = final_data.select_dtypes(include=[np.number]).columns[:6]
            
            for i, col in enumerate(numeric_cols):
                ax4_sub = plt.subplot(2, 3, 4 + i)
                final_data[col].hist(bins=30, alpha=0.7, color=f'C{i}')
                ax4_sub.set_title(f'{col}')
                ax4_sub.grid(True, alpha=0.3)
            
            # 5. Cleaning impact summary
            ax5 = fig.add_subplot(gs[2, :])
            
            # Calculate cleaning impact metrics
            metrics = {
                'Rows Removed': raw_data.shape[0] - final_data.shape[0],
                'Columns Removed': raw_data.shape[1] - final_data.shape[1],
                'Missing Values Fixed': raw_missing - final_missing,
                'Data Reduction (%)': ((raw_data.shape[0] - final_data.shape[0]) / raw_data.shape[0]) * 100
            }
            
            # Create summary table
            table_data = []
            for metric, value in metrics.items():
                if metric == 'Data Reduction (%)':
                    table_data.append([metric, f"{value:.2f}%"])
                else:
                    table_data.append([metric, f"{value:,}"])
            
            ax5.axis('tight')
            ax5.axis('off')
            table = ax5.table(cellText=table_data,
                             colLabels=['Cleaning Impact', 'Value'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(2):
                    if i == 0:  # Header
                        table[(i, j)].set_facecolor('#3498db')
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    else:
                        table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            
            plt.suptitle('Data Cleaning Pipeline - Summary Dashboard', fontsize=20, fontweight='bold')
            plt.savefig(os.path.join(self.plots_dir, 'cleaning_summary_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating summary dashboard: {e}")
    
    def generate_all_plots(self):
        """Generate all cleaning visualization plots"""
        print("Generating data cleaning visualizations...")
        print(f"Plots will be saved to: {self.plots_dir}")
        
        print("\n1. Creating data shape progression plot...")
        self.plot_data_shape_progression()
        
        print("\n2. Creating missing values analysis...")
        self.plot_missing_values_heatmap()
        
        print("\n3. Creating outliers before/after comparison...")
        self.plot_outliers_before_after()
        
        print("\n4. Creating data types distribution plot...")
        self.plot_data_types_distribution()
        
        print("\n5. Creating target correlation analysis...")
        self.plot_target_correlation_analysis()
        
        print("\n6. Creating multicollinearity analysis...")
        self.plot_multicollinearity_analysis()
        
        print("\n7. Creating summary dashboard...")
        self.plot_cleaning_summary_dashboard()
        
        print("\n All plots generated successfully!")
        print(f"📁 Check the plots directory: {self.plots_dir}")

# Usage example
if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = DataCleaningVisualizer()
    
    # Generate all plots
    visualizer.generate_all_plots()