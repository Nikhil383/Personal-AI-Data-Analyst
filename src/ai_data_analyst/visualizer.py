"""AI Data Analyst - Visualizer Module"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import io
import base64


class DataVisualizer:
    """Handles data visualization with matplotlib and seaborn."""

    # Set dark theme
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid")

    def __init__(self, df: pd.DataFrame, output_dir: Path):
        """Initialize with dataframe and output directory."""
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Custom color palette
        self.colors = ['#00D4AA', '#2D5A87', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    def _setup_figure(self, figsize: Tuple[int, int] = (10, 6)):
        """Setup figure with dark theme."""
        fig, ax = plt.subplots(figsize=figsize, facecolor='#161B22')
        ax.set_facecolor('#161B22')
        return fig, ax

    def create_histogram(self, column: str, bins: int = 30, title: Optional[str] = None) -> str:
        """Create histogram for a numeric column."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        fig, ax = self._setup_figure()
        ax.hist(self.df[column].dropna(), bins=bins, color=self.colors[0], edgecolor='#0D1117', alpha=0.8)
        ax.set_xlabel(column, color='#E6EDF3', fontsize=12)
        ax.set_ylabel('Frequency', color='#E6EDF3', fontsize=12)
        ax.set_title(title or f'Distribution of {column}', color='#E6EDF3', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#8B949E')

        # Add grid
        ax.grid(True, alpha=0.2, color='#8B949E')

        plt.tight_layout()
        return self._save_figure(fig, f'histogram_{column}')

    def create_bar_chart(self, x_column: str, y_column: Optional[str] = None,
                        title: Optional[str] = None, top_n: int = 10) -> str:
        """Create bar chart for categorical data."""
        if x_column not in self.df.columns:
            raise ValueError(f"Column '{x_column}' not found")

        fig, ax = self._setup_figure()

        if y_column and y_column in self.df.columns:
            data = self.df.groupby(x_column)[y_column].sum().sort_values(ascending=False).head(top_n)
        else:
            data = self.df[x_column].value_counts().head(top_n)

        bars = ax.bar(range(len(data)), data.values, color=self.colors[0], edgecolor='#0D1117', alpha=0.8)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right', color='#E6EDF3')
        ax.set_xlabel(x_column, color='#E6EDF3', fontsize=12)
        ax.set_ylabel(y_column or 'Count', color='#E6EDF3', fontsize=12)
        ax.set_title(title or f'{y_column or "Count"} by {x_column}', color='#E6EDF3', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#8B949E')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', color='#E6EDF3', fontsize=9)

        plt.tight_layout()
        return self._save_figure(fig, f'bar_{x_column}')

    def create_scatter(self, x_column: str, y_column: str,
                      title: Optional[str] = None, size_column: Optional[str] = None) -> str:
        """Create scatter plot."""
        if x_column not in self.df.columns or y_column not in self.df.columns:
            raise ValueError(f"Columns not found")

        fig, ax = self._setup_figure()

        sizes = 50 if not size_column or size_column not in self.df.columns else self.df[size_column] * 2
        ax.scatter(self.df[x_column], self.df[y_column], s=sizes,
                  c=self.colors[0], alpha=0.6, edgecolors='#0D1117')

        ax.set_xlabel(x_column, color='#E6EDF3', fontsize=12)
        ax.set_ylabel(y_column, color='#E6EDF3', fontsize=12)
        ax.set_title(title or f'{y_column} vs {x_column}', color='#E6EDF3', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#8B949E')
        ax.grid(True, alpha=0.2, color='#8B949E')

        plt.tight_layout()
        return self._save_figure(fig, f'scatter_{x_column}_{y_column}')

    def create_line_chart(self, x_column: str, y_columns: list,
                         title: Optional[str] = None) -> str:
        """Create line chart."""
        if x_column not in self.df.columns:
            raise ValueError(f"Column '{x_column}' not found")

        fig, ax = self._setup_figure()

        for i, y_col in enumerate(y_columns):
            if y_col in self.df.columns:
                color = self.colors[i % len(self.colors)]
                ax.plot(self.df[x_column], self.df[y_col], marker='o',
                       label=y_col, color=color, linewidth=2, markersize=4)

        ax.set_xlabel(x_column, color='#E6EDF3', fontsize=12)
        ax.set_ylabel('Value', color='#E6EDF3', fontsize=12)
        ax.set_title(title or f'Line Chart: {", ".join(y_columns)}',
                    color='#E6EDF3', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#8B949E')
        ax.legend(facecolor='#161B22', edgecolor='#8B949E', labelcolor='#E6EDF3')
        ax.grid(True, alpha=0.2, color='#8B949E')

        plt.tight_layout()
        return self._save_figure(fig, f'line_{x_column}')

    def create_box_plot(self, columns: list, title: Optional[str] = None) -> str:
        """Create box plot."""
        valid_cols = [c for c in columns if c in self.df.columns]
        if not valid_cols:
            raise ValueError("No valid columns provided")

        fig, ax = self._setup_figure()

        data_to_plot = [self.df[col].dropna() for col in valid_cols]
        bp = ax.boxplot(data_to_plot, labels=valid_cols, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor(self.colors[0])
            patch.set_alpha(0.7)

        ax.set_xlabel('Columns', color='#E6EDF3', fontsize=12)
        ax.set_ylabel('Value', color='#E6EDF3', fontsize=12)
        ax.set_title(title or f'Box Plot: {", ".join(valid_cols)}',
                    color='#E6EDF3', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#8B949E')
        ax.grid(True, alpha=0.2, color='#8B949E', axis='y')

        plt.tight_layout()
        return self._save_figure(fig, f'box_{"_".join(valid_cols)}')

    def create_correlation_heatmap(self, title: Optional[str] = None) -> str:
        """Create correlation heatmap for numeric columns."""
        numeric_df = self.df.select_dtypes(include=['number'])
        if numeric_df.empty:
            raise ValueError("No numeric columns found")

        fig, ax = self._setup_figure(figsize=(12, 10))

        corr = numeric_df.corr()
        mask = None

        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                   ax=ax)

        ax.set_title(title or 'Correlation Heatmap',
                    color='#E6EDF3', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return self._save_figure(fig, 'correlation_heatmap')

    def create_pie_chart(self, column: str, top_n: int = 5, title: Optional[str] = None) -> str:
        """Create pie chart for categorical data."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        fig, ax = self._setup_figure()

        data = self.df[column].value_counts().head(top_n)
        colors = self.colors[:len(data)]

        wedges, texts, autotexts = ax.pie(data, labels=data.index, autopct='%1.1f%%',
                                          colors=colors, explode=[0.02]*len(data),
                                          shadow=False, startangle=90)

        for text in texts:
            text.set_color('#E6EDF3')
        for autotext in autotexts:
            autotext.set_color('#0D1117')
            autotext.set_fontweight('bold')

        ax.set_title(title or f'Distribution of {column}',
                    color='#E6EDF3', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return self._save_figure(fig, f'pie_{column}')

    def _save_figure(self, fig, filename: str) -> str:
        """Save figure and return base64 encoded string."""
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                   facecolor='#161B22', edgecolor='none')
        plt.close(fig)

        # Return base64 for display
        with open(filepath, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()

        return f"data:image/png;base64,{img_data}"

    def get_chart_base64(self, fig) -> str:
        """Convert figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='#161B22', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"
