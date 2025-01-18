import logging
import random
import time
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

# Configure Rich logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)

log = logging.getLogger("rich")
console = Console()

class Animation:
    def __init__(self):
        self.tasks = {
            "Risk Analysis": ["Gathering historical data", "Running Monte Carlo simulations", "Analyzing risk factors"],
            "Market Data Acquisition": ["Connecting to financial APIs", "Fetching real-time quotes", "Storing data in database"],
            "AI Model Training": ["Preprocessing financial data", "Training predictive model", "Validating model accuracy"],
            "Portfolio Evaluation": ["Analyzing asset allocation", "Calculating returns and volatility", "Assessing risk-adjusted performance"],
            "Report Generation": ["Compiling key metrics", "Creating data visualizations", "Formatting final document"]
        }

    def spinner_animation(self):
        console.print(Panel.fit("[bold cyan]Financial Agent Task Execution[/bold cyan]", border_style="blue"))
        
        progress_table = Table.grid(expand=True)
        progress_table.add_column(justify="right", width=20)
        progress_table.add_column(width=60)
        
        with Progress(
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold blue]{task.fields[status]}"),
            console=console,
            transient=True,
        ) as progress:
            overall_task = progress.add_task("[yellow]Overall Progress", total=len(self.tasks), status="")
            task_progress = progress.add_task("Task Progress", total=100, status="")

            for task, subtasks in self.tasks.items():
                progress.update(overall_task, advance=1, status=f"Executing: {task}")
                progress.update(task_progress, completed=0, status=f"[bold yellow]{task}[/bold yellow]")
                
                for i, subtask in enumerate(subtasks):
                    time.sleep(random.uniform(0.5, 1.0))
                    progress.update(task_progress, completed=(i+1)*100/len(subtasks), status=f"[dim]{subtask}[/dim]")
                
                time.sleep(random.uniform(1.0, 2.0))
                progress.update(task_progress, completed=100, status=f"[bold green]{task} Completed[/bold green]")
                
                # Add task summary to table
                progress_table.add_row(f"[cyan]{task}:", "[green]✓ Completed[/green]")
                console.print(progress_table)
        
        console.print(Panel.fit("[bold green]All financial tasks completed successfully.[/bold green]", border_style="green"))

if __name__ == "__main__":
    animation = Animation()
    animation.spinner_animation()


"""
/usr/local/bin/python3 "/Users/simon/import logging.py"
simon@Simons-Tps-Computer ~ % /usr/local/bin/python3 "/Users/simon/import logging.py"
╭────────────────────────────────╮
│ Financial Agent Task Execution │
╰────────────────────────────────╯
                          Risk Analysis:✓ Completed                                                                                                             
                          Risk Analysis:✓ Completed                                                                                                             
                Market Data Acquisition:✓ Completed                                                                                                             
                          Risk Analysis:✓ Completed                                                                                                             
                Market Data Acquisition:✓ Completed                                                                                                             
                      AI Model Training:✓ Completed                                                                                                             
                          Risk Analysis:✓ Completed                                                                                                             
                Market Data Acquisition:✓ Completed                                                                                                             
                      AI Model Training:✓ Completed                                                                                                             
                   Portfolio Evaluation:✓ Completed                                                                                                             
                          Risk Analysis:✓ Completed                                                                                                             
                Market Data Acquisition:✓ Completed                                                                                                             
                      AI Model Training:✓ Completed                                                                                                             
                   Portfolio Evaluation:✓ Completed                                                                                                             
                      Report Generation:✓ Completed                                                                                                             
╭─────────────────────────────────────────────╮
│ All financial tasks completed successfully. │
╰─────────────────────────────────────────────╯
simon@Simons-Tps-Computer ~ % 

"""
