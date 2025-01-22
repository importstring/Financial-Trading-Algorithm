from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
import time

class LoadingBar:
    def __init__(self):
        self.console = Console()
        self.progress = None
        self.overall_task = None
        self.task_progress = None
        self.progress_table = Table.grid(expand=True)
        self.progress_table.add_column(justify="right", width=20)
        self.progress_table.add_column(width=60)
        self.tasks = {}
        self.expected_steps = []

    def initialize(self, total_tasks):
        self.update('initialize', 'Starting initialization')
        self.progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold blue]{task.fields[status]}"),
            console=self.console,
            transient=True,
        )
        self.overall_task = self.progress.add_task("[yellow]Overall Progress", total=total_tasks, status="")
        self.task_progress = self.progress.add_task("Task Progress", total=100, status="")
        self.progress.start()
        self.update('initialize', 'Initialization complete', 1, 'completed')

    def update(self, task_name, subtask_name=None, advance=0, status=None):
        if task_name not in self.tasks:
            self.tasks[task_name] = {"subtasks": [], "completed": False}
            self.progress.update(self.overall_task, advance=1, status=f"Executing: {task_name}")

        if subtask_name:
            self.tasks[task_name]["subtasks"].append(subtask_name)
            subtask_count = len(self.tasks[task_name]["subtasks"])
            completion = (subtask_count * 100) // (subtask_count + 1)
            self.progress.update(self.task_progress, completed=completion, status=f"[dim]{subtask_name}[/dim]")
        else:
            self.tasks[task_name]["completed"] = True
            self.progress.update(self.task_progress, completed=100, status=f"[bold green]{task_name} Completed[/bold green]")
            self.progress_table.add_row(f"[cyan]{task_name}:", "[green]✓ Completed[/green]")

        if status:
            self.progress.update(self.task_progress, status=status)

        self.console.clear()
        self.console.print(Panel.fit("[bold cyan]Financial Agent Task Execution[/bold cyan]", border_style="blue"))
        self.progress.refresh()
        self.console.print(self.progress_table)
        self.show_expected_steps()

    def stop(self):
        self.update('stop', 'Stopping loading process')
        if self.progress:
            self.progress.stop()
        self.console.print(Panel.fit("[bold green]All financial tasks completed successfully.[/bold green]", border_style="green"))
        self.update('stop', 'Stopped loading process', 1, 'completed')

    def set_expected_steps(self, steps):
        self.update('set_expected_steps', 'Setting expected steps')
        self.expected_steps = steps
        self.update('set_expected_steps', 'Expected steps set', 1, 'completed')

    def show_expected_steps(self):
        self.update('show_expected_steps', 'Displaying expected steps')
        if self.expected_steps:
            self.console.print("\n[bold]Expected Next Steps:[/bold]")
            for step in self.expected_steps:
                self.console.print(f"- {step}")
        self.update('show_expected_steps', 'Displayed expected steps', 1, 'completed')

    def remove_completed_step(self, step):
        self.update('remove_completed_step', 'Removing completed step')
        if step in self.expected_steps:
            self.expected_steps.remove(step)
        self.update('remove_completed_step', 'Completed step removed', 1, 'completed')

    def run_agent_tasks(self):
        self.update('run_agent_tasks', 'Starting agent tasks')
        loading_bar = LoadingBar()
        loading_bar.initialize(total_tasks=5)

        phases = [
            ("Phase 0: Planning", ["Query Ollama", "Get stock data"]),
            ("Phase 1: Research", ["Query Ollama", "Query Perplexity", "Query ChatGPT", "Get stock data"]),
            ("Phase 2: Save Information", ["Saving to class variable"]),
            ("Phase 3: Narrow Trades", ["Research stocks", "Get insights", "Reason and iterate", "Get stock data"]),
            ("Phase 4: Execute", ["Logging trades"]),
            ("Phase 5: Learn & Justify", ["Learning from trades", "Justifying actions", "Saving information", "Saving improvement notes"])
        ]

        loading_bar.set_expected_steps([phase[0] for phase in phases])

        for phase, subtasks in phases:
            loading_bar.update(phase)
            for subtask in subtasks:
                loading_bar.update(phase, subtask)
                time.sleep(0.5)
            loading_bar.update(phase)
            loading_bar.remove_completed_step(phase)

        loading_bar.stop()
        self.update('run_agent_tasks', 'Agent tasks completed', 1, 'completed')

    def create_new_tasks(self):
        self.update('create_new_tasks', 'Starting new tasks')
        loading_bar = LoadingBar()
        loading_bar.initialize(total_tasks=3)

        tasks = [
            ("Creating New Tasks", ["Analyzing market trends", "Identifying potential trades"]),
            ("Updating Agent State", ["Refreshing market data", "Adjusting trading parameters"]),
            ("Preparing Next Cycle", ["Setting up monitoring", "Scheduling next run"])
        ]

        loading_bar.set_expected_steps([task[0] for task in tasks])

        for task, subtasks in tasks:
            for subtask in subtasks:
                loading_bar.update(task, subtask)
                time.sleep(1)
            loading_bar.update(task)
            loading_bar.remove_completed_step(task)

        loading_bar.stop()
        self.update('create_new_tasks', 'Tasks creation completed', 1, 'completed')

    def update_decision_progress(self, decision):
        self.update('update_decision_progress', 'Starting decision progress updates')
        loading_bar = LoadingBar()
        loading_bar.initialize(total_tasks=1)

        subtasks = [f"Evaluating: {decision}", "Analyzing potential outcomes", "Finalizing decision"]
        loading_bar.set_expected_steps(subtasks)

        for subtask in subtasks:
            loading_bar.update("Decision Making", subtask)
            time.sleep(1)
            loading_bar.remove_completed_step(subtask)
        loading_bar.update("Decision Making")

        loading_bar.stop()
        self.update('update_decision_progress', 'Decision progress updated', 1, 'completed')
