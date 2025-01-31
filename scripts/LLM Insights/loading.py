from rich.progress import Progress, SpinnerColumn, TextColumn, TaskID
from typing import Dict, Optional
import hashlib
import time

class DynamicLoadingBar:
    """Self-managing progress system for unknown tasks"""
    
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            expand=True
        )
        self.task_map: Dict[str, TaskID] = {}
        self.task_counter = 0
        self._live = None

    def _generate_task_id(self, prefix: str) -> str:
        """Create unique task ID using function context"""
        return f"{prefix}_{self.task_counter}_{time.monotonic_ns()}"

    def start(self):
        """Begin live display context"""
        self._live = self.progress.__enter__()

    def track(self, func):
        """Decorator to auto-track any function call"""
        def wrapper(*args, **kwargs):
            task_id = self._generate_task_id(func.__name__)
            desc = f"Processing {func.__name__}"
            total = kwargs.pop('total', None)  # Allow dynamic totals
            
            task = self.progress.add_task(desc, total=total)
            self.task_map[task_id] = task
            
            try:
                result = func(*args, **kwargs)
                self.progress.update(task, description=f"✅ {desc}", completed=100)
                return result
            except Exception as e:
                self.progress.update(task, description=f"❌ {desc} - {str(e)}")
                raise
            finally:
                self.task_counter += 1
        return wrapper

    def dynamic_update(self, description: str, operation: str, advance: int = 1):
        """Universal update method for unknown tasks"""
        task_id = hashlib.md5(operation.encode()).hexdigest()
        
        if task_id not in self.task_map:
            self.task_map[task_id] = self.progress.add_task(description)
            
        self.progress.update(self.task_map[task_id], advance=advance)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.progress.__exit__(*args)
