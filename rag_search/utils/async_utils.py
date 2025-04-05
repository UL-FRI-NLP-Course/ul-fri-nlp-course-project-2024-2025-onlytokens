import asyncio
from typing import Any, Awaitable, List, TypeVar

T = TypeVar('T')

async def gather_with_concurrency(n: int, *tasks: Awaitable[T]) -> List[T]:
    """
    Run tasks with a concurrency limit.
    
    Args:
        n: Maximum number of concurrent tasks
        tasks: Async tasks to run
        
    Returns:
        List of results from the tasks
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks))

class RateLimiter:
    """Rate limiter for async requests."""
    
    def __init__(self, calls_per_second: float = 2.0):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire the rate limiter (wait if necessary)."""
        async with self.lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                wait_time = self.min_interval - time_since_last_call
                await asyncio.sleep(wait_time)
                
            self.last_call_time = asyncio.get_event_loop().time()

    async def __aenter__(self):
        await self.acquire()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
