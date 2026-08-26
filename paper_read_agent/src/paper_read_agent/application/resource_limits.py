"""Central resource and input limits."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Iterator,Sequence

@dataclass(frozen=True,slots=True)
class InputLimits:
    max_file_bytes:int=100*1024*1024;max_pages:int=1000;max_upload_batch:int=20
    max_question_chars:int=8000;max_concurrent_tasks:int=2;max_output_tokens:int=8192
    def __post_init__(self):
        if min(self.max_file_bytes,self.max_pages,self.max_upload_batch,self.max_question_chars,
               self.max_concurrent_tasks,self.max_output_tokens)<=0:raise ValueError("Resource limits must be positive")

class ResourceGuard:
    def __init__(self,limits:InputLimits):self.limits=limits;self._slots=BoundedSemaphore(limits.max_concurrent_tasks)
    def validate_question(self,question:str)->str:
        value=question.strip()
        if not value:raise ValueError("Question must not be empty")
        if len(value)>self.limits.max_question_chars:raise ValueError("Question exceeds configured length limit")
        return value
    def validate_upload_batch(self,items:Sequence[object])->None:
        if not items:raise ValueError("Upload batch must not be empty")
        if len(items)>self.limits.max_upload_batch:raise ValueError("Upload batch exceeds configured limit")
    @contextmanager
    def task_slot(self)->Iterator[None]:
        if not self._slots.acquire(blocking=False):raise RuntimeError("Concurrent task limit reached")
        try:yield
        finally:self._slots.release()
