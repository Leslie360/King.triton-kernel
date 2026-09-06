"""API request/response models for KernelGym server."""

from typing import Any

from pydantic import BaseModel, Field, root_validator, validator

from kernelgym.common import Backend, ErrorCode, Priority, TaskStatus


class EvaluationRequest(BaseModel):
    """Request model for kernel evaluation."""

    task_id: str = Field(..., description="Unique task identifier")
    reference_code: str | None = Field(default=None, description="PyTorch reference implementation")
    kernel_code: str = Field(..., description="Custom kernel implementation")
    toolkit: str = Field(default="kernelbench", description="Toolkit adapter name")
    backend_adapter: str = Field(default="kernelbench", description="Backend adapter name")
    backend: Backend = Field(default=Backend.TRITON, description="Backend type")
    num_correct_trials: int = Field(default=5, ge=1, le=20, description="Number of correctness trials")
    num_perf_trials: int = Field(default=100, ge=1, le=1000, description="Number of performance trials")
    num_warmup: int = Field(default=3, ge=0, le=100, description="Number of warmup iterations")
    timeout: int = Field(default=300, ge=10, le=3600, description="Task timeout in seconds")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority")
    device_preference: str | None = Field(default=None, description="Preferred GPU device")
    force_refresh: bool = Field(default=False, description="Force refresh, skip cached results")
    entry_point: str = Field(default="Model", description="Entry point class name for model evaluation")
    reference_backend: str | None = Field(
        default=None,
        description="Reference backend for timing (e.g., pytorch, torch_compile)",
    )
    uuid: str | None = Field(default=None, description="UUID for reference timing cache lookup")
    use_reference_cache: bool = Field(default=False, description="Use cached reference timing if available")
    is_valid: bool = Field(
        default=False,
        description="If true, use validation data cache (val_data_cache) instead of regular cache",
    )
    verbose_errors: bool | None = Field(
        default=None,
        description="Return full error traceback. None=use server default, True=full traceback, False=short message",
    )
    enable_profiling: bool | None = Field(
        default=None,
        description="Enable torch.profiler for this request. None=use server default, True=enable, False=disable",
    )
    enable_triton_detection: bool | None = Field(
        default=None,
        description="Enable Triton kernel usage detection (decoy check)",
    )
    measure_performance: bool | None = Field(
        default=None,
        description="Measure kernel performance timing (default True for kernelbench)",
    )
    run_correctness: bool | None = Field(
        default=None,
        description="Run correctness checks (default True for kernelbench)",
    )
    run_triton_detection: bool | None = Field(
        default=None,
        description="Run Triton usage detection step (overrides enable_triton_detection)",
    )
    run_performance: bool | None = Field(
        default=None,
        description="Run performance timing step (overrides measure_performance)",
    )
    cases_code: str | None = Field(
        default=None,
        description="Python code defining get_cases()/get_inputs() for kernel_simple workflow",
    )
    cases: list[Any] | None = Field(
        default=None,
        description="Inline cases for kernel_simple workflow",
    )
    workflow: str | None = Field(
        default="kernelbench",
        description="Workflow controller name (e.g. kernelbench)",
    )
    resources: dict[str, Any] | None = Field(
        default=None,
        description="Resource requirements (e.g. {'gpus': 2})",
    )

    @validator("task_id")
    def validate_task_id(cls, v):
        if not v or len(v) < 1 or len(v) > 100:
            raise ValueError("Task ID must be between 1 and 100 characters")
        return v

    @validator("reference_code", "kernel_code")
    def validate_code(cls, v):
        if v is None:
            return v
        if not v or len(v.strip()) < 10:
            raise ValueError("Code must be at least 10 characters long")
        if len(v) > 100000:
            raise ValueError("Code must be less than 100KB")
        return v

    @root_validator(skip_on_failure=True)
    def validate_reference_requirement(cls, values):
        workflow = (values.get("workflow") or "kernelbench").strip().lower()
        reference_code = values.get("reference_code")
        if workflow == "kernelbench" and not reference_code:
            raise ValueError("reference_code is required for kernelbench workflow")
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "rl_batch_001",
                "reference_code": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)",
                "kernel_code": "import torch\nimport torch.nn as nn\n\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        # Custom CUDA kernel implementation\n        return torch.relu(x)",
                "toolkit": "kernelbench",
                "backend_adapter": "kernelbench",
                "backend": "cuda",
                "num_correct_trials": 5,
                "num_perf_trials": 100,
                "timeout": 300,
                "priority": "normal",
                "entry_point": "Model",
                "is_valid": False,
                "verbose_errors": None,
                "enable_profiling": None,
            }
        }


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""

    task_id: str
    status: TaskStatus
    compiled: bool | None = None
    correctness: bool | None = None
    decoy_kernel: bool | None = None
    reference_runtime: float | None = None
    kernel_runtime: float | None = None
    speedup: float | None = None
    metadata: dict[str, Any] | None = None
    error_message: str | None = None
    error_code: ErrorCode | None = None
    submitted_at: str | None = None
    completed_at: str | None = None
    processing_time: float | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "rl_batch_001",
                "status": "completed",
                "compiled": True,
                "correctness": True,
                "reference_runtime": 2.5,
                "kernel_runtime": 1.2,
                "speedup": 2.08,
                "metadata": {"device": "cuda:0", "gpu_name": "NVIDIA H100", "backend": "cuda"},
                "submitted_at": "2025-01-16T10:30:00Z",
                "completed_at": "2025-01-16T10:30:15Z",
                "processing_time": 15.2,
                "error_code": None,
            }
        }


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation."""

    batch_id: str = Field(..., description="Unique batch identifier")
    tasks: list[EvaluationRequest] = Field(..., description="List of evaluation tasks")

    @validator("tasks")
    def validate_tasks(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Batch must contain at least one task")
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 tasks")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "rl_batch_001",
                "tasks": [
                    {
                        "task_id": "task_001",
                        "reference_code": "# PyTorch code...",
                        "kernel_code": "# Custom kernel...",
                        "backend": "cuda",
                        "entry_point": "Model",
                        "is_valid": False,
                    }
                ],
            }
        }


class BatchEvaluationResponse(BaseModel):
    """Response model for batch evaluation."""

    batch_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    results: list[EvaluationResponse]
    batch_status: TaskStatus
    submitted_at: str | None = None
    completed_at: str | None = None


class TaskStatusResponse(BaseModel):
    """Response model for task status query."""

    task_id: str
    status: TaskStatus
    progress: float | None = Field(default=None, description="Progress percentage (0-100)")
    estimated_completion: str | None = Field(default=None, description="Estimated completion time")
    queue_position: int | None = Field(default=None, description="Position in queue")
    assigned_device: str | None = Field(default=None, description="Assigned GPU device")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "rl_batch_001",
                "status": "processing",
                "progress": 45.0,
                "estimated_completion": "2025-01-16T10:32:00Z",
                "queue_position": 2,
                "assigned_device": "cuda:3",
            }
        }


class WorkflowRequest(BaseModel):
    """Generic workflow submission request."""

    workflow: str = Field(default="kernelbench", description="Workflow controller name")
    payload: dict[str, Any] = Field(default_factory=dict, description="Workflow-specific payload")
    task_id: str | None = Field(default=None, description="Optional task id override")
    force_refresh: bool = Field(default=False, description="Force refresh, skip cached results")
    resources: dict[str, Any] | None = Field(
        default=None,
        description="Resource requirements (e.g. {'gpus': 2})",
    )


class WorkflowResponse(BaseModel):
    """Generic workflow submission response."""

    task_id: str
    status: TaskStatus
    result: dict[str, Any] | None = None
    error_message: str | None = None
    error_code: ErrorCode | None = None
    submitted_at: str | None = None
    completed_at: str | None = None


class SystemHealthResponse(BaseModel):
    """Response model for system health check."""

    status: str
    timestamp: str
    gpu_status: dict[str, Any]
    queue_status: dict[str, Any]
    memory_usage: dict[str, Any]
    active_tasks: int
    total_processed: int
    uptime: float

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-16T10:30:00Z",
                "gpu_status": {
                    "cuda:0": {"utilization": 85.5, "memory_used": "12GB", "memory_total": "80GB"},
                    "cuda:1": {"utilization": 23.1, "memory_used": "4GB", "memory_total": "80GB"},
                },
                "queue_status": {"pending": 15, "processing": 8, "completed": 1250},
                "memory_usage": {"cpu_percent": 45.2, "memory_percent": 67.8},
                "active_tasks": 23,
                "total_processed": 1245,
                "uptime": 86400.5,
            }
        }


class MetricsResponse(BaseModel):
    """Response model for system metrics."""

    timestamp: str
    performance_metrics: dict[str, Any]
    resource_metrics: dict[str, Any]
    queue_metrics: dict[str, Any]
    error_metrics: dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    message: str
    error_code: ErrorCode | None = None
    task_id: str | None = None
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid code format",
                "error_code": "VALIDATION_ERROR",
                "task_id": "rl_batch_001",
                "timestamp": "2025-01-16T10:30:00Z",
            }
        }
