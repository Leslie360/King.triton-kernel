# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel data-processing utilities.

Used for handling kernel-code-generation datasets.
"""

import json
import re
from typing import Dict, List, Any, Optional


def process_kernel_data(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single kernel data item.

    Args:
        data_item: The raw data item.

    Returns:
        The processed data item.
    """
    # Normalize field names
    processed_item = {
        "prompt": data_item.get("prompt", data_item.get("instruction", "")),
        "reference_code": data_item.get("reference_code", data_item.get("reference", "")),
        "description": data_item.get("description", ""),
        "test_cases": data_item.get("test_cases", []),
        "requirements": data_item.get("requirements", {}),
        "data_source": data_item.get("data_source", "kernel"),
        "difficulty": data_item.get("difficulty", "medium"),
        "tags": data_item.get("tags", [])
    }

    # Ensure required fields are present
    if not processed_item["prompt"]:
        processed_item["prompt"] = "Generate an optimized CUDA kernel implementation."

    if not processed_item["reference_code"]:
        processed_item["reference_code"] = "# No reference code provided"

    return processed_item


def create_kernel_prompt(
    description: str,
    reference_code: str,
    requirements: Optional[Dict[str, Any]] = None,
    test_cases: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build a kernel-generation prompt.

    Args:
        description: Task description.
        reference_code: Reference implementation.
        requirements: Performance and constraint requirements.
        test_cases: Test cases.

    Returns:
        The formatted prompt string.
    """
    prompt_parts = []

    # Task description
    prompt_parts.append("# Task Description")
    prompt_parts.append(description)
    prompt_parts.append("")

    # Reference implementation
    if reference_code and reference_code.strip():
        prompt_parts.append("# Reference Implementation")
        prompt_parts.append("```python")
        prompt_parts.append(reference_code)
        prompt_parts.append("```")
        prompt_parts.append("")

    # Performance requirements
    if requirements:
        prompt_parts.append("# Performance Requirements")
        for key, value in requirements.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Test cases
    if test_cases:
        prompt_parts.append("# Test Cases")
        for i, test_case in enumerate(test_cases):
            prompt_parts.append(f"## Test Case {i+1}")
            if "input" in test_case:
                prompt_parts.append(f"Input: {test_case['input']}")
            if "expected_output" in test_case:
                prompt_parts.append(f"Expected Output: {test_case['expected_output']}")
            if "description" in test_case:
                prompt_parts.append(f"Description: {test_case['description']}")
            prompt_parts.append("")

    # Generation task
    prompt_parts.append("# Your Task")
    prompt_parts.append("Please generate an optimized CUDA kernel implementation that:")
    prompt_parts.append("1. Achieves better performance than the reference implementation")
    prompt_parts.append("2. Maintains correctness and numerical stability")
    prompt_parts.append("3. Follows CUDA best practices")
    prompt_parts.append("4. Is well-documented with comments")
    prompt_parts.append("")
    prompt_parts.append("# Generated Kernel Implementation")
    prompt_parts.append("```python")

    return "\n".join(prompt_parts)


def extract_kernel_requirements(prompt: str) -> Dict[str, Any]:
    """
    Extract kernel requirements from the prompt.

    Args:
        prompt: The prompt string.

    Returns:
        The extracted requirements dict.
    """
    requirements = {}

    # Extract performance requirements
    perf_pattern = r"# Performance Requirements\s*\n(.*?)(?=\n#|\n$)"
    perf_match = re.search(perf_pattern, prompt, re.DOTALL)
    if perf_match:
        perf_lines = perf_match.group(1).strip().split('\n')
        for line in perf_lines:
            if line.strip().startswith('-'):
                parts = line.strip()[1:].split(':')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = ':'.join(parts[1:]).strip()
                    requirements[key] = value

    # Extract test-case marker
    test_pattern = r"# Test Cases\s*\n(.*?)(?=\n#|\n$)"
    test_match = re.search(test_pattern, prompt, re.DOTALL)
    if test_match:
        requirements["has_test_cases"] = True

    # Extract task description
    desc_pattern = r"# Task Description\s*\n(.*?)(?=\n#|\n$)"
    desc_match = re.search(desc_pattern, prompt, re.DOTALL)
    if desc_match:
        requirements["description"] = desc_match.group(1).strip()

    return requirements


def load_kernel_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load a kernel dataset.

    Args:
        dataset_path: Path to the dataset.

    Returns:
        The processed dataset.
    """
    dataset = []

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            # Supports JSON Lines format
            if dataset_path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        data_item = json.loads(line)
                        processed_item = process_kernel_data(data_item)
                        dataset.append(processed_item)
            else:
                # JSON format
                data = json.load(f)
                if isinstance(data, list):
                    for data_item in data:
                        processed_item = process_kernel_data(data_item)
                        dataset.append(processed_item)
                else:
                    processed_item = process_kernel_data(data)
                    dataset.append(processed_item)

    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return []

    return dataset


def validate_kernel_code(kernel_code: str) -> Dict[str, Any]:
    """
    Validate the basic format of kernel code.

    Args:
        kernel_code: Kernel-code string.

    Returns:
        The validation result.
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    # Basic checks
    if not kernel_code.strip():
        result["valid"] = False
        result["errors"].append("Empty kernel code")
        return result

    # Check for required imports
    required_imports = ["torch", "cuda"]
    for imp in required_imports:
        if imp not in kernel_code:
            result["warnings"].append(f"Missing import: {imp}")

    # Check for kernel-function definition
    if "def " not in kernel_code:
        result["warnings"].append("No function definition found")

    # Check for CUDA-specific code
    cuda_keywords = ["cuda", "gpu", "device", "kernel", "block", "thread"]
    if not any(keyword in kernel_code.lower() for keyword in cuda_keywords):
        result["warnings"].append("No CUDA-specific keywords found")

    return result