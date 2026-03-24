import re

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    _build_runtime_instance_name,
    execute_multi_turn_func_call,
)


def test_runtime_instance_name_is_valid_python_identifier():
    instance_name = _build_runtime_instance_name(
        model_name="ollama::qwen3.5:9b::prompt",
        test_entry_id="memory_rec_sum_prereq_0-customer-0",
        class_name="MemoryAPI_rec_sum",
        is_eval_run=False,
    )

    assert re.fullmatch(r"[A-Za-z_]\w*", instance_name)
    assert ":" not in instance_name
    assert "." not in instance_name
    assert "-" not in instance_name


def test_execute_multi_turn_func_call_handles_ollama_style_model_name(tmp_path):
    initial_config = {
        "MemoryAPI_rec_sum": {
            "model_result_dir": tmp_path,
            "scenario": "customer",
            "test_id": "memory_rec_sum_prereq_0-customer-1",
            "test_category": "memory_rec_sum_prereq",
        }
    }

    execution_results, involved_instances = execute_multi_turn_func_call(
        func_call_list=["memory_append(text='remember this')"],
        initial_config=initial_config,
        involved_classes=["MemoryAPI_rec_sum"],
        model_name="ollama::qwen3.5:9b::prompt",
        test_entry_id="memory_rec_sum_prereq_0-customer-0",
    )

    assert execution_results == ['{"status": "Memory appended."}']
    assert involved_instances["MemoryAPI_rec_sum"].memory == "remember this"


def test_execute_multi_turn_func_call_uses_structured_calls_for_strings_with_apostrophe(
    tmp_path,
):
    initial_config = {
        "MemoryAPI_rec_sum": {
            "model_result_dir": tmp_path,
            "scenario": "customer",
            "test_id": "memory_rec_sum_prereq_0-customer-0",
            "test_category": "memory_rec_sum_prereq",
        }
    }
    func_call_list = [
        "memory_append(text=\"Michael's reminder\")",
    ]
    func_call_list = type("StructuredExecutionList", (list,), {})(
        func_call_list
    )
    func_call_list.structured_calls = [
        {"memory_append": {"text": "Michael's reminder"}}
    ]

    execution_results, involved_instances = execute_multi_turn_func_call(
        func_call_list=func_call_list,
        initial_config=initial_config,
        involved_classes=["MemoryAPI_rec_sum"],
        model_name="ollama::qwen3.5:9b::prompt",
        test_entry_id="memory_rec_sum_prereq_0-customer-1",
    )

    assert execution_results == ['{"status": "Memory appended."}']
    assert involved_instances["MemoryAPI_rec_sum"].memory == "Michael's reminder"
