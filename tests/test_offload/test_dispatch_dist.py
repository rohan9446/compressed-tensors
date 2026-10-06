# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_source_process, set_source_process
from compressed_tensors.offload import disable_onloading
from compressed_tensors.offload.cache import dist_batch
from compressed_tensors.offload.cache.dist_batch import batch_offload_sync
from compressed_tensors.offload.dispatch import dispatch_with_map
from compressed_tensors.offload.module import offload_module
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


CPU = torch.device("cpu")


def _linears(num: int, meta: bool = False) -> torch.nn.ModuleList:
    """Same seeded weights on every rank; `meta` mimics non-source ranks at load"""
    torch.manual_seed(0)
    with torch.device("meta" if meta else "cpu"):
        return torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(num))


def _assert_matches(model: torch.nn.Module, expected: torch.nn.Module):
    with torch.no_grad():
        for name, tensor in expected.state_dict().items():
            actual = model.get_submodule(name.rpartition(".")[0])
            actual = getattr(actual, name.rpartition(".")[2])
            assert actual.dtype == tensor.dtype, name
            assert torch.equal(actual.cpu(), tensor), name


@contextlib.contextmanager
def _count_collectives():
    with (
        patch(
            "torch.distributed.broadcast_object_list",
            wraps=dist.broadcast_object_list,
        ) as broadcast_object_list,
        patch("torch.distributed.barrier", wraps=dist.barrier) as barrier,
    ):
        yield broadcast_object_list, barrier


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_exchanges_metadata_once(accel_device, offload_folder):
    for offload_device in (CPU, "disk"):
        model = _linears(6, meta=not is_source_process())
        device_map = {str(i): (accel_device, offload_device) for i in range(6)}

        with _count_collectives() as (broadcast_object_list, barrier):
            dispatch_with_map(
                model, device_map, offload_dir=offload_folder, show_progress=False
            )

        # previously one of each per tensor (12 tensors)
        assert broadcast_object_list.call_count == 1
        assert barrier.call_count == 1
        _assert_matches(model, _linears(6))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_mixed_offload_devices(accel_device, offload_folder):
    model = _linears(4, meta=not is_source_process())
    device_map = {
        "0": (accel_device, CPU),
        "1": (accel_device, "disk"),
        "2": (accel_device, accel_device),  # tensor data broadcast, not batched
        "3": (accel_device, CPU),
    }

    with _count_collectives() as (broadcast_object_list, _):
        dispatch_with_map(
            model, device_map, offload_dir=offload_folder, show_progress=False
        )

    assert broadcast_object_list.call_count == 1
    _assert_matches(model, _linears(4))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_tied_weights(accel_device):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)
            self.head.weight = self.embed.weight

    torch.manual_seed(0)
    with torch.device("meta" if not is_source_process() else "cpu"):
        model = Tied()
    torch.manual_seed(0)
    expected = Tied()

    device_map = {"embed": (accel_device, CPU), "head": (accel_device, CPU)}
    dispatch_with_map(model, device_map, show_progress=False)

    # the tie is kept on every rank: one shared storage, not two
    with disable_onloading():
        embed, head = model.embed.weight, model.head.weight
        assert embed.untyped_storage().data_ptr() == head.untyped_storage().data_ptr()

    _assert_matches(model, expected)

    # every rank finishes checking the original values before the source mutates
    # the shared storage
    dist.barrier()

    # an in-place update on the source reaches both aliases on every rank
    if is_source_process():
        with disable_onloading(), torch.no_grad():
            model.embed.weight.fill_(1.0)
    dist.barrier()

    with torch.no_grad():
        expected.embed.weight.fill_(1.0)
    _assert_matches(model, expected)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_rebuilds_dtype_and_shape(accel_device, offload_folder):
    class Rotary(torch.nn.Module):
        def __init__(self, size: int, dtype: torch.dtype):
            super().__init__()
            self.register_buffer("inv_freq", torch.arange(size, dtype=dtype))

    for offload_device in (CPU, "disk"):
        # non-source ranks may init buffers differently than the checkpoint
        if is_source_process():
            model = Rotary(8, torch.float32)
        else:
            with torch.device("meta"):
                model = Rotary(4, torch.float64)

        device_map = {"": (accel_device, offload_device)}
        dispatch_with_map(
            model, device_map, offload_dir=offload_folder, show_progress=False
        )

        _assert_matches(model, Rotary(8, torch.float32))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_modules_missing_on_some_ranks(accel_device):
    model = _linears(3, meta=not is_source_process())
    if not is_source_process():
        model[0] = None  # e.g. a routed expert this rank does not own

    device_map = {str(i): (accel_device, CPU) for i in range(3)}
    with _count_collectives() as (broadcast_object_list, _):
        dispatch_with_map(model, device_map, show_progress=False)

    assert broadcast_object_list.call_count == 1
    expected = _linears(3)
    if not is_source_process():
        expected[0] = None
    _assert_matches(model, expected)


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_offload_after_dispatch_syncs_per_tensor(accel_device):
    model = _linears(2, meta=not is_source_process())
    device_map = {str(i): (accel_device, CPU) for i in range(2)}
    dispatch_with_map(model, device_map, show_progress=False)

    # outside of dispatch, new offloads are still exchanged immediately
    with _count_collectives() as (broadcast_object_list, barrier):
        model[0]._parameters["extra"] = torch.full((3,), 7.0)

    assert broadcast_object_list.call_count == 1
    assert barrier.call_count == 1
    assert torch.equal(model[0].extra.cpu(), torch.full((3,), 7.0))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_replica_without_local_modules(accel_device):
    model = _linears(2, meta=not is_source_process())
    if not is_source_process():
        model[0] = None
        model[1] = None

    device_map = {str(i): (accel_device, CPU) for i in range(2)}
    with _count_collectives() as (broadcast_object_list, barrier):
        dispatch_with_map(model, device_map, show_progress=False)

    # a rank with nothing to rebuild still joins the single exchange
    assert broadcast_object_list.call_count == 1
    assert barrier.call_count == 1
    if is_source_process():
        _assert_matches(model, _linears(2))


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_dispatch_rejects_nested_batch(accel_device):
    model = _linears(2, meta=not is_source_process())
    device_map = {str(i): (accel_device, CPU) for i in range(2)}

    with batch_offload_sync():
        with pytest.raises(RuntimeError, match="cannot be nested"):
            dispatch_with_map(model, device_map, show_progress=False)

    assert dist_batch._active_batch is None


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2, init_dist=True)
def test_failed_batch_resets_state(accel_device):
    model = _linears(2, meta=not is_source_process())

    # without module names, both modules record `weight` under the same key
    with pytest.raises(ValueError, match="already recorded"):
        with batch_offload_sync():
            for module in model:
                offload_module(module, accel_device, CPU)

    assert dist_batch._active_batch is None

    # a later, independent dispatch is unaffected
    model = _linears(2, meta=not is_source_process())
    device_map = {str(i): (accel_device, CPU) for i in range(2)}
    dispatch_with_map(model, device_map, show_progress=False)
    _assert_matches(model, _linears(2))


@pytest.mark.unit
@requires_gpu(3)
@torchrun(world_size=3, init_dist=True)
def test_dispatch_three_ranks_non_default_source(accel_device, offload_folder):
    with set_source_process(1):
        for offload_device in (CPU, "disk"):
            model = _linears(4, meta=not is_source_process())
            device_map = {str(i): (accel_device, offload_device) for i in range(4)}

            with _count_collectives() as (broadcast_object_list, barrier):
                dispatch_with_map(
                    model, device_map, offload_dir=offload_folder, show_progress=False
                )

            assert broadcast_object_list.call_count == 1
            assert barrier.call_count == 1
            _assert_matches(model, _linears(4))
