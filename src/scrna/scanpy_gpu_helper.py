"""
Utilities to seamlessly switch between Scanpy (CPU) and RAPIDS‑SingleCell (GPU)

from gpu_scanpy_helper import pick_backend, gpu_session

Two ways to use it
------------------
1) **Drop‑in Scanpy shim** (recommended):

    sc = pick_backend()  # looks like scanpy, uses GPU+rsc if available
    adata = sc.read_h5ad("pbmc.h5ad")
    
    # Enable memory management for large datasets
    sc.enable_memory_manager()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)
    # free up GPU memory
    sc.to_cpu(adata)
    gc.collect()

   Most existing Scanpy code should work unchanged after replacing
   `import scanpy as sc` with `sc = pick_backend()`.

2) **Explicit backend object** (if you prefer):

    from gpu_scanpy_helper import pick_backend, gpu_session
    b = pick_backend()
    b.enable_memory_manager()
    with gpu_session(b, adata):
        b.pp.normalize_total(adata)
        b.tl.pca(adata)

Environment knobs
-----------------
- Set `RSC_FORCE_CPU=1` to force CPU/Scanpy even if a GPU is present.
- Set `RSC_PREFER_GPU=0` to prefer CPU unless GPU is explicitly requested.

Notes
-----
- If a GPU exists but `rapids-singlecell`/`rsc` is not importable, a warning is emitted.
- When the GPU path is active, wrappers will move the passed `AnnData` to GPU *before*
 calling the underlying function. They do **not** automatically move it back; call
 `sc.to_cpu(adata)` or use `gpu_session(..., leave_on_gpu=False)` to return to CPU at block exit.
- Memory management can be enabled once via `sc.enable_memory_manager()` 
 and will apply to all subsequent operations for that backend instance.
"""

from __future__ import annotations

import os
import contextlib
import warnings
from typing import Any, Callable, Optional

__all__ = [
    "pick_backend",
    "gpu_session",
    "Backend",
]


class _Namespace:
    """Proxy for .pp/.tl/.pl that wraps callables.

    For .pp and .tl, we ensure AnnData is on GPU when using rsc.
    For .pl we just forward the call (plotting is typically CPU‑side).
    """

    def __init__(self, backend: "Backend", submodule: str, move_to_gpu: bool):
        self._b = backend
        self._sub = submodule
        self._move = move_to_gpu

    def __getattr__(self, name: str) -> Callable[..., Any]:
        lib = self._b._lib()
        target_ns = getattr(lib, self._sub)
        target = getattr(target_ns, name)

        if not callable(target):
            return target  # constants etc.

        def wrapper(*args: Any, **kwargs: Any):
            # Heuristics: first positional arg is almost always AnnData
            if self._move and self._b._using_rsc and args:
                self._b.to_gpu(args[0])
            return target(*args, **kwargs)

        wrapper.__name__ = getattr(target, "__name__", name)
        wrapper.__doc__ = getattr(target, "__doc__", None)
        return wrapper


class _ExternalPPNamespace:
    """Maps `sc.external.pp.*` -> `rsc.pp.*` when using rsc; otherwise passthrough.

    If a function is missing in `rsc.pp`, normal attribute errors will be raised.
    For CPU/scanpy, we forward to `scanpy.external.pp`.
    """

    def __init__(self, backend: "Backend"):
        self._b = backend

    def __getattr__(self, name: str):
        if self._b._using_rsc:
            target_ns = getattr(self._b._rsc, "pp")  # map external.pp → rsc.pp
        else:
            target_ns = getattr(self._b._sc.external, "pp")

        target = getattr(target_ns, name)  # may raise AttributeError (intended)

        if not callable(target):
            return target

        def wrapper(*args: Any, **kwargs: Any):
            # Move AnnData to GPU before the call when on rsc
            if self._b._using_rsc and args:
                self._b.to_gpu(args[0])
            return target(*args, **kwargs)

        wrapper.__name__ = getattr(target, "__name__", name)
        wrapper.__doc__ = getattr(target, "__doc__", None)
        return wrapper


class _ExternalFacade:
    def __init__(self, backend: "Backend"):
        self.pp = _ExternalPPNamespace(backend)


class Backend:
    """Scanpy-compatible facade that auto-routes to GPU (rsc) when available.

    Behaves like the `scanpy` module: exposes `.pp`, `.tl`, `.pl`, and also
    forwards other top-level attributes (e.g. `read_h5ad`, `settings`, etc.).
    
    Supports GPU memory management via `enable_memory_manager()` for efficient
    handling of large datasets.
    """

    def __init__(self, prefer_gpu: bool = True, force_cpu: bool = False):
        self.prefer_gpu = prefer_gpu
        self.force_cpu = force_cpu or os.getenv("RSC_FORCE_CPU") in {
            "1",
            "true",
            "True",
        }

        import scanpy as sc  # must exist as CPU fallback

        self._sc = sc

        self._rsc = None
        self._using_rsc = False
        self.is_gpu = False
        self._memory_manager_enabled = False

        if not self.force_cpu and self.prefer_gpu:
            self._init_rsc_if_possible()

        # Namespaces that mimic scanpy API
        self.pp = _Namespace(self, "pp", move_to_gpu=True)
        self.tl = _Namespace(self, "tl", move_to_gpu=True)
        # Always use scanpy for .pl, doesn't exist in rsc
        self.pl = self._sc.pl
        # external.pp maps to rsc.pp on GPU; passthrough on CPU
        self.external = _ExternalFacade(self)

    # ---------- detection ----------
    def _has_cuda(self) -> bool:
        try:
            import cupy as cp  # type: ignore

            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def _init_rsc_if_possible(self) -> None:
        if not self._has_cuda():
            return
        try:
            import rapids_singlecell as rsc  # preferred
        except Exception:
            return
        self._rsc = rsc
        self._using_rsc = True
        self.is_gpu = True

    # ---------- memory management ----------
    def enable_memory_manager(self) -> None:
        """Enable RAPIDS memory management for GPU operations.
        
        Use this once before calling processing functions to enable memory-efficient
        operations on large datasets. On CPU backend, this is a no-op.
        
        Example
        -------
        >>> sc = pick_backend()
        >>> sc.enable_memory_manager()
        >>> # Now all subsequent operations use managed memory
        >>> sc.pp.normalize_total(adata)
        """
        if not self._using_rsc:
            # CPU backend doesn't need memory management setup
            return
        
        try:
            import rmm
            import cupy as cp
            from rmm.allocators.cupy import rmm_cupy_allocator
            
            # Enable managed memory with RMM
            rmm.reinitialize(managed_memory=True, pool_allocator=False)
            cp.cuda.set_allocator(rmm_cupy_allocator)
            
            self._memory_manager_enabled = True
            print("Memory management enabled with RMM managed_memory")
        except Exception as e:
            warnings.warn(
                f"Failed to enable memory management: {e}. "
                "Proceeding without memory management. "
                "Ensure rmm is installed."
            )

    # ---------- device moves ----------
    def to_gpu(self, adata: Any, layer: Optional[str] = None, convert_all: bool = False) -> Any:
        if self._using_rsc and hasattr(self._rsc, "get"):
            fn = getattr(self._rsc.get, "anndata_to_GPU", None)
            if callable(fn):
                return fn(adata, layer=layer, convert_all=convert_all)
        return None

    def to_cpu(self, adata: Any, layer: Optional[str] = None, convert_all: bool = False) -> Any:
        if self._using_rsc and hasattr(self._rsc, "get"):
            fn = getattr(self._rsc.get, "anndata_to_CPU", None)
            if callable(fn):
                return fn(adata, layer=layer, convert_all=convert_all)
        return None

    # ---------- module forwarding ----------
    def _lib(self):
        return self._rsc if self._using_rsc else self._sc

    def __getattr__(self, name: str):
        # Provide scanpy‑like top‑level API: read_h5ad, read, settings, etc.
        # pp/tl/pl/external are handled via explicit attributes above.
        # If using rsc and attribute exists, use it; otherwise always fallback to scanpy
        if self._using_rsc and hasattr(self._rsc, name):
            return getattr(self._rsc, name)
        return getattr(self._sc, name)


def pick_backend(
    *, prefer_gpu: Optional[bool] = None, force_cpu: Optional[bool] = None
) -> Backend:
    """Choose GPU (rsc) or CPU (scanpy) and return a Scanpy-compatible object.

    - prefer_gpu: defaults to env var RSC_PREFER_GPU (default True)
    - force_cpu: defaults to env var RSC_FORCE_CPU (default False)

    Typical use:
        sc = pick_backend()
    """
    if prefer_gpu is None:
        prefer_gpu = os.getenv("RSC_PREFER_GPU", "1") not in {"0", "false", "False"}
    if force_cpu is None:
        force_cpu = os.getenv("RSC_FORCE_CPU") in {"1", "true", "True"}
    b = Backend(prefer_gpu=bool(prefer_gpu), force_cpu=bool(force_cpu))

    if b.prefer_gpu and not b.force_cpu and not b._using_rsc and b._has_cuda():
        warnings.warn(
            "CUDA was detected but rapids-singlecell (rsc) was not importable. "
            "Install it on this machine to enable GPU acceleration."
        )
    return b


@contextlib.contextmanager
def gpu_session(backend: Backend, adata: Any, *, leave_on_gpu: bool = False):
    """Keep AnnData on GPU for a block of work when rsc is active.

    If CPU backend is active, this is a no-op. If using rsc, moves to GPU on
    entry and (unless `leave_on_gpu=True`) back to CPU on exit.
    """
    if getattr(backend, "_using_rsc", False):
        backend.to_gpu(adata)
        try:
            yield
        finally:
            if not leave_on_gpu:
                backend.to_cpu(adata)
    else:
        yield
