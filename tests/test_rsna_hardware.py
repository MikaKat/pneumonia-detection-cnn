"""Guards for the adapter choice added in phase 4.

THE SILENT FAILURE THIS FILE EXISTS FOR
---------------------------------------
`torch_directml.device()` without an index returns adapter 0, and adapter 0 on
this machine is the integrated graphics, not the RX 5500 XT. That mistake cost
this project weeks of slow training and left NO trace in any log, because
`privateuseone:0` names the interface and not the chip. A rule in a notebook
did not prevent it and would not prevent the next one. These checks do:

  * the index handed in must actually reach torch_directml,
  * an index that does not exist must stop the run instead of quietly falling
    back to adapter 0,
  * the source must never again contain a bare `torch_directml.device()`, and
  * every return path of pick_device must carry the label, because the label
    is the only thing that names the chip in the log and in results_rsna.csv.

No GPU, no torch-directml and no training needed: torch_directml is replaced by
a stub that records what it was asked for.

  python tests\\test_rsna_hardware.py
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def _stub_torch() -> None:
    """Same rule as in test_rsna_train.py: only stub when torch is REALLY absent.

    Stubbing on the mere absence from sys.modules breaks every machine where
    torch is installed, because scipy asks array_api_compat for torch.Tensor
    during the sklearn import.
    """
    try:
        import torch            # noqa: F401
        import torchvision      # noqa: F401
        return
    except ImportError:
        pass
    for name in ["torch", "torch.nn", "torch.utils", "torch.utils.data",
                 "torchvision", "torchvision.transforms", "torchvision.models"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch"].Tensor = type("Tensor", (), {})
    sys.modules["torch"].no_grad = lambda: (lambda f: f)
    sys.modules["torch"].device = lambda s: types.SimpleNamespace(type=s)
    sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch.utils.data"].Dataset = object
    sys.modules["torch.utils.data"].DataLoader = object
    sys.modules["torchvision.models"].resnet18 = None
    sys.modules["torchvision.models"].ResNet18_Weights = None
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]


_stub_torch()

import _repo_path  # noqa: F401  (sets sys.path)
import rsna_train                                          # noqa: E402
from rsna_train import dml_adapters, pick_device           # noqa: E402

TRAIN_SRC = Path(rsna_train.__file__).read_text(encoding="utf-8")
NAMES = ["AMD Radeon(TM) Graphics", "Radeon RX 5500 XT"]


class FakeDirectML(types.ModuleType):
    """Records every index it is asked for. Two adapters, like the real machine."""

    def __init__(self):
        super().__init__("torch_directml")
        self.asked = []

    def is_available(self):
        return True

    def device_count(self):
        return len(NAMES)

    def device_name(self, i):
        return NAMES[i]

    def device(self, index=None):
        self.asked.append(index)
        return types.SimpleNamespace(type="privateuseone", index=index)


def with_fake():
    fake = FakeDirectML()
    sys.modules["torch_directml"] = fake
    return fake


def test_index_reaches_the_driver() -> None:
    print("\ntest_index_reaches_the_driver")
    fake = with_fake()
    dev, pin, label = pick_device("directml", 1)
    check("the index is passed on, not dropped", fake.asked == [1],
          f"asked for {fake.asked}")
    check("the label names the chip", "Radeon RX 5500 XT" in label, label)
    check("the label names the index", label.startswith("directml:1"), label)
    check("pin_memory stays off for DirectML", pin is False)
    check("the device is the DirectML one", dev.type == "privateuseone")

    # The default must NOT change: an old command line has to keep meaning what
    # it meant, otherwise the reproduction of the 26.07. baseline breaks.
    fake = with_fake()
    _d, _p, label0 = pick_device("directml")
    check("the default is still adapter 0", fake.asked == [0],
          f"asked for {fake.asked}")
    check("and its label names the integrated chip",
          "AMD Radeon(TM) Graphics" in label0, label0)


def test_bad_index_stops_the_run() -> None:
    print("\ntest_bad_index_stops_the_run")
    for bad in (2, 7, -1):
        fake = with_fake()
        try:
            pick_device("directml", bad)
            check(f"index {bad} is refused", False, "it returned a device")
        except SystemExit as exc:
            # A silent fallback to adapter 0 is the exact failure mode this
            # phase exists to remove, so the wrong index must never end up in
            # a call to the driver.
            check(f"index {bad} is refused", fake.asked == [],
                  f"driver was asked for {fake.asked}")
            check(f"index {bad} error lists the adapters",
                  all(n in str(exc) for n in NAMES))


def test_adapter_listing() -> None:
    print("\ntest_adapter_listing")
    with_fake()
    check("dml_adapters returns both names", dml_adapters() == NAMES)

    # A driver that is installed but finds no device.
    dead = FakeDirectML()
    dead.is_available = lambda: False
    sys.modules["torch_directml"] = dead
    check("a driver without a device gives an empty list", dml_adapters() == [])

    # torch-directml not installed at all. `None` in sys.modules makes the
    # import statement raise ImportError, which is exactly the situation on a
    # fresh machine, and `liste` is the first thing such a machine runs.
    sys.modules["torch_directml"] = None
    check("no torch-directml gives an empty list, not a crash",
          dml_adapters() == [])
    sys.modules.pop("torch_directml", None)


def _pick_device_node() -> ast.FunctionDef:
    tree = ast.parse(TRAIN_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "pick_device":
            return node
    raise AssertionError("pick_device not found in rsna_train.py")


def test_no_bare_device_call() -> None:
    """THE regression test. Reads the source, not the behaviour.

    A behavioural test would pass again the moment someone reintroduces
    `torch_directml.device()` behind a branch this test does not reach. The
    call simply must not exist in the file any more.
    """
    print("\ntest_no_bare_device_call")
    bare = []
    for node in ast.walk(ast.parse(TRAIN_SRC)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "device"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch_directml"
                and not node.args and not node.keywords):
            bare.append(node.lineno)
    check("no bare torch_directml.device() anywhere in rsna_train.py",
          not bare, f"lines {bare}")


def test_every_return_carries_the_label() -> None:
    print("\ntest_every_return_carries_the_label")
    node = _pick_device_node()
    widths = [len(r.value.elts) for r in ast.walk(node)
              if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple)]
    check("pick_device has more than one return path", len(widths) >= 3,
          f"{len(widths)} found")
    check("every return path is a triple (device, pin, label)",
          set(widths) == {3}, f"widths {widths}")


def test_provenance_lands_in_the_result_row() -> None:
    """The label has to survive into results_rsna.csv, not only into the log.

    File names have already lied once in this project: the crop runs
    overwrote the baseline checkpoints and nothing noticed, because the name
    said nothing about what produced the file. The chip belongs in the data.
    """
    print("\ntest_provenance_lands_in_the_result_row")
    check("the run prints the chip", 'print(f"  Hardware: {dev_label}")' in TRAIN_SRC)
    check("device_name goes into the result row",
          '"device_name": dev_label' in TRAIN_SRC)
    check("dml_index goes into the result row", '"dml_index"' in TRAIN_SRC)
    check("the CLI offers --dml-index", '"--dml-index"' in TRAIN_SRC)
    check("the call site passes it on",
          "pick_device(args.device, args.dml_index)" in TRAIN_SRC)


if __name__ == "__main__":
    test_index_reaches_the_driver()
    test_bad_index_stops_the_run()
    test_adapter_listing()
    test_no_bare_device_call()
    test_every_return_carries_the_label()
    test_provenance_lands_in_the_result_row()
    print("\n" + ("ALL TESTS PASSED" if not FAILED
                  else f"{len(FAILED)} FAILED: {FAILED}"))
    raise SystemExit(1 if FAILED else 0)
