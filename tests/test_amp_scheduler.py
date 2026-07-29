from __future__ import annotations

import unittest

from setv.experts.train_object import _step_optimizer_and_scheduler


class _Optimizer:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class _Scheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class _Scaler:
    def __init__(self, *, skip: bool) -> None:
        self.scale = 65536.0
        self.skip = skip

    def get_scale(self) -> float:
        return self.scale

    def step(self, optimizer: _Optimizer) -> None:
        if not self.skip:
            optimizer.step()

    def update(self) -> None:
        if self.skip:
            self.scale /= 2.0


class AmpSchedulerTests(unittest.TestCase):
    def test_scheduler_advances_after_applied_optimizer_step(self) -> None:
        optimizer = _Optimizer()
        scheduler = _Scheduler()
        applied = _step_optimizer_and_scheduler(
            scaler=_Scaler(skip=False),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.assertTrue(applied)
        self.assertEqual(optimizer.step_count, 1)
        self.assertEqual(scheduler.step_count, 1)

    def test_scheduler_waits_when_grad_scaler_skips_optimizer_step(self) -> None:
        optimizer = _Optimizer()
        scheduler = _Scheduler()
        applied = _step_optimizer_and_scheduler(
            scaler=_Scaler(skip=True),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.assertFalse(applied)
        self.assertEqual(optimizer.step_count, 0)
        self.assertEqual(scheduler.step_count, 0)


if __name__ == "__main__":
    unittest.main()
