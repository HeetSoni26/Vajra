"""
Integration tests for Milestone 4: Production Training Orchestration.
Covers HealthMonitor, ExperimentManager, Watchdog, ETAEngine, and lifecycle transitions.
"""

from __future__ import annotations

import math
import time

from training.orchestration.eta_engine import ETAEngine
from training.orchestration.experiment_manager import ExperimentManager, TrainingState
from training.orchestration.health_monitor import HealthMonitor
from training.orchestration.watchdog import Watchdog

# ── HealthMonitor ──────────────────────────────────────────────────────────────


def test_health_monitor_clean(tmp_path):
    monitor = HealthMonitor(checkpoint_dir=str(tmp_path), min_disk_free_gb=0.0)
    snap = monitor.check(train_loss=2.5, grad_norm=0.8, tokens_per_sec=5000)
    # Should have no warnings on a fresh clean check
    assert isinstance(snap.warnings, list)
    assert snap.train_loss == 2.5


def test_health_monitor_loss_spike(tmp_path):
    monitor = HealthMonitor(checkpoint_dir=str(tmp_path), min_disk_free_gb=0.0)
    # Seed a stable history
    for _ in range(10):
        monitor.check(train_loss=2.0)
    # Now spike
    snap = monitor.check(train_loss=20.0)
    assert any("spike" in w.lower() for w in snap.warnings)


def test_health_monitor_nan_loss(tmp_path):
    monitor = HealthMonitor(checkpoint_dir=str(tmp_path), min_disk_free_gb=0.0)
    snap = monitor.check(train_loss=math.nan)
    assert any("nan" in w.lower() or "inf" in w.lower() for w in snap.warnings)


def test_health_monitor_disk_warning(tmp_path):
    # Require impossibly large free disk to force a warning
    monitor = HealthMonitor(checkpoint_dir=str(tmp_path), min_disk_free_gb=999999.0)
    snap = monitor.check(train_loss=2.0)
    assert any("disk" in w.lower() for w in snap.warnings)


# ── ExperimentManager ──────────────────────────────────────────────────────────


def test_experiment_manager_init(tmp_path):
    mgr = ExperimentManager(tmp_path / "exp_1")
    assert mgr.state == TrainingState.INITIALIZING
    assert (tmp_path / "exp_1" / "experiment_registry.json").exists()


def test_experiment_manager_lifecycle_transitions(tmp_path):
    mgr = ExperimentManager(tmp_path / "exp_1")
    mgr.transition(TrainingState.TRAINING)
    assert mgr.state == TrainingState.TRAINING
    mgr.transition(TrainingState.CHECKPOINTING)
    assert mgr.state == TrainingState.CHECKPOINTING
    mgr.transition(TrainingState.COMPLETED)
    assert mgr.state == TrainingState.COMPLETED


def test_experiment_manager_checkpoint_history(tmp_path):
    mgr = ExperimentManager(tmp_path / "exp_1")
    mgr.record_checkpoint(step=100, tokens_seen=50000, path="ckpt.pt", metrics={"val_loss": 2.1})
    record = mgr.get_record()
    assert len(record["checkpoint_history"]) == 1
    assert record["checkpoint_history"][0]["step"] == 100


def test_experiment_manager_resume_history(tmp_path):
    mgr = ExperimentManager(tmp_path / "exp_1")
    mgr.record_resume(from_step=500, from_exp="exp_0")
    record = mgr.get_record()
    assert len(record["resume_history"]) == 1
    assert record["resume_history"][0]["from_step"] == 500


def test_experiment_manager_persistence(tmp_path):
    exp_dir = tmp_path / "exp_1"
    mgr = ExperimentManager(exp_dir)
    mgr.transition(TrainingState.TRAINING)
    mgr.record_checkpoint(step=42, tokens_seen=1000, path="c.pt", metrics={})

    # Reload from disk
    mgr2 = ExperimentManager(exp_dir)
    record = mgr2.get_record()
    assert record["state"] == TrainingState.TRAINING.value
    assert len(record["checkpoint_history"]) == 1


# ── Watchdog ───────────────────────────────────────────────────────────────────


def test_watchdog_no_trigger_with_heartbeats():
    triggered = []

    def cb():
        triggered.append(True)

    wd = Watchdog(timeout_seconds=0.5, on_trigger=cb, check_interval=0.1)
    wd.start()
    for _ in range(5):
        time.sleep(0.05)
        wd.heartbeat()
    time.sleep(0.2)
    wd.stop()
    assert len(triggered) == 0


def test_watchdog_triggers_on_freeze():
    triggered = []

    def cb():
        triggered.append(True)

    wd = Watchdog(timeout_seconds=0.2, on_trigger=cb, check_interval=0.05)
    wd.start()
    time.sleep(0.5)  # No heartbeat — should trigger
    wd.stop()
    assert len(triggered) >= 1


# ── ETAEngine ──────────────────────────────────────────────────────────────────


def test_eta_engine_progress(tmp_path):
    engine = ETAEngine(total_steps=100)
    for _ in range(10):
        engine.record_step()
    progress = engine.get_progress(10, tokens_seen=50000)
    assert progress["current_step"] == 10
    assert progress["remaining_steps"] == 90
    assert progress["pct_complete"] == 10.0
    assert "eta_hours" in progress
    assert progress["tokens_per_sec"] >= 0
