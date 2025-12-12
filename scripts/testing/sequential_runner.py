#!/usr/bin/env python3
"""
Sequential Test Runner for Alexandria
=====================================
Runs unit tests sequentially, capturing logs and metrics.
Focuses on "real data" tests where available.
"""

import subprocess
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
LOG_DIR = Path("docs/reports/test_logs")
TEST_SUITE = [
    {
        "name": "01_Topology",
        "path": "tests/functional/test_manifold_runner.py", 
        "type": "script", 
        "desc": "Real Manifold & Metrics"
    },
    {
        "name": "02_Memory",
        "path": "tests/integration/core/memory_storage.py",
        "type": "pytest",
        "desc": "LanceDB Storage"
    },
    {
        "name": "03_Reasoning",
        "path": "tests/functional/test_mycelial_runner.py",
        "type": "script",
        "desc": "Mycelial Network Logic"
    },
    {
        "name": "04_Agents_Executor",
        "path": "tests/unit/core/agents/test_executor.py",
        "type": "pytest",
        "desc": "Action Execution Logic"
    },
    {
        "name": "05_Loop_Cycle",
        "path": "tests/integration/core/loop/test_cycle.py",
        "type": "pytest",
        "desc": "Full Cycle Autonomy"
    }
]

def setup_logs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    return run_dir

def run_test(test, run_dir):
    print(f"🔄 Running {test['name']}: {test['desc']}...")
    
    log_file = run_dir / f"{test['name']}.log"
    start_time = time.time()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() # Ensure root is in path
    
    venv_python = "./venv/bin/python"
    
    cmd = []
    if test["type"] == "script":
        cmd = [venv_python, test["path"]]
    else:
        # Run pytest with -s to capture output in log, -v for verbosity
        cmd = [venv_python, "-m", "pytest", test["path"], "-v", "-s"]
        
    try:
        with open(log_file, "w") as f:
            # Write header
            f.write(f"=== TEST RUN: {test['name']} ===\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("="*40 + "\n\n")
            f.flush()
            
            # Execute
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
        duration = time.time() - start_time
        success = result.returncode == 0
        status = "✅ PASS" if success else "❌ FAIL"
        
        print(f"   {status} ({duration:.2f}s) -> Log: {log_file}")
        return {
            "name": test["name"],
            "success": success,
            "duration": duration,
            "log": str(log_file)
        }
        
    except Exception as e:
        print(f"   ❌ ERROR executing {test['name']}: {e}")
        return {
            "name": test["name"],
            "success": False,
            "duration": 0,
            "log": "ERROR"
        }

def main():
    print(f"🚀 Alexandria Sequential Test Runner")
    print(f"📂 Log Directory: {LOG_DIR}")
    
    run_dir = setup_logs()
    results = []
    
    total_start = time.time()
    
    for test in TEST_SUITE:
        res = run_test(test, run_dir)
        results.append(res)
        
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "="*50)
    print(f"📊 EXECUTION SUMMARY ({total_duration:.2f}s)")
    print("="*50)
    
    passed = 0
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['name']:<25} {r['duration']:>6.2f}s")
        if r["success"]: passed += 1
        
    print("-" * 50)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}")
    print(f"Logs saved in: {run_dir}")
    
    if passed < len(results):
        sys.exit(1)

if __name__ == "__main__":
    main()
