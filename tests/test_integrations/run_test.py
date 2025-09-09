#!/usr/bin/env python3
"""
Test runner for OpenAI agent tests with trace body validation.

Commands:
- gen_test <file>: Run --mode=gen then --mode=mark_dynamic on a single file
- test <file1> <file2> ...: Run --mode=test on multiple files
"""

import sys
import subprocess
import os
import re
from typing import List, Tuple
import argparse

def run_command(cmd: List[str], file_path: str) -> Tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(file_path)
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def extract_trace_errors(output: str) -> List[str]:
    """Extract trace validation errors from output"""
    errors = []
    
    # Look for trace validation errors in the output
    trace_error_patterns = [
        r'Trace body does not match expected file:.*?\n(.*?)(?=\n\[Confident AI Trace Log\]|\nTo disable|\Z)',
        r'AssertionError: Trace body does not match expected file:.*?\n(.*?)(?=\n\[Confident AI Trace Log\]|\nTo disable|\Z)',
        r'Error flushing remaining trace\(s\): (.*?)(?=\n\[Confident AI Trace Log\]|\nTo disable|\Z)'
    ]
    
    for pattern in trace_error_patterns:
        matches = re.findall(pattern, output, re.DOTALL)
        for match in matches:
            if match.strip():
                errors.append(match.strip())
    
    return errors

def gen_test(file_path: str):
    """Run gen_test command on a single file"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Running gen_test on: {file_path}")
    print("=" * 60)
    
    # Step 1: Run --mode=gen
    print("Step 1: Running --mode=gen")
    cmd1 = [sys.executable, os.path.basename(file_path), "--mode=gen"]
    success1, stdout1, stderr1 = run_command(cmd1, file_path)
    output1 = stdout1 + stderr1
    
    if success1:
        print("✅ --mode=gen: SUCCESS")
    else:
        print("❌ --mode=gen: FAILED")
        print("Output:")
        print(output1)
        return
    
    # Step 2: Run --mode=mark_dynamic
    print("\nStep 2: Running --mode=mark_dynamic")
    cmd2 = [sys.executable, os.path.basename(file_path), "--mode=mark_dynamic"]
    success2, stdout2, stderr2 = run_command(cmd2, file_path)
    output2 = stdout2 + stderr2
    
    if success2:
        print("✅ --mode=mark_dynamic: SUCCESS")
    else:
        print("❌ --mode=mark_dynamic: FAILED")
        print("Output:")
        print(output2)
        return
    
    print(f"\n�� gen_test completed successfully for: {file_path}")

def test_files(file_paths: List[str]):
    """Run test command on multiple files"""
    if not file_paths:
        print("Error: No files provided for testing")
        return
    
    print(f"Running test on {len(file_paths)} file(s)")
    print("=" * 60)
    
    total = len(file_paths)
    passed = 0
    failed = 0
    failures = []
    
    for i, file_path in enumerate(file_paths, 1):
        if not os.path.exists(file_path):
            print(f"❌ [{i}/{total}] {file_path}: FILE NOT FOUND")
            failed += 1
            failures.append((file_path, "File not found"))
            continue
        
        print(f"[{i}/{total}] Testing: {file_path}")
        cmd = [sys.executable, os.path.basename(file_path), "--mode=test"]
        success, stdout, stderr = run_command(cmd, file_path)
        output = stdout + stderr
        
        # Check for trace validation errors even if the command "succeeded"
        trace_errors = extract_trace_errors(output)
        
        if success and not trace_errors:
            print(f"✅ [{i}/{total}] {file_path}: PASSED")
            passed += 1
        else:
            print(f"❌ [{i}/{total}] {file_path}: FAILED")
            failed += 1
            
            # Collect error details
            error_details = []
            if not success:
                error_details.append(f"Command failed with return code (stdout/stderr):\n{output}")
            if trace_errors:
                error_details.append("Trace validation errors:")
                for error in trace_errors:
                    error_details.append(f"  {error}")
            
            failures.append((file_path, "\n".join(error_details)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    
    if failures:
        print("\nFAILURES:")
        for file_path, error in failures:
            print(f"\n❌ {file_path}:")
            print("-" * 40)
            print(error)
    
    if failed > 0:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Test runner for OpenAI agent tests with trace body validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py gen_test test_component_eval.py
  python run_tests.py test test_component_eval.py customer_service_agent.py
  python run_tests.py test tests/*.py
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # gen_test command
    gen_parser = subparsers.add_parser('gen_test', help='Run gen then mark_dynamic on a single file')
    gen_parser.add_argument('file', help='Python file to process')
    
    # test command
    test_parser = subparsers.add_parser('test', help='Run test mode on multiple files')
    test_parser.add_argument('files', nargs='+', help='Python files to test')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'gen_test':
        gen_test(args.file)
    elif args.command == 'test':
        test_files(args.files)

if __name__ == "__main__":
    main()