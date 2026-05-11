#!/usr/bin/env python3
"""
Test Runner for Medical Multimodal Retrieval System

This script runs all tests and generates a coverage report.
"""

import os
import sys
import unittest
import subprocess
from pathlib import Path

def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running unit tests...")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    if result.wasSuccessful():
        print("✅ All unit tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
        return False

def run_integration_tests():
    """Run integration tests"""
    print("\n🔗 Running integration tests...")
    
    try:
        # Test model loading
        sys.path.append('app')
        from models.multimodal_model import MultimodalModel
        
        model = MultimodalModel()
        print("✅ Model loading test passed")
        
        # Test data loading
        from data.data_loader import ChestXrayDataset
        print("✅ Data loading test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_api_tests():
    """Test API endpoints"""
    print("\n🌐 Running API tests...")
    
    try:
        import requests
        import time
        
        # Start API server in background
        import subprocess
        api_process = subprocess.Popen([
            sys.executable, 'app/api/main.py'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get('http://127.0.0.1:8000/health', timeout=5)
        
        if response.status_code == 200:
            print("✅ API health check passed")
            api_process.terminate()
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            api_process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        try:
            api_process.terminate()
        except:
            pass
        return False

def run_coverage():
    """Generate coverage report"""
    print("\n📊 Generating coverage report...")
    
    try:
        # Install coverage if not available
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "coverage"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Run coverage
        subprocess.check_call([
            sys.executable, "-m", "coverage", "run",
            "-m", "unittest", "discover", "tests"
        ])
        
        # Generate report
        subprocess.check_call([
            sys.executable, "-m", "coverage", "report",
            "-m", "--include=app/*"
        ])
        
        # Generate HTML report
        subprocess.check_call([
            sys.executable, "-m", "coverage", "html",
            "--include=app/*", "--directory=htmlcov"
        ])
        
        print("✅ Coverage report generated in htmlcov/")
        return True
        
    except Exception as e:
        print(f"❌ Coverage generation failed: {e}")
        return False

def run_linting():
    """Run code linting"""
    print("\n🔍 Running code linting...")
    
    try:
        # Install flake8 if not available
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "flake8"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Run flake8
        result = subprocess.run([
            sys.executable, "-m", "flake8", 
            "app/", "--max-line-length=100", "--ignore=E501,W503"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Code linting passed")
            return True
        else:
            print("❌ Linting issues found:")
            print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ Linting failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🧪 Medical Multimodal Retrieval System Test Suite")
    print("=" * 60)
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    results = []
    
    # Run different test types
    if "--unit" in sys.argv or "--all" in sys.argv:
        results.append(("Unit Tests", run_unit_tests()))
    
    if "--integration" in sys.argv or "--all" in sys.argv:
        results.append(("Integration Tests", run_integration_tests()))
    
    if "--api" in sys.argv or "--all" in sys.argv:
        results.append(("API Tests", run_api_tests()))
    
    if "--coverage" in sys.argv or "--all" in sys.argv:
        results.append(("Coverage", run_coverage()))
    
    if "--lint" in sys.argv or "--all" in sys.argv:
        results.append(("Linting", run_linting()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
