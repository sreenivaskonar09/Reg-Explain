#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for CCAR Stress Testing Framework
Tests all endpoints for PPNR forecasting, model training, and SHAP explanations
"""
import requests
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

class CCARAPITester:
    def __init__(self, base_url="https://cet1-tracker.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def log_test(self, name: str, success: bool, response_data: Any = None, error: str = None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            
        result = {
            'test_name': name,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'response_data': response_data,
            'error': error
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if error:
            print(f"    Error: {error}")
        if response_data and isinstance(response_data, dict):
            print(f"    Response keys: {list(response_data.keys())}")

    def run_test(self, name: str, method: str, endpoint: str, 
                 expected_status: int = 200, data: Optional[Dict] = None,
                 params: Optional[Dict] = None, timeout: int = 30) -> tuple:
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, params=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            
            try:
                response_data = response.json() if response.content else {}
            except:
                response_data = {"raw_response": response.text[:500]}
                
            if not success:
                error = f"Expected {expected_status}, got {response.status_code}. Response: {response.text[:200]}"
            else:
                error = None
                
            self.log_test(name, success, response_data, error)
            return success, response_data

        except Exception as e:
            error = f"Request failed: {str(e)}"
            self.log_test(name, False, None, error)
            return False, {}

    def test_health_check(self):
        """Test API health endpoint"""
        return self.run_test(
            "Health Check",
            "GET",
            "/api/health"
        )

    def test_scenarios_endpoint(self):
        """Test scenarios endpoint"""
        return self.run_test(
            "Get Scenarios",
            "GET", 
            "/api/scenarios"
        )

    def test_macro_current(self):
        """Test current macro conditions"""
        return self.run_test(
            "Get Current Macro Conditions",
            "GET",
            "/api/macro/current"
        )

    def test_generate_synthetic_banks(self):
        """Test synthetic bank generation"""
        return self.run_test(
            "Generate Synthetic Banks",
            "POST",
            "/api/banks/synthetic",
            params={"n_banks": 5, "n_quarters": 12}
        )

    def test_get_banks(self):
        """Test get banks endpoint"""
        return self.run_test(
            "Get Banks Data",
            "GET",
            "/api/banks"
        )

    def test_model_status(self):
        """Test model status endpoint"""
        return self.run_test(
            "Get Model Status",
            "GET",
            "/api/model/status"
        )

    def test_train_models(self):
        """Test model training endpoint"""
        training_config = {
            "lstm_units": 32,
            "lstm_dropout": 0.2,
            "lstm_epochs": 20,  # Reduced for testing
            "xgb_estimators": 50,  # Reduced for testing
            "xgb_depth": 4,
            "xgb_learning_rate": 0.1
        }
        
        return self.run_test(
            "Train Models",
            "POST",
            "/api/model/train",
            data=training_config,
            timeout=120  # Training takes longer
        )

    def test_run_forecast(self):
        """Test forecast execution"""
        forecast_request = {
            "scenario_type": "baseline",
            "n_quarters": 9
        }
        
        return self.run_test(
            "Run Forecast - Baseline",
            "POST",
            "/api/forecast/run",
            data=forecast_request,
            timeout=60
        )

    def test_run_forecast_adverse(self):
        """Test adverse scenario forecast"""
        forecast_request = {
            "scenario_type": "severely_adverse",
            "n_quarters": 9
        }
        
        return self.run_test(
            "Run Forecast - Severely Adverse",
            "POST",
            "/api/forecast/run",
            data=forecast_request,
            timeout=60
        )

    def test_forecast_results(self):
        """Test forecast results endpoint"""
        return self.run_test(
            "Get Forecast Results",
            "GET",
            "/api/forecast/results"
        )

    def test_shap_global_explanation(self):
        """Test SHAP global explanation"""
        return self.run_test(
            "SHAP Global Explanation",
            "POST",
            "/api/explain/global",
            params={"scenario_type": "severely_adverse"},
            timeout=45
        )

    def test_ai_explanation(self):
        """Test AI-powered explanation"""
        return self.run_test(
            "AI Explanation (Gemini)",
            "POST",
            "/api/explain/ai",
            params={"scenario_type": "severely_adverse"},
            timeout=60
        )

    def run_comprehensive_test_suite(self):
        """Run all tests in logical order"""
        print("🚀 Starting CCAR Stress Testing API Test Suite")
        print(f"📡 Testing against: {self.base_url}")
        print("=" * 60)

        # 1. Basic connectivity
        print("\n📋 Phase 1: Basic Connectivity")
        self.test_health_check()
        
        # 2. Data endpoints
        print("\n📊 Phase 2: Data Endpoints")
        self.test_scenarios_endpoint()
        self.test_macro_current()
        
        # 3. Bank data generation
        print("\n🏦 Phase 3: Bank Data Generation")
        self.test_generate_synthetic_banks()
        time.sleep(2)  # Allow data to be stored
        self.test_get_banks()
        
        # 4. Model training and status
        print("\n🧠 Phase 4: Model Training")
        self.test_model_status()
        
        # Train models (this may take time)
        print("⏳ Training models (this may take 1-2 minutes)...")
        train_success, _ = self.test_train_models()
        
        if train_success:
            time.sleep(3)  # Allow models to be saved
            self.test_model_status()  # Check status after training
        
        # 5. Forecasting
        print("\n📈 Phase 5: Capital Forecasting")
        self.test_run_forecast()
        time.sleep(2)
        self.test_run_forecast_adverse()
        time.sleep(2)
        self.test_forecast_results()
        
        # 6. Explainability (only if models are trained)
        if train_success:
            print("\n🔍 Phase 6: Model Explainability")
            self.test_shap_global_explanation()
            time.sleep(2)
            
            # Test AI explanation (may fail if Gemini API issues)
            print("🤖 Testing AI explanation (may timeout if API issues)...")
            self.test_ai_explanation()

        # Print summary
        self.print_summary()
        return self.tests_passed == self.tests_run

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/max(self.tests_run,1)*100):.1f}%")
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r['success']]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  - {test['test_name']}: {test['error']}")
        
        # Show critical issues
        critical_failures = [
            r for r in failed_tests 
            if any(keyword in r['test_name'].lower() 
                   for keyword in ['health', 'scenarios', 'banks'])
        ]
        
        if critical_failures:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_failures)}):")
            for test in critical_failures:
                print(f"  - {test['test_name']}")
                
        print("\n" + "=" * 60)

    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/app/test_reports/backend_test_{timestamp}.json"
            
        results = {
            'test_summary': {
                'total_tests': self.tests_run,
                'passed_tests': self.tests_passed,
                'failed_tests': self.tests_run - self.tests_passed,
                'success_rate': (self.tests_passed/max(self.tests_run,1)*100),
                'test_timestamp': datetime.now().isoformat(),
                'base_url': self.base_url
            },
            'detailed_results': self.test_results
        }
        
        try:
            import os
            os.makedirs('/app/test_reports', exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Test results saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")


def main():
    """Main test execution"""
    # Use the public endpoint from environment
    api_url = "https://cet1-tracker.preview.emergentagent.com"
    
    print("🏛️ CCAR Stress Testing Framework - Backend API Tests")
    print(f"🌐 Target URL: {api_url}")
    
    tester = CCARAPITester(api_url)
    
    try:
        success = tester.run_comprehensive_test_suite()
        tester.save_results()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        tester.print_summary()
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        tester.print_summary()
        return 1


if __name__ == "__main__":
    sys.exit(main())