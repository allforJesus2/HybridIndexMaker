# licensing_system.py
import hashlib
import json
import datetime
from pathlib import Path
import uuid
import base64
from cryptography.fernet import Fernet
import winreg
import requests


class LicenseManager:
    def __init__(self):
        # Use a proper secret key - this should match any encryption used on your server
        self.secret_key = b'YourActualSecretKeyHere1234567890abc='  # Replace with your actual key
        self.trial_days = 30
        self.firebase_verification_enabled = True  # Allow toggling Firebase verification
        
        # Generate the cipher suite from the secret key
        try:
            self.cipher_suite = Fernet(base64.b64encode(hashlib.sha256(self.secret_key).digest()))
        except Exception as e:
            print(f"Error initializing cipher suite: {str(e)}")
            self.cipher_suite = None
        
        self.current_version = "1.0.0"

    def check_license(self):
        """Check if there's a valid license or trial period"""
        # First check for a valid license file
        license_file = Path.home() / ".pidvision_license"
        if license_file.exists():
            license_key = license_file.read_text().strip()
            is_valid, license_data = self.validate_license(license_key)
            if is_valid:
                # Check if license is valid for current version
                if self._is_version_covered(license_data):
                    return True, "Licensed Version"
                return False, "License needs upgrade"

        # If no valid license, check trial status
        is_trial_valid, days_remaining = self.check_trial_status()
        if is_trial_valid:
            return True, f"Trial Version ({days_remaining} days remaining)"

        return False, "Trial Expired"

    def _is_version_covered(self, license_data):
        """Check if the current version is covered by the license"""
        if 'valid_from_version' not in license_data:
            return False

        if 'updates_until' not in license_data:
            return False

        updates_until = datetime.datetime.strptime(license_data['updates_until'], "%Y-%m-%d")
        if datetime.datetime.now() > updates_until:
            return False

        return True

    def _get_hardware_id(self):
        """Generate a unique hardware ID using system information"""
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\SystemInformation") as key:
                system_manufacturer = winreg.QueryValueEx(key, "SystemManufacturer")[0]
                system_product_name = winreg.QueryValueEx(key, "SystemProductName")[0]

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                processor_id = winreg.QueryValueEx(key, "ProcessorNameString")[0]

            hardware_str = f"{system_manufacturer}-{system_product_name}-{processor_id}"
            return hashlib.sha256(hardware_str.encode()).hexdigest()
        except:
            return None

    def check_trial_status(self):
        """Check if trial period is still valid"""
        trial_file = Path.home() / ".pidvision_trial"

        if not trial_file.exists():
            # First time running - start trial
            trial_data = {
                "start_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "hardware_id": self._get_hardware_id()
            }
            trial_file.write_text(json.dumps(trial_data))
            return True, self.trial_days

        # Read existing trial data
        trial_data = json.loads(trial_file.read_text())
        start_date = datetime.datetime.strptime(trial_data["start_date"], "%Y-%m-%d")
        days_elapsed = (datetime.datetime.now() - start_date).days

        # Verify hardware ID hasn't changed (prevent trial reset by moving files)
        if trial_data["hardware_id"] != self._get_hardware_id():
            return False, 0

        if days_elapsed <= self.trial_days:
            return True, self.trial_days - days_elapsed
        return False, 0

    def activate_license(self, license_key):
        """Activate a license key"""
        is_valid, license_data = self.validate_license(license_key)
        if is_valid:
            license_file = Path.home() / ".pidvision_license"
            license_file.write_text(license_key)
            return True, "License activated successfully"
        return False, "Invalid license key"

    def _verify_with_firebase(self, license_key):
        """Verify license key with Firebase"""
        try:
            # The endpoint for callable functions
            url = 'https://us-central1-pidvision-website.cloudfunctions.net/verifyLicense'
            
            # Format the request body according to Firebase Callable Functions spec
            headers = {
                'Content-Type': 'application/json',
                'Origin': 'https://pidvision-website.web.app'  # Add Origin header
            }
            
            # Update the data structure to match the Cloud Function expectation
            data = {
                'data': {
                    'licenseKey': license_key,
                    'hardwareId': self._get_hardware_id()  # Add hardware ID
                }
            }
            
            # Add detailed logging
            print(f"Sending verification request to Firebase...")
            print(f"Request data: {data}")
            
            response = requests.post(url, json=data, headers=headers, timeout=10)  # Add timeout
            
            print(f"Firebase verification response status: {response.status_code}")
            print(f"Firebase verification response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                # Check for error in response
                if 'error' in result:
                    print(f"Firebase returned error: {result['error']}")
                    return False
                    
                # Firebase callable functions wrap the response in a 'result' field
                if 'result' in result and result['result'].get('isValid', False):
                    print("License verified successfully")
                    return True
                
                print("License verification failed - invalid response format")
                return False
                
            print(f"Firebase verification failed with status {response.status_code}")
            
            # If we get a 500 error, fall back to local validation
            if response.status_code == 500:
                print("Falling back to local validation due to server error")
                return None  # Signal to fall back to local validation
                
            return False
            
        except requests.exceptions.Timeout:
            print("Firebase verification timed out - falling back to local validation")
            return None  # Signal to fall back to local validation
            
        except requests.exceptions.ConnectionError:
            print("Firebase connection error - falling back to local validation")
            return None  # Signal to fall back to local validation
            
        except Exception as e:
            print(f"Firebase verification error: {str(e)}")
            return None  # Signal to fall back to local validation

    def validate_license(self, license_key):
        """Validate a license key"""
        if not license_key:
            return False, None
        
        # First try Firebase verification (online)
        if self.firebase_verification_enabled:
            try:
                print(f"Attempting Firebase verification for key: {license_key}")
                firebase_result = self._verify_with_firebase(license_key)
                
                # If firebase_result is None, fall back to local validation
                if firebase_result is None:
                    print("Firebase verification unavailable - falling back to local validation")
                elif firebase_result:
                    print("Firebase verification successful")
                    return True, {"license_type": "verified_by_firebase"}
                else:
                    print("Firebase verification explicitly failed")
                    # Don't fall back to local validation if Firebase explicitly fails
                    return False, None
                    
            except Exception as e:
                print(f"Firebase validation error, falling back to local validation: {str(e)}")
                # Fall back to local validation if Firebase is unreachable
                pass
        
        # Local validation (offline fallback)
        try:
            # Verify the key format (32 character hex string)
            if len(license_key) == 32 and all(c in '0123456789ABCDEF' for c in license_key.upper()):
                print("Local validation successful")
                return True, {"license_type": "offline_verified"}
            print("Local validation failed - invalid key format")
            return False, None
        except Exception as e:
            print(f"Local license validation error: {str(e)}")
            return False, None

    def test_connection(self):
        """Test connection to Firebase and general internet connectivity"""
        print("Starting connection tests...")
        
        results = []
        
        # Test general internet connectivity first (using Google)
        try:
            print(f"Testing Internet Connection...")
            response = requests.get("https://www.google.com", timeout=5)
            results.append(("Internet Connection", response.status_code == 200))
            print(f"Internet test response: {response.status_code}")
        except Exception as e:
            print(f"Internet connectivity test failed: {str(e)}")
            results.append(("Internet Connection", False))

        # Test Firebase connection
        try:
            print(f"Testing Firebase Host...")
            url = 'https://pidvision-website.web.app'
            response = requests.get(url, timeout=5)
            print(f"Firebase host response: {response.status_code}")
            results.append(("Firebase Host", response.status_code == 200))
        except Exception as e:
            print(f"Firebase host test failed: {str(e)}")
            results.append(("Firebase Host", False))

        # Test Firebase function
        try:
            print(f"Testing Firebase Function...")
            url = 'https://us-central1-pidvision-website.cloudfunctions.net/verifyLicense'
            headers = {
                'Content-Type': 'application/json',
                'Origin': 'https://pidvision-website.web.app'
            }
            # Send a properly formatted test request
            data = {
                'data': {
                    'licenseKey': 'TEST-KEY-000000'  # Only send licenseKey
                }
            }
            response = requests.post(url, json=data, headers=headers, timeout=5)
            print(f"Firebase function response: {response.status_code}")
            print(f"Firebase function response text: {response.text}")
            
            # Consider any response from the server (even an error about invalid license)
            # as a successful connection test as long as we got a response
            is_connected = response.status_code in [200, 400, 403]  # Accept validation errors
            results.append(("Firebase Function", is_connected))
            
        except requests.exceptions.RequestException as e:
            print(f"Firebase function test failed: {str(e)}")
            results.append(("Firebase Function", False))

        # Format detailed results
        details = []
        all_passed = True
        for service, passed in results:
            status = "✓" if passed else "✗"
            details.append(f"{service}: {status}")
            if not passed:
                all_passed = False

        # Print final results
        print(f"Final results: {results}")
        print(f"Detailed results:\n{chr(10).join(details)}")

        return all_passed, "\n".join(details)
