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
        self.secret_key = b'YOUR_SECRET_KEY_HERE'  # Needs to be set properly
        self.trial_days = 30
        self.cipher_suite = Fernet(base64.b64encode(hashlib.sha256(self.secret_key).digest()))
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
            response = requests.post(
                'https://pidvision.com/verifyLicense',
                json={'licenseKey': license_key}
            )
            return response.json().get('isValid', False)
        except Exception as e:
            print(f"Firebase verification failed: {str(e)}")
            return False

    def validate_license(self, license_key):
        """Validate a license key"""
        try:
            # Local validation
            decoded_key = base64.urlsafe_b64decode(license_key)
            decrypted_data = self.cipher_suite.decrypt(decoded_key)
            license_data = json.loads(decrypted_data)
            
            # Add Firebase verification
            if not self._verify_with_firebase(license_key):
                return False, None
            
            return True, license_data
        except:
            return False, None

    def generate_license_key(self, user_email, license_type="standard"):
        """Generate a license key for a user"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        license_data = {
            "email": user_email,
            "timestamp": timestamp,
            "license_id": str(uuid.uuid4()),
            "license_type": license_type,
            "valid_from_version": self.current_version,
            "updates_until": (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        }

        encrypted_data = self.cipher_suite.encrypt(json.dumps(license_data).encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
