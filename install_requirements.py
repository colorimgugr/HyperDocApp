import subprocess
import sys
import os

# => List of required Python packages for your project
REQUIRED_PACKAGES = [
    "PyQt5",
    "numpy",
    "scipy",
    "h5py",
    "matplotlib",
    "pandas",
    "scikit-learn",
    "Pillow",
    "spectral",
    "opencv-python",
]

def pip_install(package):
    print(f"🔧 Trying to install {package} (wheel only)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "--only-binary=:all:"
        ])
        print(f" [ :-) ] {package} installed from precompiled wheel.")
    except subprocess.CalledProcessError:
        print(f"[ !!!] Failed to install {package} as a precompiled package.")
        print(" WARNING : To keep the installation simple and safe, this setup does not allow compiling from source.")
        print("You may need to update Python or install Visual C++ Build Tools if you really want to compile:")
        raise
def install_packages():
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
            print(f"[ :-) ] {package} is already installed.")
        except ImportError:
            pip_install(package)

def install_matlab_engine():
    base = r"C:\Program Files\MATLAB"
    if not os.path.isdir(base):
        print("[ !!! ] MATLAB not found in C:\\Program Files\\MATLAB")
        return

    versions = sorted([v for v in os.listdir(base) if v.startswith("R")], reverse=True)

    for version in versions:
        eng_path = os.path.join(base, version, "extern", "engines", "python")
        setup_py = os.path.join(eng_path, "setup.py")

        if os.path.exists(setup_py):
            print(f" -> Installing MATLAB Engine from {setup_py}...")

            dist_folder = os.path.join(eng_path, "dist")
            try:
                os.makedirs(dist_folder, exist_ok=True)
                print(f" Ensured dist folder exists: {dist_folder}")
            except Exception as e:
                print(f" [!!!] Failed to create dist folder: {e}")
                return

            try:
                subprocess.check_call(
                    [sys.executable, "setup.py", "install"],
                    cwd=eng_path  # Execute from MATLAB's folder
                )
                print(" [ :-) ] MATLAB Engine installed successfully.")
            except subprocess.CalledProcessError as e:
                print("[ !!!!!!!!! ] Failed to install MATLAB Engine.")
                print(e)
            return

    print(" [ !!!!!!!!! ] No valid MATLAB version found with a Python engine.")

if __name__ == "__main__":
    print("\n => Installing required Python packages...")
    install_packages()

    print("\n => Attempting to install MATLAB Engine (optional)...")
    install_matlab_engine()

    print("\n Installation complete.")
