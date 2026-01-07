# Hyperdoc App — Hyperspectral Image Processing Suite

<img src="/interface/icons/hyperdoc_logo_transparente.ico" width="32" /> **Hyperdoc App** is a modular PyQt5-based application designed for loading, visualizing, registering, segmenting, and classifying hyperspectral cubes (VNIR, SWIR, or combined). It provides a unified graphical interface to handle all stages of hyperspectral data analysis.

---

## 📚 Table of Contents
[Overview](#-overview)  
[Key Features](#-key-features)  
[1. Metadata Tool](#-1-metadata-tool)  
[2. Data Visualization Tool](#-2-data-visualization-tool)  
[3. Registration Tool](#-3-registration-tool)  
[4. Ground Truth Tool](#-4-ground-truth-tool)  
[5. Minicube Extract Tool](#-5-minicube-extract-tool)  
[6. Identification Tool](#-6-identification-tool)  
[7. Unmixing Tool](#-7-unmixing-tool)  
[8. Illumination Tool](#-7-illumination-tool)  
[9. HDF5 / File Browser](#-8-hdf5--file-browser)  
[10. White Calibration Window](#-9-white-calibration-window)  
[Cube Management and Synchronization](#-cube-management-and-synchronization)  
[Tips & Tricks](#-tips--tricks)  
[Launching the Application](#-launching-the-application)
[Authors & Credits](#-authors--credits)  
[License](#-license)

## 🧭 Overview

Hypertool integrates multiple processing modules ("tools") into a single workspace. Each tool is accessible through dockable panels and shares a common cube and metadata manager.

Supported formats include **MATLAB (.mat)**, **HDF5 (.h5)**, and **ENVI (.hdr)**.

---

## ✨ Key Features

Load and manage multiple hyperspectral cubes simultaneously

Advanced visualization (RGB, grayscale, spectra, GT overlay)

Precise VNIR–SWIR registration

Ground truth creation & class labelling

Minicube extraction with editable metadata

Endmember extraction & spectral unmixing

Synchronized abundance map gallery

Machine learning classification (KNN, SVM, LDA, CNN, RF…)

Illumination modeling (render RGB under D65, A, F2, etc.)

Full HDF5 explorer to load unknown file structures

White calibration (flat-field and reference panel workflows)

---

## 🖥️ Interface Overview

Upon launching the app (`python MainWindow.py` or via the packaged executable):

- A **toolbar** provides quick access to all tools.
- A **menu bar** lists all dockable tool windows under **Tools**.
- Each tool is shown in a **dockable panel**, which can be undocked, resized, or made fullscreen.

### Main Tools
| Tool | Description |
|------|--------------|
| <img src="/interface/icons/metadata_icon.svg" width="26" /> **Metadata** | View and edit metadata associated with hyperspectral cubes. |
| <img src="/interface/icons/icon_data_viz.svg" width="32" /> **Data Visualization** | Display RGB composites, spectra, and Ground Truth overlays. |
| <img src="/interface/icons/registration_icon.svg" width="32" /> **Registration** | Align VNIR and SWIR cubes spatially. |
| <img src="/interface/icons/GT_icon_1.png" width="32" /> **Ground Truth** | Create and manage labeled maps for supervised learning. |
| <img src="/interface/icons/minicube_icon.svg" width="32" /> **Minicube Extract** | Select and extract subcubes from a larger hyperspectral cube. |
| <img src="/interface/icons/Ident_icon.png" width="32" /> **Identification** | Classify pixels using ML models (KNN, SVM, CNN, etc.). |
| <img src="/interface/icons/illumination_icon.png" width="32" /> **Illumination** | Visualize reflectance under different light sources. |

All modules interact through the shared **HypercubeManager**, ensuring synchronization of loaded cubes and metadata updates.

---

 ## <img src="/interface/icons/metadata_icon.svg" width="26" /> 1. Metadata Tool

**Purpose:** Inspect, edit, and generate metadata for a loaded cube.

### Main Actions
- Browse cube metadata via dropdown.
- Enable **Edit Mode** to modify values.
- Validate changes with **Validate** or **Validate All**.
- Generate new metadata or copy from another cube.
- Add or remove fields dynamically.

---

## <img src="/interface/icons/icon_data_viz.svg" width="32" /> 2. Data Visualization Tool

**Purpose:** Display hyperspectral cubes and explore their spectral or spatial properties.

### Features
- Load VNIR/SWIR cubes (auto-detect pairs).
- Display RGB composites or grayscale.
- Overlay Ground Truth (GT) masks.
- Show spectra (mean and standard deviation) interactively.
- Flip or rotate images for alignment.

### Workflow
1. Click **Open Hypercube** to load a `.mat`, `.h5`, or `.hdr` cube.
2. If both VNIR & SWIR are present, they are auto-paired.
3. Adjust RGB channels via sliders or presets.
4. Load a Ground Truth PNG (`_GT.png`) if available.
5. Modify transparency to compare GT overlay.

---

## <img src="/interface/icons/registration_icon.svg" width="32" /> 3. Registration Tool

**Purpose:** Align two hyperspectral cubes (e.g., VNIR and SWIR) spatially.

### Workflow
1. Load a **Fixed Cube** and a **Moving Cube**.
2. Choose the registration method: ORB, AKAZE, or SIFT.
3. Optionally crop regions for feature matching.
4. Detect features and compute transformation.
5. Inspect overlays (color blend or checkerboard).
6. Save the aligned cube for future processing.

### Tips
- Enable *Auto-load complementary cube* to automatically find the VNIR/SWIR pair.
- The aligned cube can serve as the parent for further minicube extraction.

---

## <img src="/interface/icons/GT_icon_1.png" width="32" /> 4. Ground Truth Tool

**Purpose:** Create or edit pixel-level classification maps used for supervised training.

### Key Functions
- Define number of classes (`nclass_box`).
- Draw or erase regions directly on the image.
- Assign class names and colors.
- Save Ground Truth as PNG or matrix.
- Compute average spectra per class.

### Tips
- Toggle transparency to inspect GT overlay.
- Merge or reassign classes interactively.
- Use **Live Spectra** mode to preview pixel spectrum.

---

## <img src="/interface/icons/minicube_icon.svg" width="32" /> 5. Minicube Extract Tool

**Purpose:** Select and extract smaller subcubes (regions of interest) from a full hyperspectral cube.

### Steps
1. Load a cube.
2. Adjust RGB or grayscale visualization.
3. Draw a rectangular region with the mouse.
4. Click **Save Minicube**.
5. Edit metadata before saving.

---

## <img src="/interface/icons/Ident_icon.png" width="32" /> 6. Identification Tool

**Purpose:** Perform spectral classification using trained machine learning models.

### Features
- Load VNIR + SWIR cubes (fused).
- Run multiple classifiers (KNN, SVM, LDA, CNN, RDF, etc.).
- Display classification maps per model.
- Clean and post-process classification maps.
- Save outputs as HDF5 or PNG with palette.

### Workflow
1. Load cubes using the VNIR/SWIR dialog.
2. Add models to the job queue.
3. Start classification or train new models.
4. Review maps and performance.
5. Save final results and metadata.

---

---

## <img src="/interface/icons/unmixing_icon.svg" width="32" /> 7. Unmixing Tool

### Purpose
The **Unmixing Tool** estimates **per-pixel abundance maps** by decomposing each pixel spectrum into a linear combination of **endmember spectra**.  
It is designed for exploratory analysis, material mapping, and comparison of different unmixing strategies on hyperspectral cubes.

The tool supports **library-based**, **manual**, and **automatic** endmember definitions and provides integrated **job management** and **visualization**.

---

### Key Features
- Unmixing of VNIR, SWIR, or fused VNIR+SWIR hyperspectral cubes
- Multiple endmember sources:
  - **Spectral library (CSV)**
  - **Manual selection** from image pixels or regions
  - **Automatic extraction** (e.g. ATGP, N-FINDR)
- Several unmixing strategies:
  - Least-squares based methods
  - Constrained / iterative solvers
- Spectral preprocessing options:
  - RAW spectra
  - First or second spectral derivatives
  - L1 / L2 normalization
- Job queue system:
  - Run and compare multiple unmixing configurations
  - Progress tracking and cancellation
- Interactive visualization of abundance maps
- Save and reload unmixing jobs (`.h5`) for later inspection

---

### Typical Workflow
1. **Load a hyperspectral cube**  
   Open a VNIR, SWIR, or fused cube from the main application.

2. **Define endmembers**  
   Choose one of the following:
   - Load a spectral library (CSV)
   - Select endmembers manually from the image
   - Extract endmembers automatically

3. **Configure unmixing parameters**  
   Select preprocessing, normalization, wavelength handling, and the unmixing algorithm.

4. **Add job to queue and run**  
   Launch one or several unmixing jobs and monitor their execution.

5. **Visualize abundance maps**  
   Inspect spatial distributions of estimated abundances directly in the viewer.

6. **Save or reload results**  
   Export unmixing jobs to `.h5` files or reload previous results for comparison.

---

### Tips & Notes
- Endmember and cube wavelengths must overlap; only the common spectral range is used.
- Using spectral derivatives can enhance material contrast but may increase noise sensitivity.
- Running multiple jobs with different normalizations or algorithms is recommended for comparison.
- Saved unmixing jobs can be reloaded later without recomputing abundances.
- For large cubes or complex constraints, unmixing may take time; prefer incremental testing with small regions first.

---


## <img src="/interface/icons/illumination_icon.png" width="32" /> 8. Illumination Tool

**Purpose:** Convert reflectance cubes to RGB appearance under chosen illuminants.

### Steps
1. Load a reflectance cube (covering 400–780 nm).
2. Choose an illuminant (e.g., D65, A, F2, etc.).
3. Adjust gamma and distance factor.
4. View and compare the resulting color images.
5. Save the computed RGB image.

---

## <img src="/interface/icons/file_browser_icon.png" width="32" /> 9. HDF5 / File Browser

**Purpose:** Inspect unknown HDF5 or MATLAB files and manually select the paths for data, wavelengths, and metadata.

### Usage
1. Open the **File Browser** dock.
2. Explore file tree structure.
3. Assign dataset paths to the cube, wavelength, and metadata.
4. Click **OK** to confirm.

---

## ⚪ 10. White Calibration Window

**Purpose:** Perform white reference calibration to convert raw data to reflectance.

### Options
- Load white capture image.
- Select calibration mode (mean, horizontal/vertical/full flat field).
- Choose a white reference (Sphere Optics, Teflon, etc.).
- Validate calibration and apply.

---

## 💾 Cube Management and Synchronization

- Every saved cube is tracked by the **Hypercube Manager**.
- Metadata changes are propagated automatically to all tools.
- The toolbar cube list updates dynamically.
- Multiple cubes can be loaded simultaneously and switched via the cube dropdown.

---

## 🧰 Tips & Tricks

- Use the **toolbar icons** for one-click access to tools.
- Docks can be detached, resized, or made fullscreen (⛶ button).
- Ctrl+scroll or drag to zoom/pan viewers
---

## 🧑‍💻 Launching the Application

the simplest way for windows : 
- Dowload the .zip file
- Extract it on your disk
- Lauch the .exe file


```bash
# From your project root
git clone https://github.com/yourusername/Hypertool.git
cd Hypertool
python install_requirements.py
python MainWindow.py
```

To build an executable:
```bash
pyinstaller --noconfirm --noconsole --exclude-module tensorflow --exclude-module torch \
  --icon="interface/icons/hyperdoc_logo_transparente.ico" \
  --add-data "interface/icons:Hypertool/interface/icons" \
  --add-data "hypercubes/white_ref_reflectance_data:hypercubes/white_ref_reflectance_data" \
  --add-data "ground_truth/Materials labels and palette assignation - Materials_labels_palette.csv:ground_truth" \
  --add-data "data_vizualisation/Spatially registered minicubes equivalence.csv:data_vizualisation" \
  MainWindow.py
```

---

## 🧩 Authors & Credits

Developed by **CIMLab / Hyperdoc Project Team**.

---

## 📄 License

MIT License © 2025 — CIMLab Hyperdoc Project.

