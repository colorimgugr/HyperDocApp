# Hyperdoc App — Hyperspectral Image Processing Suite

<img src="/interface/icons/hyperdoc_logo_transparente.ico" width="32" /> **Hyperdoc App** is a modular PyQt5-based application designed for loading, visualizing, registering, segmenting, and classifying hyperspectral cubes (VNIR, SWIR, or combined). It provides a unified graphical interface to handle all stages of hyperspectral data analysis.

---

## 🧭 Overview

Hypertool integrates multiple processing modules ("tools") into a single workspace. Each tool is accessible through dockable panels and shares a common cube and metadata manager.

Supported formats include **MATLAB (.mat)**, **HDF5 (.h5)**, and **ENVI (.hdr)**.

---

## 🖥️ Interface Overview

Upon launching the app (`python MainWindow.py` or via the packaged executable):

- A **toolbar** provides quick access to all tools.
- A **menu bar** lists all dockable tool windows under **Tools**.
- Each tool is shown in a **dockable panel**, which can be undocked, resized, or made fullscreen.

### Main Tools
| Tool | Description |
|------|--------------|
| <img src="/interface/icons/metadata_icon.svg" width="32" /> **Metadata** | View and edit metadata associated with hyperspectral cubes. |
| <img src="/interface/icons/icon_data_viz.svg" width="32" /> **Data Visualization** | Display RGB composites, spectra, and Ground Truth overlays. |
| <img src="/interface/icons/registration_icon.svg" width="32" /> **Registration** | Align VNIR and SWIR cubes spatially. |
| <img src="/interface/icons/GT_icon_1.png" width="32" /> **Ground Truth** | Create and manage labeled maps for supervised learning. |
| <img src="/interface/icons/metadata_icon.svg" width="32" /> **Minicube Extract** | Select and extract subcubes from a larger hyperspectral cube. |
| <img src="/interface/icons/Ident_icon.png" width="32" /> **Identification** | Classify pixels using ML models (KNN, SVM, CNN, etc.). |
| <img src="/interface/icons/illumination_icon.png" width="32" /> **Illumination** | Visualize reflectance under different light sources. |

All modules interact through the shared **HypercubeManager**, ensuring synchronization of loaded cubes and metadata updates.

---

## 🧩 1. Metadata Tool

**Purpose:** Inspect, edit, and generate metadata for a loaded cube.

### Main Actions
- Browse cube metadata via dropdown.
- Enable **Edit Mode** to modify values.
- Validate changes with **Validate** or **Validate All**.
- Generate new metadata or copy from another cube.
- Add or remove fields dynamically.

---

## 🎨 2. Data Visualization Tool

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

## 🔍 3. Registration Tool

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

## 🧪 4. Ground Truth Tool

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
- Use **Live Spectra** mode to preview pixel responses.

---

## 🧊 5. Minicube Extract Tool

**Purpose:** Select and extract smaller subcubes (regions of interest) from a full hyperspectral cube.

### Steps
1. Load a cube.
2. Adjust RGB or grayscale visualization.
3. Draw a rectangular region with the mouse.
4. Click **Save Minicube**.
5. Edit metadata before saving.

---

## 🧠 6. Identification Tool

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

## 💡 7. Illumination Tool

**Purpose:** Convert reflectance cubes to RGB appearance under chosen illuminants.

### Steps
1. Load a reflectance cube (covering 400–780 nm).
2. Choose an illuminant (e.g., D65, A, F2, etc.).
3. Adjust gamma and distance factor.
4. View and compare the resulting color images.
5. Save the computed RGB image.

---

## 🗂️ 8. HDF5 / File Browser

**Purpose:** Inspect unknown HDF5 or MATLAB files and manually select the paths for data, wavelengths, and metadata.

### Usage
1. Open the **File Browser** dock.
2. Explore file tree structure.
3. Assign dataset paths to the cube, wavelength, and metadata.
4. Click **OK** to confirm.

---

## ⚪ 9. White Calibration Window

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
- The **cube list** menu helps quickly switch between active datasets.
- Keyboard shortcuts and sliders allow fast channel tuning.

---

## 🧑‍💻 Launching the Application

```bash
# From your project root
git clone https://github.com/yourusername/Hypertool.git
cd Hypertool
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

Includes code contributions for hyperspectral data handling, image registration, machine learning classification, and illumination modeling.

---

## 📄 License

MIT License © 2025 — CIMLab Hyperdoc Project.

