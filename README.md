# Hyperdoc App — Hyperspectral Image Processing Application

<img src="/interface/icons/hyperdoc_logo_transparente.ico" width="32" /> **Hyperdoc App** is a modular PyQt5-based application designed for loading, visualizing, registering, segmenting, classifying and unmixing hyperspectral cubes (VNIR, SWIR, or combined). It provides a unified graphical interface to handle all stages of hyperspectral data analysis with or without the Hyperdoc database.

> **Open science notice**  
> This project is source-available for academic, research, and educational use only.  
> Commercial use is strictly prohibited.

---

## 📚 Table of Contents
[Launching the Application and Overview](#launching)  
[1. Minicube Extract Tool](#-1-minicube-extract-tool)  
[2. Data Visualization Tool](#-2-data-visualization-tool)  
[3. Registration Tool](#-3-registration-tool)  
[4. Ground Truth Tool](#-4-ground-truth-tool)  
[5. Metadata Tool](#-5-metadata-tool)  
[6. Identification Tool](#-6-identification-tool)  
[7. Unmixing Tool](#-7-unmixing-tool)  
[8. Illumination Tool](#-8-illumination-tool)  
[9. HDF5 / File Browser](#-9-hdf5--file-browser)  
[10. White Calibration Window](#-10-white-calibration-window)  
[Authors & Credits](#-authors--credits)  
[License](#-license)


<a id="launching"></a>
## 🧑‍💻 Launching the Application and Overview

**(Add link to video)**

The simplest way for windows : 
- Dowload the .zip file at this adress : **(Add link to formulario de descarga)**
- Extract it on your disk
- Lauch the .exe file


```bash
# From your project root
git clone https://github.com/llanick/Hypertool.git
cd Hypertool
python install_requirements.py
python MainWindow.py
```

Upon launching the app (`python MainWindow.py` or via the packaged executable):

- A **toolbar** provides quick access to all tools.
- Each tool is shown in a **dockable panel**, which can be undocked, resized, or made fullscreen (⛶ button).
- Ctrl+scroll or drag to zoom/pan viewers

Most tools in Hyperdoc provide **detailed tooltips** describing parameters, algorithms, and expected behavior.  
Tooltips can be **enabled or disabled globally** using the **“Show tooltips” button in the main toolbar**.
This option is especially useful when discovering a new tool or unfamiliar parameters, and can be turned off at any time for a cleaner interface.

⚠️ **Display Resolution Notice**

For screen resolutions **below 1920×1080**, some interface elements or text may appear slightly compressed
or partially truncated in some elements.  
For the best user experience, a **Full HD (1920×1080) display or higher** is recommended.


### Main Tools
| Tool | Description |
|------|--------------|
| <img src="/interface/icons/minicube_icon.svg" width="32" /> **Minicube Extract** | Select and extract subcubes from a larger hyperspectral cube. |
| <img src="/interface/icons/icon_data_viz.svg" width="32" /> **Data Visualization** | Display RGB composites, spectra, and Ground Truth overlays. |
| <img src="/interface/icons/registration_icon.svg" width="32" /> **Registration** | Align VNIR and SWIR cubes spatially. |
| <img src="/interface/icons/GT_icon_1.png" width="32" /> **Ground Truth** | Create and manage labeled maps for supervised learning. |
| <img src="/interface/icons/metadata_icon.svg" width="26" /> **Metadata** | View and edit metadata associated with hyperspectral cubes. |
| <img src="/interface/icons/Ident_icon.png" width="32" /> **Identification** | Classify pixels using ML models (KNN, SVM, CNN, etc.). |
| <img src="/interface/icons/unmixing_icon.png" width="32" /> **Unmixing** | Get relative abundance of each selected endmembers at each pixel of your hypercube. |
| <img src="/interface/icons/illumination_icon.png" width="32" /> **Illumination** | Visualize reflectance under different light sources. |

All modules interact through the shared **HypercubeManager**, ensuring synchronization of loaded cubes and metadata updates.


### Supported Hypercube File Formats (MATLAB / HDF5 / ENVI)

Hyperdoc can load hyperspectral cubes from **.h5 / .hdf5**, **.mat**, and **ENVI** files:

- **HDF5 (.h5 / .hdf5)**  
  Default Hyperdoc/HDF5 layout uses:
  - Dataset: **`DataCube`** stored as *(bands, lines, samples)* and internally displayed as *(height, width, bands)*. In this convention, **lines** correspond to the image height (rows, Y axis) and **samples** correspond to the image width (columns, X axis).
  - Metadata stored primarily in **HDF5 attributes** (file-level attributes).
  - Wavelengths are stored as an attribute **`wl`** (when available and consistent with the cube band count).
  - If some metadata fields cannot be stored as attributes, they may be written under **`meta/`** as datasets.  
  If the file structure is non-standard, Hyperdoc opens the **HDF5 / File Browser** so you can manually select the data, wavelength, and metadata paths.

- **MATLAB (.mat)**  
 Hyperdoc can load `.mat` files containing standard numeric arrays with the following names:
  - **`DataCube`**  
  Hyperspectral cube, typically stored as `(bands, lines, samples)`
  - **`wl`**  
  Wavelength vector (nm)
  - **`metadata`** *(optional)*  
  MATLAB struct containing acquisition or experimental metadata

Files saved as serialized MATLAB **`hypercube` objects** (Hyperspectral Imaging Toolbox)
may not expose these variables directly and therefore **may not be readable**.
For maximum compatibility, users are encouraged to export `DataCube`, `wl`,
and `metadata` as standard MATLAB variables before saving.

- **ENVI (.hdr + binary data)**  
  ENVI cubes are supported in the standard form: a **`.hdr` header** file associated with a binary file (`.img`, `.dat`, etc.).  
  Wavelengths are read from the ENVI metadata fields **`wavelength`** (or **`wl`** if present). Saving in ENVI format produces a standard ENVI header and its associated binary data.

---

## <img src="/interface/icons/minicube_icon.svg" width="32" /> 1. Minicube Extract Tool

**Purpose:** Select and extract smaller subcubes (regions of interest) from a full hyperspectral cube.

### Steps
1. Load a cube.
2. Adjust RGB or grayscale visualization.
3. Draw a rectangular region with the **right click** of the mouse.
4. Click **Save Minicube**.
5. Edit metadata before saving if needed.

---

## <img src="/interface/icons/icon_data_viz.svg" width="32" /> 2. Data Visualization Tool

**Purpose:** Display hyperspectral cubes and explore their spectral or spatial properties.

### Key Functions
- Load VNIR/SWIR cubes (auto-detect pairs).
- Display RGB composites or grayscale.
- Overlay Ground Truth (GT) masks if exists.
- Show spectra (mean and standard deviation) interactively.
- Visualize metadata.
---

## <img src="/interface/icons/registration_icon.svg" width="32" /> 3. Registration Tool

**Purpose:** Align two hyperspectral cubes (e.g., VNIR and SWIR) spatially.

### Steps
1. Load a **Fixed Cube** and a **Moving Cube**.
2. Choose the registration method: ORB or AKAZE.
3. Optionally crop regions for feature matching (only for images that have local big differences).
4. Detect features and compute transformation.
5. Inspect overlays (color blend or checkerboard).
6. If needed : modify/suppres/select features to keep and launch re-register. 
7. Save the aligned cube (or both cubes) for future processing.

### Tips
- Enable *Auto-load complementary cube* to automatically find the VNIR/SWIR pair if exist (_VNIR of _SWIR suffix in filename is needed).
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

---

 ## <img src="/interface/icons/metadata_icon.svg" width="26" /> 5. Metadata Tool

**Purpose:** Inspect, edit, and generate metadata for a loaded cube.

### Key Functions
- Enable **Edit Mode** to modify values.
- Validate changes with **Validate** or **Validate All**.
- Generate new metadata or copy from another cube.
- Add or remove fields dynamically.

---


## <img src="/interface/icons/Ident_icon.png" width="32" /> 6. Identification Tool

**Purpose:** Perform spectral classification using trained machine learning models.


### Steps
- Load VNIR + SWIR cubes (fused) using the VNIR/SWIR dialog.
- Launch the ink/substrate binarization with the chosen algorithm and parameters.
- Run multiple classifiers (KNN, SVM, LDA, CNN, RDF, etc.) by adding model to the job queue. If spectral range between the loaded cube and our database is different, a training step will be added.
- Display classification maps per model.
- Clean and post-process classification maps.
- Save outputs as HDF5 or PNG with palette.

ℹ️ **Current Scope of the Identification Tool**
In this first public version of HyperdocApp, the **Identification Tool** is configured to classify pixels into
 **three ink classes**, following the taxonomy defined in our reference study
 (*Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy*, 2025).
 https://www.sciencedirect.com/science/article/pii/S1386142525002227

 The implemented classes are:
 - **MGP** — *Pure metallo-gallate inks*, typically iron-gall inks without carbon additives.
 - **CC** — *Carbon-containing inks*, including pure carbon inks and mixtures of carbon with metallo-gallate or sepia.
 - **NCC** — *Non-carbon-containing inks*, such as sepia inks and mixtures of sepia with metallo-gallate.

 This classification scheme reflects the experimental design and conservation-oriented interpretation
 described in the reference work. While the underlying machine-learning framework is generic,
 extension to finer subclasses or additional material categories will be addressed in future releases.

---

## <img src="/interface/icons/unmixing_icon.svg" width="32" /> 7. Unmixing Tool

### Purpose
The **Unmixing Tool** estimates **per-pixel abundance maps** by decomposing each pixel spectrum into a linear combination of **endmember spectra**.  
Endmembers can originate from hyperspectral data (**VNIR / SWIR**), spectral libraries, or **external spectroscopic sources such as FTIR**.

The tool is designed for exploratory analysis, material mapping, and comparison of different unmixing strategies on hyperspectral cubes.

### Key Features
- Unmixing of VNIR, SWIR, or fused VNIR+SWIR hyperspectral cubes and FTIR specra.
- Flexible endmember sources:
  - **Spectral libraries (CSV)**
  - **Manual selection** from image pixels or regions
  - **Automatic extraction** (e.g. ATGP, N-FINDR)
- Support for heterogeneous spectral domains:
  - Automatic wavelength range intersection
  - Interpolation of reference spectra onto cube wavelengths when required
- Multiple unmixing strategies:
  - Least-squares based methods
  - Constrained / iterative solvers with cost function.
- Spectral preprocessing :
  - Savgol filtered spectra
  - First or second spectral derivatives
  - L1 / L2 normalization
- Job queue system:
  - Run and compare multiple unmixing configurations
  - Progress tracking and cancellation
- Interactive visualization of abundance maps
- Save and reload unmixing jobs (`.h5`) for later inspection

### Steps
1. **Load a hyperspectral cube**  
   Open a VNIR y/o a SWIR cube and fused them (if two cubes selected) to use in the tool.

2. **Define endmembers**  
   Choose one or more of the following:
   - Load a spectral library (if reflectance + FTIR spectra, they have to be merge already in a csv file and in nm) (CSV)
   - Select endmembers manually from the image
   - Extract endmembers automatically

3. **Handle spectral domains**  
   When mixing hyperspectral and FTIR spectra:
   - The tool automatically aligns spectra on the **common wavelength range**
   - Reference spectra are interpolated to match cube wavelengths if needed.

4. **Configure unmixing parameters**  
   Select preprocessing, normalization, wavelength handling (with the "Band selection" button in Spectra tab), and the unmixing algorithm.

5. **Add job to queue and run**  
   Launch one or several unmixing jobs and monitor their execution.

6. **Visualize abundance maps**  
   Inspect spatial distributions of estimated abundances directly in the viewer. 

7. **Save or reload results**  
   Export unmixing jobs to `.h5` files or reload previous results for comparison.

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

### Steps
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


## 🧩 Authors & Credits

Developed by **CIMLab / Hyperdoc Project Team**.

---

## 📄 License

This project is released as **source-available software for academic, research, and educational use only**.

Commercial use is strictly prohibited.

See the `LICENSE.txt` file for full license terms.

