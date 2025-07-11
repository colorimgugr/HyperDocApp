# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['MainWindow.py'],
    pathex=[],
    binaries=[],
    datas=[('interface/icons', 'Hypertool/interface/icons'), ('ground_truth/Materials labels and palette assignation - Materials_labels_palette.csv', 'ground_truth'), ('data_vizualisation/Spatially registered minicubes equivalence.csv', 'data_vizualisation')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'torch', 'matlab'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MainWindow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['interface\\icons\\hyperdoc_logo_transparente.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MainWindow',
)
