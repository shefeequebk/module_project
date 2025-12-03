import subprocess
import os
import sys
import pypandoc
# pypandoc.download_pandoc()

# -----------------------------
# HARD-CODED PATHS – EDIT THESE
# -----------------------------
LATEX_FILE = r"D:\CVAI\1\CS6461\module_project\test_scripts\report.tex"
OUTPUT_DOCX = r"D:\CVAI\1\CS6461\module_project\test_scripts\report.docx"

# Optional: if you have a .bib file (set to None if not used)
BIB_FILE = None  # r"/home/pi/project/report/references.bib"

# Optional: extract images into a folder the .docx can embed
EXTRACT_MEDIA = False
MEDIA_DIR = r"/home/pi/project/report/media"

# ---------------------------------
# BUILD EXTRA ARGUMENTS FOR PANDOC
# ---------------------------------
extra_args = [
    "--reference-location=block",
    "--wrap=none"
]

# Extract media (images) into a folder:
if MEDIA_DIR:
    extra_args.append(f"--extract-media={MEDIA_DIR}")

# Add bibliography if provided:
if BIB_FILE:
    extra_args.append(f"--bibliography={BIB_FILE}")

# ---------------------------------
# PERFORM CONVERSION
# ---------------------------------
print("Converting LaTeX → Word (.docx)...")

try:
    pypandoc.convert_file(
        source_file=LATEX_FILE,
        to="docx",
        outputfile=OUTPUT_DOCX,
        extra_args=extra_args
    )

    print("✅ Conversion complete!")
    print("📄 DOCX saved to:", OUTPUT_DOCX)
    if MEDIA_DIR:
        print("🖼 Media extracted to:", MEDIA_DIR)

except Exception as e:
    print("\n❌ Conversion failed:")
    print(e)