## Install Tesseract OCR
* Windows: Since you're on Windows (as indicated by the file paths like D:\BackDataService), you need to install Tesseract manually.
    - Download the Tesseract installer from the official source: Tesseract at UB Mannheim.
    - Run the installer. By default, it installs to C:\Program Files\Tesseract-OCR (or a similar location).
    - During installation, ensure you check the option to install additional language packs if needed (e.g., for ocr_languages in your code).
## Add Tesseract to Your System PATH
* After installation, you need to add the Tesseract executable to your system's PATH environment variable so Python can find it.
    - Open the Start menu, search for "Environment Variables," and select "Edit the system environment variables."
    - In the System Properties window, click "Environment Variables."
    - In the "System variables" section, find and edit the Path variable.
    - Add a new entry: C:\Program Files\Tesseract-OCR (adjust the path if Tesseract was installed elsewhere).
    - Click OK to save changes and close all dialogs.
* Restart your terminal or IDE to ensure the updated PATH is recognized.