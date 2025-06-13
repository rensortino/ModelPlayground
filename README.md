# Minimal iOS TFLite Model Test App

This is a minimal iOS app (UIKit, Swift) for testing TensorFlow Lite models. It allows you to pick an image, preprocess it, run inference using a TFLite model (`exposure_cls.tflite`), and display the result.

## Features
- Pick an image from the photo library
- Preprocess image to 256x256
- Run inference with a TFLite model (binary classification)
- Display the two-class probability result

## Setup Instructions

### 1. Create a New Xcode Project
1. Open **Xcode**.
2. Select **File > New > Project...**
3. Choose **App** under the iOS tab and click **Next**.
4. Enter **Product Name**: `ios_model_test` (or any name you prefer).
5. Set **Interface** to `Storyboard` (default) or `SwiftUI` (if you want to adapt the code), and **Language** to `Swift`.
6. Click **Next** and choose a location to save your project.
7. In the Project Navigator, open `ViewController.swift` and replace its contents with the code provided in this repo.
8. Add your `exposure_cls.tflite` file:
    - Right-click on the project folder in the Project Navigator.
    - Select **Add Files to "ios_model_test"...**
    - Choose your `exposure_cls.tflite` file and make sure **Add to targets** is checked.

### 2. Close Xcode Before Installing Pods
After creating the project and adding the files, close Xcode before proceeding to the next steps.

### 3. Install CocoaPods (if not already installed)
```
sudo gem install cocoapods
```

### 4. Create a Podfile
```
pod init
```
Edit the `Podfile` to include:
```
pod 'TensorFlowLiteSwift'
```

### 5. Install Pods
```
pod install
```

### 6. Open the Workspace
```
open ios_model_test.xcworkspace
```

### 7. Build and Run
- Select a simulator or device
- Build and run the app

## Testing the App

### 1. Run the App
- Open the `.xcworkspace` file in Xcode.
- Select a simulator (e.g., iPhone 14) from the device dropdown in the top toolbar.
- Click the **Run** button (or press `Cmd + R`).

### 2. Test Image Selection
- Once the app is running, tap the **"Pick Image"** button.
- The simulator will open the photo library. If no images are available, you can add test images:
  - Drag and drop an image onto the simulator window, or
  - Use the simulator menu: **Features > Photos > Import...**

### 3. Verify Model Inference
- After selecting an image, the app will:
  - Display the selected image in the image view.
  - Preprocess the image to 256x256.
  - Run the TFLite model (`exposure_cls.tflite`) on the image.
  - Display the two-class probability result in the label below the button.

### 4. Debugging Tips
- If the model fails to load or run, check the console output in Xcode for error messages.
- Ensure the `exposure_cls.tflite` file is correctly added to the app target.
- Verify the model input size (256x256) and preprocessing steps match your model's requirements.

## Testing on a Physical iPhone

### 1. Connect Your iPhone
- Use a USB cable to connect your iPhone to your Mac.
- Trust the computer on your iPhone if prompted.

### 2. Configure Xcode
- Open the `.xcworkspace` file in Xcode.
- In the top toolbar, select your iPhone from the device dropdown.
- If your iPhone is not listed, ensure it is unlocked and trusted on your Mac.

### 3. Sign the App
- In Xcode, select the project in the Project Navigator.
- Under the "Signing & Capabilities" tab, ensure "Automatically manage signing" is checked.
- Select your Apple ID or team from the dropdown.
- If you don't have an Apple ID, you can create one for free.

### 4. Run the App
- Click the **Run** button (or press `Cmd + R`).
- The app will build and install on your iPhone.
- You may need to trust the developer on your iPhone:
  - Go to **Settings > General > Device Management** on your iPhone.
  - Trust the developer certificate.

### 5. Test the App
- Open the app on your iPhone.
- Tap the **"Pick Image"** button to select an image from your photo library.
- Verify that the app preprocesses the image and displays the model's output.

## Usage
1. Tap the "Pick Image" button
2. Select an image from your photo library
3. The app will preprocess the image, run inference, and display the result

## Notes
- The model input size is 256x256
- The output is a two-class probability vector
- Minimal error handling and UI for clarity

---

## Main Files
- `ViewController.swift`: Main logic for image picking, preprocessing, and inference
- `exposure_cls.tflite`: Your TFLite model file

---

For any issues, please open an issue or PR. 