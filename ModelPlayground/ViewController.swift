import UIKit
import TensorFlowLite
import Vision // Import Vision for image processing utilities

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    // MARK: - UI Components
    private let imageView = UIImageView()
    private let resultLabel = UILabel()
    private let pickButton = UIButton(type: .system)

    // MARK: - TFLite Properties
    private var classifierInterpreter: Interpreter?
    private var detectionInterpreter: Interpreter?

    // Input size for the models
    private let classifierInputSize = CGSize(width: 256, height: 256)
    private let detectionInputSize = CGSize(width: 320, height: 320)

    // MARK: - View Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        setupUI()
        loadModels()
    }
    
    func getFirstBytes(from image: UIImage, count: Int) -> [UInt8]? {
        guard let cgImage = image.cgImage,
              let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else {
            return nil
        }

        let numberOfBytes = min(count, CFDataGetLength(data))
        var byteBuffer = [UInt8](repeating: 0, count: numberOfBytes)
        memcpy(&byteBuffer, bytes, numberOfBytes)

        return byteBuffer
    }

    // MARK: - UI Setup
    private func setupUI() {
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleAspectFit
        imageView.layer.borderWidth = 2
        imageView.layer.borderColor = UIColor.systemBlue.cgColor
        view.addSubview(imageView)

        resultLabel.translatesAutoresizingMaskIntoConstraints = false
        resultLabel.textAlignment = .center
        resultLabel.numberOfLines = 0
        view.addSubview(resultLabel)

        pickButton.setTitle("Pick Image", for: .normal)
        pickButton.translatesAutoresizingMaskIntoConstraints = false
        pickButton.addTarget(self, action: #selector(pickImage), for: .touchUpInside)
        view.addSubview(pickButton)

        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            imageView.widthAnchor.constraint(equalToConstant: 256),
            imageView.heightAnchor.constraint(equalToConstant: 256),

            pickButton.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 20),
            pickButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            resultLabel.topAnchor.constraint(equalTo: pickButton.bottomAnchor, constant: 20),
            resultLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            resultLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20)
        ])
    }

    // MARK: - Model Loading
    private func loadModels() {
        // 1. Load Classifier Model
        guard let classifierModelPath = Bundle.main.path(forResource: "exposure_cls", ofType: "tflite") else {
            resultLabel.text = "Classifier model not found."
            return
        }
        // 2. Load Detection Model
        guard let detectionModelPath = Bundle.main.path(forResource: "car_detector", ofType: "tflite") else {
            resultLabel.text = "Detection model not found."
            return
        }

        do {
            classifierInterpreter = try Interpreter(modelPath: classifierModelPath)
            try classifierInterpreter?.allocateTensors()

            detectionInterpreter = try Interpreter(modelPath: detectionModelPath)
            try detectionInterpreter?.allocateTensors()
        } catch {
            resultLabel.text = "Failed to load models. Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Image Picking
    @objc private func pickImage() {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        guard let originalImage = info[.originalImage] as? UIImage else { return }

        // Step 1: Run detection to get the bounding box
        guard let boundingBox = runDetectionModel(on: originalImage) else {
            resultLabel.text = "Could not detect a bounding box."
            // Optionally, run the classifier on the whole image as a fallback
            // runClassifierModel(on: originalImage)
            return
        }

        // Step 2: Crop the original image using the bounding box
        guard let croppedImage = cropImage(originalImage, to: boundingBox) else {
            resultLabel.text = "Failed to crop image."
            return
        }
        
        // Update the UI to show the cropped image
        imageView.image = croppedImage

        // Step 3: Run the classifier on the cropped image
        runClassifierModel(on: croppedImage)
    }

    // MARK: - Detection Model Inference
    private func runDetectionModel(on image: UIImage) -> CGRect? {
        guard let detectionInterpreter = detectionInterpreter else { return nil }
        // Preprocess the image for the detection model
        guard let inputData = preprocess(image: image, size: detectionInputSize) else {
            print("Failed to preprocess for detection.")
            return nil
        }
        
        let floatArray = inputData.withUnsafeBytes { Array(UnsafeBufferPointer<Float32>(start: $0.baseAddress?.assumingMemoryBound(to: Float32.self), count: inputData.count / MemoryLayout<Float32>.size)) }

        do {
            try detectionInterpreter.copy(inputData, toInputAt: 0)
            try detectionInterpreter.invoke()
            
            
            // The output tensor at index 0 contains the detection scores
            let scoreTensor = try detectionInterpreter.output(at: 0)
            let scores = [Float](unsafeData: scoreTensor.data) ?? []

            // Take the index of the max value of score
            guard let maxScoreIndex = scores.firstIndex(of: scores.max() ?? 0) else {
                print("No valid detection scores found.")
                return nil
            }
            
            // The output tensor at index 1 contains the bounding boxes
            let boxTensor = try detectionInterpreter.output(at: 1)
            let boxes = [Float](unsafeData: boxTensor.data) ?? []

            // Take the bounding box corresponding to the max score
            let mainBox = boxes[maxScoreIndex * 4 ..< (maxScoreIndex + 1) * 4]
            
            // IMPORTANT: The order of [ymin, xmin, ymax, xmax] depends on your detection model's output.
            // Adjust the indices if your model provides a different order.
            // These values are typically normalized to [0, 1].
            guard boxes.count >= 4 else { return nil }
            var yMin = CGFloat(mainBox[0])
            var xMin = CGFloat(mainBox[1])
            var yMax = CGFloat(mainBox[2])
            var xMax = CGFloat(mainBox[3])

            // Clip bounding box coordinates to ensure they are within the image bounds
            let imageWidth = CGFloat(image.size.width)
            let imageHeight = CGFloat(image.size.height)
            let xMinClipped = max(0, min(xMin, 1))
            let yMinClipped = max(0, min(yMin, 1))
            let xMaxClipped = max(0, min(xMax, 1))
            let yMaxClipped = max(0, min(yMax, 1))


            xMin = min(xMinClipped, imageWidth)
            yMin = min(yMinClipped, imageHeight)
            xMax = max(xMaxClipped, 0)
            yMax = max(yMaxClipped, 0)

            return CGRect(x: xMin, y: yMin, width: xMax - xMin, height: yMax - yMin)

        } catch {
            print("Detection inference failed: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - Classifier Model Inference
    private func runClassifierModel(on image: UIImage) {
        guard let classifierInterpreter = classifierInterpreter else { return }
        // Preprocess the cropped image for the classifier model
        guard let inputData = preprocess(image: image, size: classifierInputSize) else {
            resultLabel.text = "Failed to preprocess image for classification."
            return
        }

        do {
            try classifierInterpreter.copy(inputData, toInputAt: 0)
            try classifierInterpreter.invoke()
            let outputTensor = try classifierInterpreter.output(at: 0)
            let results = [Float](unsafeData: outputTensor.data) ?? []
            let issueProbability = results[0]
            resultLabel.text = "Predicted issue probability: \(issueProbability)"
        } catch {
            resultLabel.text = "Classification failed: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Image Preprocessing (Unified for both models)
    private func preprocess(image: UIImage, size: CGSize) -> Data? {
        // 1. Resize the image to the specified input size.
        guard let resizedImage = image.resized(to: size) else { return nil }
        guard let cgImage = resizedImage.cgImage else { return nil }

        let width = Int(size.width)
        let height = Int(size.height)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        var data = Data(count: width * height * bytesPerPixel)

        // 2. Create a CGContext and draw the image data into it.
        data.withUnsafeMutableBytes { rawBufferPointer in
            // Safely get the base address of the buffer.
            guard let baseAddress = rawBufferPointer.baseAddress else { return }

            // Create a CGContext with the correct parameters.
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            ) else {
                return
            }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        // 3. Normalize the RGB data and convert it to a Float32 array for the model.
        var floatArray = [Float32](repeating: 0.0, count: width * height * 3)
        for i in 0 ..< (width * height) {
            let r = Float32(data[i * 4 + 0]) / 255.0
            let g = Float32(data[i * 4 + 1]) / 255.0
            let b = Float32(data[i * 4 + 2]) / 255.0
            floatArray[i * 3 + 0] = r
            floatArray[i * 3 + 1] = g
            floatArray[i * 3 + 2] = b
        }

        // 4. Return the float array as Data.
        return floatArray.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    // MARK: - Image Cropping Utility
    private func cropImage(_ image: UIImage, to normalizedRect: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        
        // Denormalize the bounding box coordinates
        let cropRect = CGRect(
            x: normalizedRect.origin.x * imageWidth,
            y: normalizedRect.origin.y * imageHeight,
            width: normalizedRect.size.width * imageWidth,
            height: normalizedRect.size.height * imageHeight
        )
        
        // Perform the crop
        if let croppedCGImage = cgImage.cropping(to: cropRect) {
            return UIImage(cgImage: croppedCGImage)
        }
        
        return nil
    }
}

// MARK: - Helper Extensions
extension Array where Element == Float {
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Float>.stride == 0 else { return nil }
        self = unsafeData.withUnsafeBytes {
            Array(UnsafeBufferPointer<Float>(start: $0.baseAddress?.assumingMemoryBound(to: Float.self), count: unsafeData.count / MemoryLayout<Float>.stride))
        }
    }
}

extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
