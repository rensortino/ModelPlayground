import UIKit
import TensorFlowLite

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    private let imageView = UIImageView()
    private let resultLabel = UILabel()
    private let pickButton = UIButton(type: .system)
    private var interpreter: Interpreter?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        setupUI()
        loadModel()
    }

    private func setupUI() {
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.contentMode = .scaleAspectFit
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

    private func loadModel() {
        guard let modelPath = Bundle.main.path(forResource: "exposure_cls", ofType: "tflite") else {
            resultLabel.text = "Model not found."
            return
        }
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()
        } catch {
            resultLabel.text = "Failed to load model."
        }
    }

    @objc private func pickImage() {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        guard let image = info[.originalImage] as? UIImage else { return }
        imageView.image = image
        runModel(on: image)
    }

    private func runModel(on image: UIImage) {
        guard let interpreter = interpreter else { return }
        guard let inputData = preprocess(image: image) else {
            resultLabel.text = "Failed to preprocess image."
            return
        }
        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            let outputTensor = try interpreter.output(at: 0)
            let results = [Float](unsafeData: outputTensor.data) ?? []
            resultLabel.text = "Result: \(results)"
        } catch {
            resultLabel.text = "Inference failed."
        }
    }

    private func preprocess(image: UIImage) -> Data? {
        // 1. Resize the image to the target dimensions
        let targetSize = CGSize(width: 256, height: 256)
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            print("Error: Could not resize image.")
            return nil
        }
        UIGraphicsEndImageContext()
        guard let cgImage = resizedImage.cgImage else {
            print("Error: Could not get CGImage from resized image.")
            return nil
        }
//        guard let cgImage = resized?.cgImage else { return nil }
        let debugUIImage = UIImage(cgImage: cgImage)

        // Convert to RGB data, normalized to [0,1]
        // 2. Get RGBA data from the image
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4 // Use 4 for RGBA
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        var rgbaData = Data(count: width * height * bytesPerPixel)
        
        // Use `withUnsafeMutableBytes` for safe pointer access
        // This provides a scoped, safe pointer to the underlying buffer of `rgbaData`.
        rgbaData.withUnsafeMutableBytes { rawMutableBufferPointer in
            guard let baseAddress = rawMutableBufferPointer.baseAddress else { return }

            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                print("Error: Unable to create CGContext")
                return
            }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        // 3. Normalize the RGB data to [0,1] Float32 and handle channel order
        // TFLite models often expect data in [R,G,B, R,G,B, ...] format.
        let byteCount = width * height * 3 // We only need 3 channels (R,G,B) for the model
        var floatArray = [Float32](repeating: 0.0, count: byteCount)
        for i in 0..<(width * height) {
            let R = Float32(rgbaData[i * 4 + 0]) / 255.0
            let G = Float32(rgbaData[i * 4 + 1]) / 255.0
            let B = Float32(rgbaData[i * 4 + 2]) / 255.0

            // Store in the float array
            floatArray[i * 3 + 0] = R
            floatArray[i * 3 + 1] = G
            floatArray[i * 3 + 2] = B
        }
        // 4. Convert the float array to Data
        // Create Data by copying bytes to avoid a dangling pointer
        // This initializer creates a new Data object with its own copy of the bytes.
        return floatArray.withUnsafeBufferPointer { buffer in
            return Data(buffer: buffer)
        }

    }
}

// Helper to convert Data to [Float]
extension Array where Element == Float {
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Float>.stride == 0 else { return nil }
        self = unsafeData.withUnsafeBytes {
            Array(UnsafeBufferPointer<Float>(start: $0.baseAddress?.assumingMemoryBound(to: Float.self), count: unsafeData.count / MemoryLayout<Float>.stride))
        }
    }
}
