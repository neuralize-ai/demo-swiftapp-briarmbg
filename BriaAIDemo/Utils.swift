import CoreML
import UIKit
import Accelerate


func pixelBufferFromImage(image: UIImage) -> CVPixelBuffer? {
    guard let cgImage = image.cgImage else { return nil }

    let frameSize = CGSize(width: cgImage.width, height: cgImage.height)
    var pixelBuffer: CVPixelBuffer?

    let options: [String: Any] = [
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
    ]
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     Int(frameSize.width),
                                     Int(frameSize.height),
                                     kCVPixelFormatType_32ARGB,
                                     options as CFDictionary,
                                     &pixelBuffer)

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

    CVPixelBufferLockBaseAddress(buffer, [])

    let pixelData = CVPixelBufferGetBaseAddress(buffer)
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: pixelData,
                            width: Int(frameSize.width),
                            height: Int(frameSize.height),
                            bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                            space: rgbColorSpace,
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

    context?.draw(cgImage, in: CGRect(origin: .zero, size: frameSize))

    CVPixelBufferUnlockBaseAddress(buffer, [])

    return buffer
}


func createGrayscaleImage(from array: MLMultiArray) -> CGImage? {
    guard array.dataType == .float32, let shape = array.shape as? [Int], shape.count == 2 else {
        print("The multi-array must be 2D and contain float values.")
        return nil
    }
    
    let height = shape[0]
    let width = shape[1]
    let bytesPerPixel = 1
    let bitsPerComponent = 8
    let bytesPerRow = width * bytesPerPixel
    let bitmapInfo = CGImageAlphaInfo.none.rawValue

    guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerComponent,
                                  bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: bitmapInfo) else {
        print("Failed to create context.")
        return nil
    }

    guard let buffer = context.data else {
        print("Context data could not be accessed.")
        return nil
    }
    
    let pixelBuffer = buffer.bindMemory(to: UInt8.self, capacity: width * height)
    for y in 0..<height {
        for x in 0..<width {
            let value = array[y * width + x].floatValue
            let pixelIndex = y * width + x
            pixelBuffer[pixelIndex] = UInt8(value)
        }
    }

    return context.makeImage()
}

public func predict(computeUnit: MLComputeUnits) async -> (CGImage, Double, Double) {
    let config = MLModelConfiguration()
    config.computeUnits = computeUnit
    
    let loadStart = Date()
    let model = try! bria_rmbg_coreml(configuration: config)
    let loadEnd = Date()
    let loadTimeMs = loadEnd.timeIntervalSince(loadStart) * 1000
    
    
    let imageUrl = Bundle.main.url(forResource: "example_input", withExtension: "jpg")!
    
    
    let imageData = try! Data(contentsOf: imageUrl)
    let image = UIImage(data: imageData)!
    
    let inferenceStart = Date()
    let pixelBuffer = pixelBufferFromImage(image: image)!
    let prediction = try! await model.prediction(input: pixelBuffer)
    let mask = createGrayscaleImage(from: prediction.var_2306)
    let answer = image.cgImage!.masking(mask!)!
    let inferenceEnd = Date()
    let inferenceTimeMs = inferenceEnd.timeIntervalSince(inferenceStart) * 1000
    
    
    return (answer, loadTimeMs, inferenceTimeMs)


}

func serializePixelBufferToFile(pixelBuffer: CVPixelBuffer, fileURL: URL) -> Bool {
    // Lock the pixel buffer base address
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    
    // Get the pixel buffer attributes
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    
    // Get the base address of the pixel buffer
    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        return false
    }
    
    // Calculate the total size of the pixel buffer data
    let dataSize = height * bytesPerRow
    
    // Create a Data object from the pixel buffer's raw data
    let data = Data(bytes: baseAddress, count: dataSize)
    
    // Unlock the pixel buffer base address
    CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
    
    // Write the data to the specified file URL
    do {
        try data.write(to: fileURL)
        return true
    } catch {
        print("Failed to write pixel buffer data to file: \(error)")
        return false
    }
}

// Function to get the Documents directory URL
func getDocumentsDirectory() -> URL {
    let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return paths[0]
}
