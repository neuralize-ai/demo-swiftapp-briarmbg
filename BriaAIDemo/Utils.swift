import CoreML
import UIKit
import Accelerate
import Vision


typealias ImageHandler = (UIImage) -> Void

struct ImageModel {
    
    var isRunning: Bool = false

    public mutating func loadModelAndPredictImage(image: UIImage, computeUnit: MLComputeUnits) async -> (UIImage?, Double?, Double?) {
        
        if isRunning {
            print("currently running, exiting")
            return (nil, nil, nil)
        }
        
        isRunning = true
        
        let width = image.cgImage!.width
        let height = image.cgImage!.width
        
        let config = MLModelConfiguration()
        config.computeUnits = computeUnit
        
        let loadStart = Date()
        let model = try! bria_rmbg_coreml(configuration: config)
        let loadEnd = Date()
        let loadTimeMs = loadEnd.timeIntervalSince(loadStart) * 1000
        
        guard let visionModel = try? VNCoreMLModel(for: model.model) else {
            fatalError("App failed to create a `VNCoreMLModel` instance.")
        }
        
        // create vision request
        let imageRequest = VNCoreMLRequest(model: visionModel)
        imageRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        
        let handler = VNImageRequestHandler(cgImage: image.cgImage!)
        let requests: [VNRequest] = [imageRequest]

        // Start the image classification request.
        do {
            let inferenceStart = Date()
            try await handler.perform(requests)
            let inferenceEnd = Date()
            let inferenceTimeMs = inferenceEnd.timeIntervalSince(inferenceStart) * 1000
            
            let output =  requests[0].results!.first! as! VNPixelBufferObservation
            let ciImage = CIImage(cvPixelBuffer: output.pixelBuffer)
            let context = CIContext()
            let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
            
            // RESIZING
            let bitsPerComponent = cgImage.bitsPerComponent
            let bytesPerRow = cgImage.bytesPerRow
            let colorSpace = CGColorSpaceCreateDeviceGray()
            let resizeContext = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue).rawValue)!
            resizeContext.interpolationQuality = CGInterpolationQuality.high
            resizeContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            
            
            let resizedMaskCg = resizeContext.makeImage()
            let maskedImage = image.cgImage!.masking(resizedMaskCg!)
            
            isRunning = false
            return (UIImage(cgImage: maskedImage!), loadTimeMs, inferenceTimeMs)
            
        } catch {
            print("Failed to perform inference: \(error)")
            
            isRunning = false
            return (nil, nil, nil)
        }
        
    }
}


//
//public func predict(computeUnit: MLComputeUnits) async -> (CGImage, Double, Double) {
//    let config = MLModelConfiguration()
//    config.computeUnits = computeUnit
//    
//    let loadStart = Date()
//    let model = try! bria_rmbg_coreml(configuration: config)
//    
//    // Create a Vision instance using the image classifier's model instance.
//    
//    let loadEnd = Date()
//    let loadTimeMs = loadEnd.timeIntervalSince(loadStart) * 1000
//    
//    
//    let imageUrl = Bundle.main.url(forResource: "example_input", withExtension: "jpg")!
//    let imageData = try! Data(contentsOf: imageUrl)
//    let image = UIImage(data: imageData)!
//    
//    let inferenceStart = Date()
//    // todo: use VNImageRequestHandler
//    let pixelBuffer = pixelBufferFromImage(image: image)!
//    let prediction = try! await model.prediction(input: pixelBuffer)
//    let mask = createGrayscaleImage(from: prediction.var_2306)
//    let answer = image.cgImage!.masking(mask!)!
//    let inferenceEnd = Date()
//    let inferenceTimeMs = inferenceEnd.timeIntervalSince(inferenceStart) * 1000
//    
//    
//    return (answer, loadTimeMs, inferenceTimeMs)
//}
//
//// Function to get the Documents directory URL
//func getDocumentsDirectory() -> URL {
//    let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
//    return paths[0]
//}
