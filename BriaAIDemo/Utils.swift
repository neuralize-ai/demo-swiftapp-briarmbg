import CoreML
import UIKit
import Accelerate
import Vision


typealias ImageHandler = (UIImage) -> Void

struct ImageModel {
    
    
    var originalWidth: Int? = nil
    var originalHeight: Int? = nil
    var isRunning: Bool = false
    
    var inputImage: UIImage? = nil
    var outputMask: UIImage? = nil
    
    public func resizeOutputMask(cgImage: CGImage) -> CGImage {
        
        let bitsPerComponent = cgImage.bitsPerComponent
        let bytesPerRow = cgImage.bytesPerRow
        let colorSpace = cgImage.colorSpace!
        let bitmapInfo = cgImage.bitmapInfo
        
        let resizeContext = CGContext(data: nil, width: originalWidth!, height: originalHeight!, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)!
        
        resizeContext.interpolationQuality = CGInterpolationQuality.high
        resizeContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: originalWidth!, height: originalHeight!))
        
        let resizedMaskCg = resizeContext.makeImage()!
        return resizedMaskCg
    }


//    private func visionRequestHandler(_ request: VNRequest, error: Error?) {
//        print("handler called")
//        let output =  request.results!.first! as! VNPixelBufferObservation
//        let ciImage = CIImage(cvPixelBuffer: output.pixelBuffer)
//        let context = CIContext()
//        let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
//        let imageResized = resizeOutputMask(cgImage: cgImage)
//        handler(imageResized)
//    }


    public mutating func loadModelAndPredictImage(image: UIImage, computeUnit: MLComputeUnits) async -> UIImage? {
        
        if isRunning {
            return nil
        }
        
        inputImage = image
        isRunning = true
        originalWidth = image.cgImage!.width
        originalHeight = image.cgImage!.height
        
        let config = MLModelConfiguration()
        config.computeUnits = computeUnit
        let model = try! bria_rmbg_coreml_3(configuration: config)
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
            try await handler.perform(requests)
            let output =  requests[0].results!.first! as! VNPixelBufferObservation
            let ciImage = CIImage(cvPixelBuffer: output.pixelBuffer)
            let context = CIContext()
            let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
            let imageResized = resizeOutputMask(cgImage: cgImage)
            let answer = image.cgImage!.masking(imageResized)
            isRunning = false
            return UIImage(cgImage: answer!)
            
        } catch {
            print("Failed to perform text recognition: \(error)")
            return nil
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
