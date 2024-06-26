//
//  ContentView.swift
//  BriaAIDemo
//
//  Created by Ivan Chan on 24/06/2024.
//

import CoreML
import SwiftUI


struct ContentView: View {
    @State private var whisperCppText: String = "Loading..."

    @State private var outputImage: CGImage? = nil
    @State private var inferenceTimeMs: Double? = nil
    @State private var loadTimeMs: Double? = nil
    @State private var computeUnit: MLComputeUnits? = nil
    @State private var loading: Bool = false
    
    @State private var aneClicked: Bool = false
    @State private var gpuClicked: Bool = false
    @State private var cpuClicked: Bool = false
    
    @State private var imageModel = ImageModel()
    
    
    
    let images = [
        UIImage(contentsOfFile: Bundle.main.path(forResource: "example_car", ofType: "jpg")!)!,
        UIImage(contentsOfFile: Bundle.main.path(forResource: "example_input", ofType: "jpg")!)!,
        UIImage(contentsOfFile: Bundle.main.path(forResource: "example_tree", ofType: "jpg")!)!,
    ]
    @State private var index = 0

    
    var body: some View {
        VStack {
            Text("https://www.runlocal.ai")
            if outputImage == nil {
                Image(uiImage: images[index])
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 400, height: 400)
            } else {
                Image(uiImage: UIImage(cgImage: outputImage!))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 400, height: 400)
            }
            
            HStack(spacing: 10) {
                Button("Run with ANE") {
                    outputImage = nil
                    loadTimeMs = nil
                    inferenceTimeMs = nil
                    loading = true
                    computeUnit = MLComputeUnits.all
                    cpuClicked = false
                    aneClicked = true
                    gpuClicked = false
                    Task {
                        let output = await imageModel.loadModelAndPredictImage(image: images[index], computeUnit: computeUnit!)
                        DispatchQueue.main.async {
                            self.outputImage = output.0?.cgImage
                            self.loadTimeMs = output.1
                            self.inferenceTimeMs = output.2
                            self.loading = false
                        }
                    }
                }.fontWeight(aneClicked ? .bold : .regular).tint(aneClicked ? .red : .blue)
                
                Button("Run with GPU") {
                    outputImage = nil
                    loadTimeMs = nil
                    inferenceTimeMs = nil
                    loading = true
                    computeUnit = MLComputeUnits.cpuAndGPU
                    cpuClicked = false
                    aneClicked = false
                    gpuClicked = true
                    Task {
                        let output = await imageModel.loadModelAndPredictImage(image: images[index], computeUnit: computeUnit!)
                        DispatchQueue.main.async {
                            self.outputImage = output.0?.cgImage
                            self.loadTimeMs = output.1
                            self.inferenceTimeMs = output.2
                            self.loading = false
                        }
                    }
                }.fontWeight(gpuClicked ? .bold : .regular).tint(gpuClicked ? .red : .blue)
                
                Button("Run with CPU") {
                    outputImage = nil
                    loadTimeMs = nil
                    inferenceTimeMs = nil
                    loading = true
                    computeUnit = MLComputeUnits.cpuOnly
                    cpuClicked = true
                    aneClicked = false
                    gpuClicked = false
                    Task {
                        let output = await imageModel.loadModelAndPredictImage(image: images[index], computeUnit: computeUnit!)
                        DispatchQueue.main.async {
                            self.outputImage = output.0?.cgImage
                            self.loadTimeMs = output.1
                            self.inferenceTimeMs = output.2
                            self.loading = false
                        }
                    }
                }.fontWeight(cpuClicked ? .bold : .regular).tint(cpuClicked ? .red : .blue)
                
            }
            
            Text("Model: briaai/RMBG-1.4")
                .padding()
            
            if outputImage != nil {
                Text("Inference Time (ms): \(String(format: "%.0f", inferenceTimeMs!))").bold()
                Text("Load Time (ms): \(String(format: "%.0f", loadTimeMs!))").bold()
                    .padding(.bottom)
            }
            
            if loading {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .scaleEffect(2) // Optional: scale the spinner
                    .padding()
            }
            
            Button("Reset Image") {
                outputImage = nil
                loadTimeMs = nil
                inferenceTimeMs = nil
            }
            .padding(.bottom)
            
            Button("Next Image") {
                index = (index + 1) % images.count
                outputImage = nil
                loadTimeMs = nil
                inferenceTimeMs = nil
                
            }
    
        }
        
        .padding()
    }
}

extension ContentView {
    func completionHandler(image: UIImage) {
        print("completion Handler received")
    }
}
#Preview {
    ContentView()
}
