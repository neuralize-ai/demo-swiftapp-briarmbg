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

    let image = UIImage(contentsOfFile: Bundle.main.path(forResource: "example_input", ofType: "jpg")!)!
    let logo = UIImage(contentsOfFile: Bundle.main.path(forResource: "neuralize-logo", ofType: "png")!)!
    
    var body: some View {
        VStack {
            Image(uiImage: logo)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 100, height: 100)
            Text("https://www.runlocal.ai")
            if outputImage == nil {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 500, height: 500)
            } else {
                Image(uiImage: UIImage(cgImage: outputImage!))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 400, height: 400)
            }
            
            HStack(spacing: 30) {
                Button("Run with ANE") {
                    outputImage = nil
                    loadTimeMs = nil
                    inferenceTimeMs = nil
                    loading = true
                    computeUnit = MLComputeUnits.all
                    Task {
                        let output = await predict(computeUnit: computeUnit!)
                        outputImage = output.0
                        loadTimeMs = output.1
                        inferenceTimeMs = output.2
                        loading = false
                    }
                }
                Button("Run with GPU") {
                    outputImage = nil
                    loadTimeMs = nil
                    inferenceTimeMs = nil
                    loading = true
                    computeUnit = MLComputeUnits.cpuAndGPU
                    Task {
                        let output = await predict(computeUnit: computeUnit!)
                        outputImage = output.0
                        loadTimeMs = output.1
                        inferenceTimeMs = output.2
                        loading = false
                    }
                }
                Button("Run with CPU") {
                    outputImage = nil
                    loadTimeMs = nil
                    inferenceTimeMs = nil
                    loading = true
                    computeUnit = MLComputeUnits.cpuOnly
                    Task {
                        let output = await predict(computeUnit: computeUnit!)
                        outputImage = output.0
                        loadTimeMs = output.1
                        inferenceTimeMs = output.2
                        loading = false
                    }
                }
                
            }.padding()
            
            Text("Model name: briaai/RMBG-1.4")
                .padding()
            
            if outputImage != nil {
                Text("Inference Time (ms): \(String(format: "%.0f", inferenceTimeMs!))")
                Text("Load Time (ms): \(String(format: "%.0f", loadTimeMs!))")
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
    
        }
        
        .padding()
    }
}

#Preview {
    ContentView()
}
