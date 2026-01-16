import SwiftUI
import SceneKit

struct ContentView: View {
    @State private var world = FizzWorldWrapper()
    @State private var smc = FizzSMCWrapper()
    @State private var isRunning = false
    @State private var inferenceEnabled = false
    @State private var selectedEntityId: UInt64?
    @State private var timer: Timer?

    var body: some View {
        HSplitView {
            // 3D Scene View
            SceneView3D(world: world, selectedEntityId: $selectedEntityId)
                .frame(minWidth: 600)

            // Sidebar
            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("Fizz Debugger")
                        .font(.headline)
                    Spacer()
                    Text("t=\(world.timestep)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .padding()
                .background(.bar)

                Divider()

                // Controls
                ControlsView(
                    world: world,
                    isRunning: $isRunning,
                    inferenceEnabled: $inferenceEnabled,
                    onStep: stepSimulation,
                    onToggleRun: toggleSimulation,
                    onInitInference: initInference
                )
                .padding()

                Divider()

                // Inference Panel (when enabled)
                if inferenceEnabled {
                    InferenceView(smc: smc, world: world)
                    Divider()
                }

                // Entity List
                EntityListView(
                    world: world,
                    selectedEntityId: $selectedEntityId
                )

                Divider()

                // Inspector
                if let entityId = selectedEntityId,
                   let entity = world.entities.first(where: { $0.id == entityId }) {
                    EntityInspector(world: world, entity: entity, smc: inferenceEnabled ? smc : nil)
                        .padding()
                } else {
                    Text("Select an entity")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(width: 320)
        }
        .onDisappear {
            timer?.invalidate()
        }
    }

    private func stepSimulation() {
        world.step()
        if inferenceEnabled {
            // Generate mock RGB observation (in real use, capture from SceneKit)
            let width: UInt32 = 32
            let height: UInt32 = 32
            var rgbData = [UInt8](repeating: 0, count: Int(width * height * 3))

            // Simple observation: draw spheres as colored circles
            for entity in world.entities {
                let x = Int((entity.position.x + 5) / 10 * Float(width))
                let y = Int((1 - (entity.position.z + 5) / 10) * Float(height))
                let color = entity.physicsType.color

                for dy in -2...2 {
                    for dx in -2...2 {
                        let px = x + dx
                        let py = y + dy
                        if px >= 0 && px < Int(width) && py >= 0 && py < Int(height) {
                            let idx = (py * Int(width) + px) * 3
                            rgbData[idx] = UInt8(color.r * 255)
                            rgbData[idx + 1] = UInt8(color.g * 255)
                            rgbData[idx + 2] = UInt8(color.b * 255)
                        }
                    }
                }
            }

            smc.stepWithImage(rgbData: rgbData, width: width, height: height)
        }
    }

    private func toggleSimulation() {
        isRunning.toggle()
        if isRunning {
            timer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { _ in
                Task { @MainActor in
                    stepSimulation()
                }
            }
        } else {
            timer?.invalidate()
            timer = nil
        }
    }

    private func initInference() {
        let positions = world.entities.map { $0.position }
        let velocities = world.entities.map { $0.velocity }
        smc.initWithPrior(positions: positions, velocities: velocities)
    }
}

// MARK: - Controls View

struct ControlsView: View {
    let world: FizzWorldWrapper
    @Binding var isRunning: Bool
    @Binding var inferenceEnabled: Bool
    let onStep: () -> Void
    let onToggleRun: () -> Void
    let onInitInference: () -> Void

    @State private var spawnPhysicsType: FizzWorldWrapper.PhysicsType = .standard
    @State private var spawnHeight: Float = 5.0

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Simulation controls
            HStack {
                Button(action: onStep) {
                    Image(systemName: "forward.frame")
                }
                .disabled(isRunning)
                .help("Step once")

                Button(action: onToggleRun) {
                    Image(systemName: isRunning ? "pause.fill" : "play.fill")
                }
                .help(isRunning ? "Pause" : "Run")

                Spacer()

                Text("\(world.entities.count) entities")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Inference toggle
            HStack {
                Toggle("Inference", isOn: $inferenceEnabled)
                    .toggleStyle(.switch)

                if inferenceEnabled {
                    Button("Init") {
                        onInitInference()
                    }
                    .buttonStyle(.bordered)
                    .help("Initialize SMC with current entities")
                }
            }

            Divider()

            // Spawn controls
            Text("Spawn Entity")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                Picker("Type", selection: $spawnPhysicsType) {
                    ForEach(FizzWorldWrapper.PhysicsType.allCases, id: \.self) { type in
                        Text(type.description).tag(type)
                    }
                }
                .labelsHidden()
                .frame(width: 100)

                Slider(value: $spawnHeight, in: 1...10) {
                    Text("Height")
                }
                .frame(width: 80)

                Text("y=\(spawnHeight, specifier: "%.1f")")
                    .font(.caption)
                    .frame(width: 40)
            }

            HStack {
                Button("Add") {
                    // Random x/z position
                    let x = Float.random(in: -3...3)
                    let z = Float.random(in: -3...3)
                    world.addEntity(
                        position: SIMD3(x, spawnHeight, z),
                        physicsType: spawnPhysicsType
                    )
                }

                Button("Add 5") {
                    for _ in 0..<5 {
                        let x = Float.random(in: -3...3)
                        let z = Float.random(in: -3...3)
                        let y = spawnHeight + Float.random(in: 0...2)
                        world.addEntity(
                            position: SIMD3(x, y, z),
                            physicsType: spawnPhysicsType
                        )
                    }
                }

                Spacer()

                Button("Clear All") {
                    for entity in world.entities {
                        world.removeEntity(id: entity.id)
                    }
                }
                .foregroundStyle(.red)
            }
        }
    }
}

// MARK: - Entity List View

struct EntityListView: View {
    let world: FizzWorldWrapper
    @Binding var selectedEntityId: UInt64?

    var body: some View {
        List(selection: $selectedEntityId) {
            ForEach(world.entities) { entity in
                HStack {
                    Circle()
                        .fill(Color(
                            red: Double(entity.physicsType.color.r),
                            green: Double(entity.physicsType.color.g),
                            blue: Double(entity.physicsType.color.b)
                        ))
                        .frame(width: 10, height: 10)

                    Text("Entity \(entity.id)")
                        .font(.system(.body, design: .monospaced))

                    Spacer()

                    Text(entity.physicsType.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .tag(entity.id)
            }
        }
        .listStyle(.sidebar)
    }
}

// MARK: - Entity Inspector

struct EntityInspector: View {
    let world: FizzWorldWrapper
    let entity: FizzWorldWrapper.EntityState
    let smc: FizzSMCWrapper?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Inspector")
                .font(.headline)

            Group {
                LabeledContent("ID") {
                    Text("\(entity.id)")
                        .font(.system(.body, design: .monospaced))
                }

                LabeledContent("Type (truth)") {
                    HStack {
                        Circle()
                            .fill(Color(
                                red: Double(entity.physicsType.color.r),
                                green: Double(entity.physicsType.color.g),
                                blue: Double(entity.physicsType.color.b)
                            ))
                            .frame(width: 8, height: 8)
                        Text(entity.physicsType.description)
                    }
                }

                // Show posterior if inference is enabled
                if let smc = smc, let idx = world.entities.firstIndex(where: { $0.id == entity.id }) {
                    let posteriors = smc.entityPosteriors
                    if idx < posteriors.count {
                        let post = posteriors[idx]
                        PosteriorBarView(posterior: post)
                    }
                }

                Divider()

                Text("Position")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                VectorView(label: "pos", value: entity.position)

                Text("Velocity")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                VectorView(label: "vel", value: entity.velocity)
            }

            Divider()

            // Actions
            HStack {
                Button("Push Up") {
                    world.applyForce(to: entity.id, force: SIMD3(0, 50, 0))
                }

                Button("Push Right") {
                    world.applyForce(to: entity.id, force: SIMD3(20, 0, 0))
                }
            }

            Button("Remove") {
                world.removeEntity(id: entity.id)
            }
            .foregroundStyle(.red)
        }
    }
}

// MARK: - Inference View

struct InferenceView: View {
    let smc: FizzSMCWrapper
    let world: FizzWorldWrapper

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Inference")
                .font(.headline)
                .padding(.horizontal)

            // ESS and Temperature
            HStack {
                Label("ESS", systemImage: "chart.bar")
                Spacer()
                Text("\(smc.ess, specifier: "%.1f")")
                    .font(.system(.body, design: .monospaced))
            }
            .font(.caption)
            .padding(.horizontal)

            HStack {
                Label("Temp", systemImage: "thermometer")
                Spacer()
                Text("\(smc.temperature, specifier: "%.2f")")
                    .font(.system(.body, design: .monospaced))
            }
            .font(.caption)
            .padding(.horizontal)

            Divider()

            // Camera belief
            Text("Camera Belief")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .padding(.horizontal)

            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("pos:")
                    Text("(\(smc.cameraBelief.meanPosition.x, specifier: "%.1f"), \(smc.cameraBelief.meanPosition.y, specifier: "%.1f"), \(smc.cameraBelief.meanPosition.z, specifier: "%.1f"))")
                }
                HStack {
                    Text("yaw:")
                    Text("\(smc.cameraBelief.meanYaw * 180 / .pi, specifier: "%.1f")°")
                }
                HStack {
                    Text("σ²:")
                    Text("(\(smc.cameraBelief.positionVariance.x, specifier: "%.2f"), \(smc.cameraBelief.positionVariance.y, specifier: "%.2f"), \(smc.cameraBelief.positionVariance.z, specifier: "%.2f"))")
                }
            }
            .font(.system(.caption, design: .monospaced))
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }
}

// MARK: - Posterior Bar View

struct PosteriorBarView: View {
    let posterior: FizzSMCWrapper.EntityPosterior

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Inferred")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 2) {
                PosteriorBar(value: posterior.standard, color: .gray, label: "S")
                PosteriorBar(value: posterior.bouncy, color: .green, label: "B")
                PosteriorBar(value: posterior.sticky, color: .red, label: "St")
                PosteriorBar(value: posterior.slippery, color: .cyan, label: "Sl")
            }
            .frame(height: 20)

            Text("→ \(posterior.mostLikely.description)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

struct PosteriorBar: View {
    let value: Float
    let color: Color
    let label: String

    var body: some View {
        VStack(spacing: 1) {
            GeometryReader { geo in
                Rectangle()
                    .fill(color.opacity(0.3))
                    .overlay(alignment: .bottom) {
                        Rectangle()
                            .fill(color)
                            .frame(height: geo.size.height * CGFloat(value))
                    }
            }
            Text(label)
                .font(.system(size: 8))
                .foregroundStyle(.secondary)
        }
    }
}

struct VectorView: View {
    let label: String
    let value: SIMD3<Float>

    var body: some View {
        HStack {
            Text("x: \(value.x, specifier: "%+.2f")")
            Text("y: \(value.y, specifier: "%+.2f")")
            Text("z: \(value.z, specifier: "%+.2f")")
        }
        .font(.system(.caption, design: .monospaced))
    }
}

#Preview {
    ContentView()
}
