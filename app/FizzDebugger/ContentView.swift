import SwiftUI
import SceneKit

struct ContentView: View {
    @State private var world = FizzWorldWrapper()
    @State private var isRunning = false
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
                    onStep: { world.step() },
                    onToggleRun: toggleSimulation
                )
                .padding()

                Divider()

                // Entity List
                EntityListView(
                    world: world,
                    selectedEntityId: $selectedEntityId
                )

                Divider()

                // Inspector
                if let entityId = selectedEntityId,
                   let entity = world.entities.first(where: { $0.id == entityId }) {
                    EntityInspector(world: world, entity: entity)
                        .padding()
                } else {
                    Text("Select an entity")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(width: 300)
        }
        .onDisappear {
            timer?.invalidate()
        }
    }

    private func toggleSimulation() {
        isRunning.toggle()
        if isRunning {
            timer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { _ in
                Task { @MainActor in
                    world.step()
                }
            }
        } else {
            timer?.invalidate()
            timer = nil
        }
    }
}

// MARK: - Controls View

struct ControlsView: View {
    let world: FizzWorldWrapper
    @Binding var isRunning: Bool
    let onStep: () -> Void
    let onToggleRun: () -> Void

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

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Inspector")
                .font(.headline)

            Group {
                LabeledContent("ID") {
                    Text("\(entity.id)")
                        .font(.system(.body, design: .monospaced))
                }

                LabeledContent("Type") {
                    Text(entity.physicsType.description)
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
