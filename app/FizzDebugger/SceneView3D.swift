import SwiftUI
import SceneKit

struct SceneView3D: NSViewRepresentable {
    let world: FizzWorldWrapper
    @Binding var selectedEntityId: UInt64?

    func makeNSView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = createScene()
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = NSColor(calibratedWhite: 0.1, alpha: 1.0)
        scnView.showsStatistics = true

        // Click handler
        let clickGesture = NSClickGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleClick(_:)))
        scnView.addGestureRecognizer(clickGesture)

        context.coordinator.scnView = scnView
        return scnView
    }

    func updateNSView(_ scnView: SCNView, context: Context) {
        context.coordinator.updateEntities(world.entities, selectedId: selectedEntityId)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(selectedEntityId: $selectedEntityId)
    }

    private func createScene() -> SCNScene {
        let scene = SCNScene()

        // Ground plane
        let groundGeometry = SCNFloor()
        groundGeometry.reflectivity = 0.1
        let groundMaterial = SCNMaterial()
        groundMaterial.diffuse.contents = NSColor(calibratedWhite: 0.3, alpha: 1.0)
        groundMaterial.isDoubleSided = true
        groundGeometry.materials = [groundMaterial]

        let groundNode = SCNNode(geometry: groundGeometry)
        groundNode.name = "ground"
        scene.rootNode.addChildNode(groundNode)

        // Grid lines
        for i in -10...10 {
            let lineNode = createLineNode(
                from: SCNVector3(Float(i), 0.01, -10),
                to: SCNVector3(Float(i), 0.01, 10),
                color: NSColor(calibratedWhite: 0.4, alpha: 0.5)
            )
            scene.rootNode.addChildNode(lineNode)

            let lineNode2 = createLineNode(
                from: SCNVector3(-10, 0.01, Float(i)),
                to: SCNVector3(10, 0.01, Float(i)),
                color: NSColor(calibratedWhite: 0.4, alpha: 0.5)
            )
            scene.rootNode.addChildNode(lineNode2)
        }

        // Camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.zNear = 0.1
        cameraNode.camera?.zFar = 100
        cameraNode.position = SCNVector3(x: 8, y: 6, z: 8)
        cameraNode.look(at: SCNVector3(x: 0, y: 2, z: 0))
        scene.rootNode.addChildNode(cameraNode)

        // Ambient light
        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = 500
        scene.rootNode.addChildNode(ambientLight)

        // Directional light
        let directionalLight = SCNNode()
        directionalLight.light = SCNLight()
        directionalLight.light?.type = .directional
        directionalLight.light?.intensity = 1000
        directionalLight.light?.castsShadow = true
        directionalLight.position = SCNVector3(5, 10, 5)
        directionalLight.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(directionalLight)

        // Entities container
        let entitiesNode = SCNNode()
        entitiesNode.name = "entities"
        scene.rootNode.addChildNode(entitiesNode)

        return scene
    }

    private func createLineNode(from: SCNVector3, to: SCNVector3, color: NSColor) -> SCNNode {
        let vertices = [from, to]
        let source = SCNGeometrySource(vertices: vertices)
        let indices: [Int32] = [0, 1]
        let data = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(data: data, primitiveType: .line, primitiveCount: 1, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [source], elements: [element])
        let material = SCNMaterial()
        material.diffuse.contents = color
        geometry.materials = [material]
        return SCNNode(geometry: geometry)
    }

    class Coordinator: NSObject {
        @Binding var selectedEntityId: UInt64?
        weak var scnView: SCNView?
        private var entityNodes: [UInt64: SCNNode] = [:]

        init(selectedEntityId: Binding<UInt64?>) {
            _selectedEntityId = selectedEntityId
        }

        func updateEntities(_ entities: [FizzWorldWrapper.EntityState], selectedId: UInt64?) {
            guard let scene = scnView?.scene,
                  let entitiesNode = scene.rootNode.childNode(withName: "entities", recursively: false) else {
                return
            }

            // Track which entities we've seen
            var seenIds = Set<UInt64>()

            for entity in entities {
                seenIds.insert(entity.id)

                if let node = entityNodes[entity.id] {
                    // Update existing node
                    SCNTransaction.begin()
                    SCNTransaction.animationDuration = 0.016 // ~60fps
                    node.position = SCNVector3(entity.position.x, entity.position.y, entity.position.z)
                    SCNTransaction.commit()

                    // Update selection highlight
                    updateNodeAppearance(node, entity: entity, isSelected: entity.id == selectedId)
                } else {
                    // Create new node
                    let node = createEntityNode(entity: entity, isSelected: entity.id == selectedId)
                    entitiesNode.addChildNode(node)
                    entityNodes[entity.id] = node
                }
            }

            // Remove nodes for entities that no longer exist
            let toRemove = entityNodes.keys.filter { !seenIds.contains($0) }
            for id in toRemove {
                entityNodes[id]?.removeFromParentNode()
                entityNodes.removeValue(forKey: id)
            }
        }

        private func createEntityNode(entity: FizzWorldWrapper.EntityState, isSelected: Bool) -> SCNNode {
            let sphere = SCNSphere(radius: 0.3)
            let material = SCNMaterial()

            let color = entity.physicsType.color
            material.diffuse.contents = NSColor(
                calibratedRed: CGFloat(color.r),
                green: CGFloat(color.g),
                blue: CGFloat(color.b),
                alpha: 1.0
            )
            material.specular.contents = NSColor.white

            sphere.materials = [material]

            let node = SCNNode(geometry: sphere)
            node.name = "entity_\(entity.id)"
            node.position = SCNVector3(entity.position.x, entity.position.y, entity.position.z)

            if isSelected {
                addSelectionHighlight(to: node)
            }

            return node
        }

        private func updateNodeAppearance(_ node: SCNNode, entity: FizzWorldWrapper.EntityState, isSelected: Bool) {
            // Remove old highlight
            node.childNode(withName: "highlight", recursively: false)?.removeFromParentNode()

            if isSelected {
                addSelectionHighlight(to: node)
            }
        }

        private func addSelectionHighlight(to node: SCNNode) {
            let ring = SCNTorus(ringRadius: 0.5, pipeRadius: 0.02)
            let material = SCNMaterial()
            material.diffuse.contents = NSColor.yellow
            material.emission.contents = NSColor.yellow
            ring.materials = [material]

            let highlightNode = SCNNode(geometry: ring)
            highlightNode.name = "highlight"
            highlightNode.eulerAngles.x = .pi / 2
            node.addChildNode(highlightNode)
        }

        @objc func handleClick(_ gesture: NSClickGestureRecognizer) {
            guard let scnView = scnView else { return }
            let location = gesture.location(in: scnView)
            let hitResults = scnView.hitTest(location, options: nil)

            for hit in hitResults {
                if let name = hit.node.name, name.hasPrefix("entity_") {
                    let idString = String(name.dropFirst("entity_".count))
                    if let id = UInt64(idString) {
                        selectedEntityId = id
                        return
                    }
                }
            }

            // Clicked on nothing - deselect
            selectedEntityId = nil
        }
    }
}
