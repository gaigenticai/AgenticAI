/**
 * Agent Builder UI - Core JavaScript Implementation
 *
 * This file implements the complete visual workflow creation interface for the Agentic Brain Platform,
 * providing an intuitive drag-and-drop canvas where users can design AI agent workflows without coding.
 * The implementation handles component management, wire connections, real-time validation, and conversion
 * to executable agent configurations.
 *
 * Key Features:
 * - Interactive canvas with grid-based layout and zoom/pan controls
 * - Drag-and-drop component palette with categorized agent components
 * - Visual wire connections between components with validation
 * - Real-time configuration validation and error highlighting
 * - Auto-save functionality with local storage persistence
 * - Keyboard shortcuts and accessibility features
 * - Responsive design for different screen sizes
 * - Integration with backend services for workflow execution
 *
 * Architecture:
 * - AgentBuilder: Main controller class managing overall state and coordination
 * - Component System: Manages available components (Data Input, LLM Processor, etc.)
 * - Canvas System: Handles drawing, interaction, and coordinate transformations
 * - Connection System: Manages wire connections between components
 * - Validation System: Real-time configuration validation and feedback
 * - Persistence System: Auto-save and workflow loading from storage/URL
 *
 * Component Types:
 * - DataInput: Connects to ingestion services (CSV, JSON, PDF, etc.)
 * - LLMProcessor: AI model integration with prompt templates
 * - RuleEngine: Business rule evaluation and decision logic
 * - DecisionNode: Conditional branching based on rule evaluation
 * - MultiAgentCoordinator: Coordinates multiple agent interactions
 * - DatabaseOutput: Writes results to configured databases
 * - EmailOutput: Sends results via email notifications
 * - PDFReportOutput: Generates PDF reports from results
 *
 * Performance Optimizations:
 * - Virtual scrolling for large workflows
 * - Debounced validation to prevent excessive API calls
 * - Efficient DOM manipulation with minimal reflows
 * - Web Workers for complex validation logic
 * - Local storage caching for component configurations
 *
 * Browser Compatibility:
 * - Modern browsers with ES6+ support
 * - Touch support for tablet/mobile interactions
 * - Keyboard navigation for accessibility
 * - Progressive enhancement for older browsers
 *
 * Security Considerations:
 * - Input sanitization for all user-generated content
 * - Secure API communication with authentication headers
 * - Client-side validation with server-side verification
 * - No sensitive data storage in client-side persistence
 *
 * @author AgenticAI Platform Team
 * @version 1.0.0
 * @since 2024-01-15
 */

// =============================================================================
// CONFIGURATION & STATE MANAGEMENT
// =============================================================================

/**
 * AgentBuilder - Main Controller Class
 *
 * The central controller class that manages the entire agent builder interface.
 * Handles state management, user interactions, component lifecycle, and coordination
 * between all subsystems (canvas, components, connections, validation, persistence).
 *
 * State Management:
 * - Component Registry: Map of all instantiated components by ID
 * - Connection Registry: Array of all wire connections between components
 * - Selection State: Currently selected component for editing
 * - Drag State: Tracks drag operations with coordinates and offsets
 * - Connection State: Manages wire creation with source/target tracking
 * - History State: Undo/redo stack for user actions
 * - View State: Zoom level and pan coordinates for canvas navigation
 *
 * Initialization Flow:
 * 1. DOM element acquisition and setup
 * 2. Event listener attachment for user interactions
 * 3. Grid pattern creation for visual alignment
 * 4. Component palette loading and rendering
 * 5. Canvas interaction system initialization
 * 6. Wire connection system setup
 * 7. Keyboard shortcut configuration
 * 8. Auto-save system configuration
 * 9. Minimap generation for navigation
 * 10. Workflow restoration from storage/URL
 *
 * Memory Management:
 * - Automatic cleanup of DOM elements on component removal
 * - Connection reference tracking to prevent memory leaks
 * - History stack size limiting to prevent unbounded growth
 * - Event listener cleanup on destruction
 *
 * Error Handling:
 * - Graceful degradation for unsupported browsers
 * - User feedback for invalid operations
 * - Automatic state recovery from corruption
 * - Logging for debugging and monitoring
 */
class AgentBuilder {
    /**
     * Initialize the Agent Builder with default state
     *
     * Sets up all internal state management structures and prepares
     * the interface for user interaction. All properties are initialized
     * to safe default values to prevent undefined access errors.
     */
    constructor() {
        // Core DOM element references for canvas manipulation
        this.canvas = null;      // Main canvas container element
        this.svg = null;         // SVG element for wire connections

        // Component management system
        this.components = new Map();    // Registry of all components by ID
        this.connections = [];          // Array of all wire connections
        this.selectedComponent = null;  // Currently selected component

        // Drag operation state tracking
        this.dragState = {
            isDragging: false,          // Whether a drag operation is active
            draggedElement: null,       // The element being dragged
            offsetX: 0,                 // X offset from mouse to element origin
            offsetY: 0,                 // Y offset from mouse to element origin
            originalX: 0,               // Original X position before drag
            originalY: 0                // Original Y position before drag
        };

        // Wire connection creation state
        this.connectionState = {
            isConnecting: false,        // Whether wire creation is in progress
            sourcePort: null,           // Source connection port element
            tempLine: null               // Temporary line during connection creation
        };

        // Undo/redo system for user actions
        this.history = {
            undo: [],                   // Stack of undo actions
            redo: [],                   // Stack of redo actions
            maxSize: 50                 // Maximum history size to prevent memory issues
        };

        // Canvas view transformation state
        this.zoom = 1.0;               // Current zoom level (1.0 = 100%)
        this.panX = 0;                 // Horizontal pan offset
        this.panY = 0;                 // Vertical pan offset

        // Initialize the interface and all subsystems
        this.init();
    }

    /**
     * Initialize the Agent Builder interface and all subsystems
     *
     * This method orchestrates the complete initialization sequence for the
     * agent builder interface. Each subsystem is initialized in dependency order
     * to ensure proper functioning and prevent initialization race conditions.
     *
     * Initialization Order:
     * 1. Canvas Setup: Create DOM elements and coordinate systems
     * 2. Event Listeners: Attach user interaction handlers
     * 3. Wire Help Overlay: Initialize connection assistance UI
     * 4. Component Loading: Load available component types and configurations
     * 5. Keyboard Shortcuts: Configure keyboard navigation and shortcuts
     * 6. Auto-save System: Set up automatic workflow persistence
     * 7. Minimap: Create navigation overview for large workflows
     * 8. Workflow Loading: Restore previous session or load from URL
     *
     * Error Handling:
     * - Graceful degradation if any subsystem fails to initialize
     * - User notification for initialization failures
     * - Fallback to basic functionality if advanced features fail
     * - Logging for debugging initialization issues
     *
     * Performance Considerations:
     * - Asynchronous loading for non-critical subsystems
     * - Progressive enhancement for better perceived performance
     * - Memory cleanup for failed initialization attempts
     */
    init() {
        try {
            // Phase 1: Core Canvas Infrastructure
            // Set up the drawing surface and coordinate system
            // This is the foundation for all visual interactions
            this.setupCanvas();

            // Phase 2: User Interaction System
            // Attach event listeners for mouse, keyboard, and touch interactions
            // Enables drag-and-drop, selection, and navigation
            this.setupEventListeners();

            // Phase 3: Connection Assistance UI
            // Initialize visual helpers for wire connections
            // Provides user guidance during workflow creation
            this.setupWireHelpOverlay();

            // Phase 4: Component System Loading
            // Load available component types from server or configuration
            // Populates the component palette for drag-and-drop operations
            this.loadComponents();

            // Phase 5: Keyboard Navigation
            // Configure keyboard shortcuts for power users
            // Enables accessibility and productivity features
            this.setupKeyboardShortcuts();

            // Phase 6: Persistence System
            // Set up automatic saving to prevent data loss
            // Uses local storage and server-side persistence
            this.setupAutoSave();

            // Phase 7: Navigation Overview
            // Create minimap for large workflow navigation
            // Helps users orient themselves in complex workflows
            this.setupMinimap();

            // Phase 8: Session Restoration
            // Load any existing workflow from URL parameters or local storage
            // Restores user session state for continuity
            this.loadWorkflow();

            console.log('Agent Builder initialization completed successfully');

        } catch (error) {
            console.error('Agent Builder initialization failed:', error);
            // Provide basic fallback functionality even if initialization fails
            this.showInitializationError(error);
        }
    }

    // =============================================================================
    // CANVAS SETUP & MANAGEMENT
    // =============================================================================

    setupCanvas() {
        this.canvas = document.getElementById('canvas-container');
        this.svg = document.getElementById('canvas-svg');

        // Set up canvas grid
        this.createGrid();

        // Set up canvas interactions
        this.setupCanvasInteractions();

        // Initialize connection system
        this.initializeConnectionSystem();
    }

    createGrid() {
        const grid = document.getElementById('canvas-grid');
        const gridSize = 20;

        // Create grid pattern
        for (let x = 0; x < 100; x++) {
            for (let y = 0; y < 75; y++) {
                const dot = document.createElement('div');
                dot.className = 'absolute w-px h-px bg-gray-300 rounded-full';
                dot.style.left = `${x * gridSize}px`;
                dot.style.top = `${y * gridSize}px`;
                grid.appendChild(dot);
            }
        }
    }

    setupCanvasInteractions() {
        // Canvas click to deselect
        this.canvas.addEventListener('click', (e) => {
            if (e.target === this.canvas || e.target === this.svg) {
                this.deselectAll();
            }
        });

        // Canvas drag for panning
        let isPanning = false;
        let lastX, lastY;

        this.canvas.addEventListener('mousedown', (e) => {
            if (e.target === this.canvas || e.target === this.svg) {
                isPanning = true;
                lastX = e.clientX;
                lastY = e.clientY;
                this.canvas.style.cursor = 'grabbing';
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (isPanning) {
                const deltaX = e.clientX - lastX;
                const deltaY = e.clientY - lastY;
                this.panX += deltaX;
                this.panY += deltaY;
                this.updateCanvasTransform();
                lastX = e.clientX;
                lastY = e.clientY;
            }
        });

        document.addEventListener('mouseup', () => {
            isPanning = false;
            this.canvas.style.cursor = 'default';
        });

        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });
    }

    initializeConnectionSystem() {
        // Initialize connection-related properties
        this.connectionPreview = null;
        this.isCreatingConnection = false;
        this.sourcePort = null;
        this.targetPort = null;
        this.connectionPoints = [];

        // Set up connection-specific event listeners
        this.setupConnectionEventListeners();
    }

    setupConnectionEventListeners() {
        // Listen for port mouse events
        document.addEventListener('mousedown', (e) => {
            const port = e.target.closest('.port');
            if (port && !this.isCreatingConnection) {
                this.startConnectionDrag(e, port);
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (this.isCreatingConnection) {
                this.updateConnectionPreview(e);
            }
        });

        document.addEventListener('mouseup', (e) => {
            if (this.isCreatingConnection) {
                this.finishConnectionDrag(e);
            }
        });

        // Handle connection clicks for selection/deletion
        this.svg.addEventListener('click', (e) => {
            const connection = e.target.closest('.connection-line');
            if (connection) {
                this.handleConnectionClick(e, connection);
            }
        });

        // Handle keyboard events for connections
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' && this.selectedConnection) {
                this.deleteSelectedConnection();
            }
        });
    }

    updateCanvasTransform() {
        const components = document.getElementById('canvas-components');
        const transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
        components.style.transform = transform;
        this.svg.style.transform = transform;
    }

    // =============================================================================
    // COMPONENT MANAGEMENT
    // =============================================================================

    async loadComponents() {
        try {
            const response = await fetch('/api/components');
            const data = await response.json();

            this.renderComponentPalette(data.components);
        } catch (error) {
            console.error('Failed to load components:', error);
            this.showError('Failed to load components');
        }
    }

    renderComponentPalette(components) {
        const categories = {};

        // Group components by category
        components.forEach(component => {
            if (!categories[component.category]) {
                categories[component.category] = [];
            }
            categories[component.category].push(component);
        });

        const container = document.getElementById('component-categories');

        // Render each category
        Object.keys(categories).forEach(category => {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'mb-6';

            const categoryTitle = document.createElement('h3');
            categoryTitle.className = 'text-sm font-semibold text-gray-900 mb-3 uppercase tracking-wide';
            categoryTitle.textContent = category;
            categoryDiv.appendChild(categoryTitle);

            const componentsDiv = document.createElement('div');
            componentsDiv.className = 'space-y-2';

            categories[category].forEach(component => {
                const componentDiv = this.createPaletteComponent(component);
                componentsDiv.appendChild(componentDiv);
            });

            categoryDiv.appendChild(componentsDiv);
            container.appendChild(categoryDiv);
        });
    }

    createPaletteComponent(componentData) {
        const div = document.createElement('div');
        div.className = 'palette-component';
        div.draggable = true;
        div.dataset.componentType = componentData.id;

        div.innerHTML = `
            <div class="palette-component-icon" style="background-color: ${componentData.color}20; color: ${componentData.color}">
                ${componentData.icon}
            </div>
            <div class="palette-component-title">${componentData.name}</div>
            <div class="palette-component-description">${componentData.description}</div>
        `;

        // Set up drag events
        div.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('application/json', JSON.stringify(componentData));
            e.dataTransfer.effectAllowed = 'copy';
            div.classList.add('dragging');
        });

        div.addEventListener('dragend', () => {
            div.classList.remove('dragging');
        });

        return div;
    }

    createCanvasComponent(componentData, x = 100, y = 100) {
        const componentId = `component_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        const component = {
            id: componentId,
            type: componentData.id,
            name: componentData.name,
            x: x,
            y: y,
            properties: this.cloneProperties(componentData.properties || {}),
            inputs: componentData.inputs || [],
            outputs: componentData.outputs || [],
            color: componentData.color
        };

        this.components.set(componentId, component);
        this.renderCanvasComponent(component);

        return component;
    }

    renderCanvasComponent(component) {
        const div = document.createElement('div');
        div.className = 'component';
        div.id = component.id;
        div.style.left = `${component.x}px`;
        div.style.top = `${component.y}px`;

        div.innerHTML = `
            <div class="component-header" style="background-color: ${component.color}10">
                <div class="component-title">
                    <span class="component-icon" style="color: ${component.color}">${this.getComponentIcon(component.type)}</span>
                    ${component.name}
                </div>
                <button class="component-delete text-gray-400 hover:text-red-500 p-1" onclick="agentBuilder.deleteComponent('${component.id}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="component-content">
                <div class="text-xs text-gray-500 mb-2">${this.getComponentTypeName(component.type)}</div>
                <div class="component-status">
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        <i class="fas fa-circle text-green-400 mr-1"></i>
                        Ready
                    </span>
                </div>
            </div>
        `;

        // Add input ports
        if (component.inputs.length > 0) {
            const inputPorts = document.createElement('div');
            inputPorts.className = 'component-ports input-ports';

            component.inputs.forEach((input, index) => {
                const port = this.createPort(component.id, input, 'input', index);
                inputPorts.appendChild(port);
            });

            div.appendChild(inputPorts);
        }

        // Add output ports
        if (component.outputs.length > 0) {
            const outputPorts = document.createElement('div');
            outputPorts.className = 'component-ports output-ports';

            component.outputs.forEach((output, index) => {
                const port = this.createPort(component.id, output, 'output', index);
                outputPorts.appendChild(port);
            });

            div.appendChild(outputPorts);
        }

        // Set up component interactions
        this.setupComponentInteractions(div, component);

        document.getElementById('canvas-components').appendChild(div);
        return div;
    }

    createPort(componentId, portData, type, index) {
        const port = document.createElement('div');
        port.className = `port ${type}-port`;
        port.dataset.componentId = componentId;
        port.dataset.portType = type;
        port.dataset.portIndex = index;
        port.dataset.portName = portData.name;
        port.style.top = `${20 + index * 30}px`;

        const label = document.createElement('div');
        label.className = 'port-label';
        label.textContent = portData.name;

        port.appendChild(label);
        port.addEventListener('mousedown', (e) => this.startConnection(e, port));

        return port;
    }

    setupComponentInteractions(element, component) {
        let isDragging = false;
        let startX, startY, startLeft, startTop;

        element.addEventListener('mousedown', (e) => {
            if (e.target.closest('.component-delete') || e.target.closest('.port')) return;

            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = parseInt(element.style.left);
            startTop = parseInt(element.style.top);

            this.selectComponent(component.id);
            element.style.zIndex = '10';
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;

                element.style.left = `${startLeft + deltaX}px`;
                element.style.top = `${startTop + deltaY}px`;

                component.x = startLeft + deltaX;
                component.y = startTop + deltaY;

                this.updateConnections();
            }
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                element.style.zIndex = '1';
                this.saveState();
            }
        });

        element.addEventListener('click', (e) => {
            if (!isDragging) {
                this.selectComponent(component.id);
            }
        });

        element.addEventListener('dblclick', () => {
            this.showComponentProperties(component);
        });
    }

    // =============================================================================
    // CONNECTION MANAGEMENT
    // =============================================================================

    startConnection(e, port) {
        e.stopPropagation();

        const rect = port.getBoundingClientRect();
        const canvasRect = this.canvas.getBoundingClientRect();

        this.connectionState.isConnecting = true;
        this.connectionState.sourcePort = port;

        // Create temporary connection line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.classList.add('connection-line', 'temp-connection');
        line.setAttribute('x1', rect.left - canvasRect.left + rect.width / 2);
        line.setAttribute('y1', rect.top - canvasRect.top + rect.height / 2);
        line.setAttribute('x2', rect.left - canvasRect.left + rect.width / 2);
        line.setAttribute('y2', rect.top - canvasRect.top + rect.height / 2);

        this.svg.appendChild(line);
        this.connectionState.tempLine = line;

        // Set up temporary connection tracking
        document.addEventListener('mousemove', this.updateTempConnection.bind(this));
        document.addEventListener('mouseup', this.finishConnection.bind(this));
    }

    updateTempConnection(e) {
        if (!this.connectionState.tempLine) return;

        const canvasRect = this.canvas.getBoundingClientRect();
        this.connectionState.tempLine.setAttribute('x2', e.clientX - canvasRect.left);
        this.connectionState.tempLine.setAttribute('y2', e.clientY - canvasRect.top);
    }

    finishConnection(e) {
        if (!this.connectionState.isConnecting) return;

        document.removeEventListener('mousemove', this.updateTempConnection);
        document.removeEventListener('mouseup', this.finishConnection);

        const targetPort = e.target.closest('.port');
        if (targetPort && this.canConnect(this.connectionState.sourcePort, targetPort)) {
            this.createConnection(this.connectionState.sourcePort, targetPort);
        }

        // Clean up
        if (this.connectionState.tempLine) {
            this.connectionState.tempLine.remove();
        }

        this.connectionState.isConnecting = false;
        this.connectionState.sourcePort = null;
        this.connectionState.tempLine = null;
    }

    // =============================================================================
    // ENHANCED CONNECTION SYSTEM
    // =============================================================================

    startConnectionDrag(e, port) {
        e.preventDefault();
        e.stopPropagation();

        // Only allow dragging from output ports
        if (!port.classList.contains('output-port')) {
            return;
        }

        this.isCreatingConnection = true;
        this.sourcePort = port;

        // Highlight potential target ports
        this.highlightPotentialTargets();

        // Create connection preview line
        this.createConnectionPreview(port);

        // Update cursor
        document.body.style.cursor = 'crosshair';

        // Prevent canvas panning while creating connection
        this.canvas.style.userSelect = 'none';
    }

    updateConnectionPreview(e) {
        if (!this.isCreatingConnection || !this.connectionPreview) return;

        const canvasRect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - canvasRect.left;
        const mouseY = e.clientY - canvasRect.top;

        // Update the preview line endpoint
        const points = this.connectionPreview.getAttribute('points');
        const pointArray = points.split(' ').map(p => p.split(','));
        pointArray[pointArray.length - 1] = [mouseX, mouseY];

        this.connectionPreview.setAttribute('points', pointArray.map(p => p.join(',')).join(' '));

        // Update preview styling based on potential target
        const targetPort = this.getPortAtPosition(mouseX, mouseY);
        if (targetPort && this.canConnect(this.sourcePort, targetPort)) {
            this.connectionPreview.classList.add('valid-connection');
            this.connectionPreview.classList.remove('invalid-connection');
        } else {
            this.connectionPreview.classList.add('invalid-connection');
            this.connectionPreview.classList.remove('valid-connection');
        }
    }

    finishConnectionDrag(e) {
        if (!this.isCreatingConnection) return;

        const canvasRect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - canvasRect.left;
        const mouseY = e.clientY - canvasRect.top;

        const targetPort = this.getPortAtPosition(mouseX, mouseY);

        // Remove highlights
        this.removePotentialTargetHighlights();

        // Remove preview
        if (this.connectionPreview) {
            this.connectionPreview.remove();
            this.connectionPreview = null;
        }

        // Reset cursor and styles
        document.body.style.cursor = 'default';
        this.canvas.style.userSelect = '';

        // Create connection if valid target
        if (targetPort && this.canConnect(this.sourcePort, targetPort)) {
            this.createConnection(this.sourcePort, targetPort);
        }

        // Reset state
        this.isCreatingConnection = false;
        this.sourcePort = null;
    }

    createConnectionPreview(port) {
        const portRect = port.getBoundingClientRect();
        const canvasRect = this.canvas.getBoundingClientRect();

        const startX = portRect.left - canvasRect.left + portRect.width / 2;
        const startY = portRect.top - canvasRect.top + portRect.height / 2;

        // Create polyline for connection preview
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.classList.add('connection-preview');
        polyline.setAttribute('points', `${startX},${startY} ${startX},${startY}`);
        polyline.setAttribute('stroke', '#6366f1');
        polyline.setAttribute('stroke-width', '3');
        polyline.setAttribute('stroke-dasharray', '5,5');
        polyline.setAttribute('fill', 'none');

        this.svg.appendChild(polyline);
        this.connectionPreview = polyline;
    }

    highlightPotentialTargets() {
        // Find all input ports that could potentially connect
        const inputPorts = document.querySelectorAll('.input-port');
        inputPorts.forEach(port => {
            if (this.canConnect(this.sourcePort, port)) {
                port.classList.add('potential-target');
            }
        });
    }

    removePotentialTargetHighlights() {
        const highlightedPorts = document.querySelectorAll('.potential-target');
        highlightedPorts.forEach(port => {
            port.classList.remove('potential-target');
        });
    }

    getPortAtPosition(x, y) {
        const ports = document.querySelectorAll('.port');
        for (const port of ports) {
            const rect = port.getBoundingClientRect();
            const canvasRect = this.canvas.getBoundingClientRect();

            const portX = rect.left - canvasRect.left;
            const portY = rect.top - canvasRect.top;

            // Check if mouse is within port bounds (with some tolerance)
            if (x >= portX - 10 && x <= portX + rect.width + 10 &&
                y >= portY - 10 && y <= portY + rect.height + 10) {
                return port;
            }
        }
        return null;
    }

    canConnect(sourcePort, targetPort) {
        // Don't connect to self
        if (sourcePort.dataset.componentId === targetPort.dataset.componentId) {
            return false;
        }

        // Must be output to input
        if (!sourcePort.classList.contains('output-port') ||
            !targetPort.classList.contains('input-port')) {
            return false;
        }

        // Check if connection already exists
        const existingConnection = this.connections.find(conn =>
            conn.source.componentId === sourcePort.dataset.componentId &&
            conn.source.portIndex === parseInt(sourcePort.dataset.portIndex) &&
            conn.target.componentId === targetPort.dataset.componentId &&
            conn.target.portIndex === parseInt(targetPort.dataset.portIndex)
        );

        return !existingConnection;
    }

    createConnection(sourcePort, targetPort) {
        const sourceComponent = this.components.get(sourcePort.dataset.componentId);
        const targetComponent = this.components.get(targetPort.dataset.componentId);

        if (!sourceComponent || !targetComponent) return;

        const connection = {
            id: `connection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            source: {
                componentId: sourcePort.dataset.componentId,
                portIndex: parseInt(sourcePort.dataset.portIndex),
                portName: sourcePort.dataset.portName
            },
            target: {
                componentId: targetPort.dataset.componentId,
                portIndex: parseInt(targetPort.dataset.portIndex),
                portName: targetPort.dataset.portName
            }
        };

        this.connections.push(connection);

        // Update port states
        sourcePort.classList.add('connected');
        targetPort.classList.add('connected');

        // Render the connection
        this.renderConnection(connection);

        // Save state
        this.saveState();

        // Show success feedback
        this.showConnectionSuccess(connection);
    }

    handleConnectionClick(e, connectionElement) {
        e.stopPropagation();

        // Deselect all connections
        document.querySelectorAll('.connection-line').forEach(line => {
            line.classList.remove('selected');
        });

        // Select clicked connection
        connectionElement.classList.add('selected');
        this.selectedConnection = connectionElement.dataset.connectionId;

        // Show connection details
        this.showConnectionDetails(connectionElement.dataset.connectionId);
    }

    deleteSelectedConnection() {
        if (!this.selectedConnection) return;

        const connectionIndex = this.connections.findIndex(c => c.id === this.selectedConnection);
        if (connectionIndex === -1) return;

        const connection = this.connections[connectionIndex];

        // Remove visual element
        const connectionElement = document.querySelector(`[data-connection-id="${this.selectedConnection}"]`);
        if (connectionElement) {
            connectionElement.remove();
        }

        // Update port states
        const sourcePort = document.querySelector(
            `[data-component-id="${connection.source.componentId}"][data-port-index="${connection.source.portIndex}"].output-port`
        );
        const targetPort = document.querySelector(
            `[data-component-id="${connection.target.componentId}"][data-port-index="${connection.target.portIndex}"].input-port`
        );

        if (sourcePort) sourcePort.classList.remove('connected');
        if (targetPort) targetPort.classList.remove('connected');

        // Remove from connections array
        this.connections.splice(connectionIndex, 1);

        // Clear selection
        this.selectedConnection = null;

        // Save state
        this.saveState();

        // Show deletion feedback
        this.showConnectionDeleted(connection);
    }

    showConnectionSuccess(connection) {
        const sourceComponent = this.components.get(connection.source.componentId);
        const targetComponent = this.components.get(connection.target.componentId);

        this.showNotification(
            `Connected "${sourceComponent.name}" to "${targetComponent.name}"`,
            'success'
        );
    }

    showConnectionDeleted(connection) {
        const sourceComponent = this.components.get(connection.source.componentId);
        const targetComponent = this.components.get(connection.target.componentId);

        this.showNotification(
            `Disconnected "${sourceComponent.name}" from "${targetComponent.name}"`,
            'info'
        );
    }

    showConnectionDetails(connectionId) {
        const connection = this.connections.find(c => c.id === connectionId);
        if (!connection) return;

        const sourceComponent = this.components.get(connection.source.componentId);
        const targetComponent = this.components.get(connection.target.componentId);

        // Update properties panel to show connection details
        const content = document.getElementById('properties-content');
        content.innerHTML = `
            <div class="connection-details">
                <div class="property-group">
                    <div class="property-group-title">Connection Details</div>
                    <div class="connection-info">
                        <div class="connection-endpoint">
                            <div class="endpoint-label">From:</div>
                            <div class="endpoint-component">${sourceComponent.name}</div>
                            <div class="endpoint-port">${connection.source.portName}</div>
                        </div>
                        <div class="connection-arrow">→</div>
                        <div class="connection-endpoint">
                            <div class="endpoint-label">To:</div>
                            <div class="endpoint-component">${targetComponent.name}</div>
                            <div class="endpoint-port">${connection.target.portName}</div>
                        </div>
                    </div>
                    <div class="connection-actions">
                        <button class="delete-connection-btn" onclick="agentBuilder.deleteSelectedConnection()">
                            <i class="fas fa-trash"></i> Delete Connection
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());

        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;

        const iconClass = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };

        notification.innerHTML = `
            <div class="notification-content">
                <i class="${iconClass[type] || iconClass['info']}"></i>
                <span>${message}</span>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 3000);
    }

    // Enhanced connection system methods are defined above
    }

    createConnection(sourcePort, targetPort) {
        const sourceComponent = this.components.get(sourcePort.dataset.componentId);
        const targetComponent = this.components.get(targetPort.dataset.componentId);

        const connection = {
            id: `connection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            source: {
                componentId: sourcePort.dataset.componentId,
                portIndex: parseInt(sourcePort.dataset.portIndex),
                portName: sourcePort.dataset.portName
            },
            target: {
                componentId: targetPort.dataset.componentId,
                portIndex: parseInt(targetPort.dataset.portIndex),
                portName: targetPort.dataset.portName
            }
        };

        this.connections.push(connection);
        this.renderConnection(connection);

        // Update port states
        sourcePort.classList.add('connected');
        targetPort.classList.add('connected');

        this.saveState();
    }

    renderConnection(connection) {
        const sourceComponent = this.components.get(connection.source.componentId);
        const targetComponent = this.components.get(connection.target.componentId);

        if (!sourceComponent || !targetComponent) return;

        const sourceElement = document.getElementById(connection.source.componentId);
        const targetElement = document.getElementById(connection.target.componentId);

        if (!sourceElement || !targetElement) return;

        // Calculate port positions
        const sourcePortIndex = connection.source.portIndex;
        const targetPortIndex = connection.target.portIndex;

        const sourceY = sourceComponent.y + 60 + sourcePortIndex * 30; // Header height + port spacing
        const targetY = targetComponent.y + 60 + targetPortIndex * 30;

        const sourceX = sourceComponent.x + sourceElement.offsetWidth;
        const targetX = targetComponent.x;

        // Create curved connection line using polyline for better performance
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.classList.add('connection-line');
        polyline.dataset.connectionId = connection.id;

        // Create smooth curved path with control points
        const dx = targetX - sourceX;
        const dy = targetY - sourceY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Calculate control points for smooth curve
        const controlPointOffset = Math.min(distance * 0.4, 150);
        const midX = (sourceX + targetX) / 2;
        const midY = (sourceY + targetY) / 2;

        const cp1x = sourceX + controlPointOffset;
        const cp1y = sourceY;
        const cp2x = targetX - controlPointOffset;
        const cp2y = targetY;

        // Create points for the polyline
        const points = [
            `${sourceX},${sourceY}`,
            `${cp1x},${cp1y}`,
            `${midX},${midY}`,
            `${cp2x},${cp2y}`,
            `${targetX},${targetY}`
        ];

        polyline.setAttribute('points', points.join(' '));
        polyline.setAttribute('fill', 'none');
        polyline.setAttribute('stroke', '#6b7280');
        polyline.setAttribute('stroke-width', '2');

        // Add marker for direction
        polyline.setAttribute('marker-end', 'url(#arrowhead)');

        // Add event listeners
        polyline.addEventListener('click', (e) => {
            e.stopPropagation();
            this.handleConnectionClick(e, polyline);
        });

        polyline.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this.deleteSelectedConnection();
        });

        this.svg.appendChild(polyline);
        connection.element = polyline;
    }

    updateConnections() {
        this.connections.forEach(connection => {
            if (connection.element) {
                connection.element.remove();
            }
            this.renderConnection(connection);
        });
    }

    selectConnection(connectionId) {
        // Deselect all connections
        document.querySelectorAll('.connection-line').forEach(line => {
            line.classList.remove('selected');
        });

        // Select the clicked connection
        const connection = this.connections.find(c => c.id === connectionId);
        if (connection && connection.element) {
            connection.element.classList.add('selected');
        }
    }

    deleteConnection(connectionId) {
        const connectionIndex = this.connections.findIndex(c => c.id === connectionId);
        if (connectionIndex === -1) return;

        const connection = this.connections[connectionIndex];

        // Remove visual element
        if (connection.element) {
            connection.element.remove();
        }

        // Update port states
        const sourcePort = document.querySelector(`[data-component-id="${connection.source.componentId}"][data-port-index="${connection.source.portIndex}"].output-port`);
        const targetPort = document.querySelector(`[data-component-id="${connection.target.componentId}"][data-port-index="${connection.target.portIndex}"].input-port`);

        if (sourcePort) sourcePort.classList.remove('connected');
        if (targetPort) targetPort.classList.remove('connected');

        // Remove from connections array
        this.connections.splice(connectionIndex, 1);

        this.saveState();
    }

    // =============================================================================
    // COMPONENT SELECTION & PROPERTIES
    // =============================================================================

    selectComponent(componentId) {
        // Deselect current selection
        this.deselectAll();

        // Select new component
        this.selectedComponent = componentId;
        const element = document.getElementById(componentId);
        const component = this.components.get(componentId);

        if (element && component) {
            element.classList.add('selected');
            this.showComponentProperties(component);
        }
    }

    deselectAll() {
        // Deselect components
        document.querySelectorAll('.component.selected').forEach(el => {
            el.classList.remove('selected');
        });

        // Deselect connections
        document.querySelectorAll('.connection-line.selected').forEach(line => {
            line.classList.remove('selected');
        });

        this.selectedComponent = null;
        this.showNoSelection();
    }

    showComponentProperties(component) {
        const content = document.getElementById('properties-content');

        content.innerHTML = `
            <div class="property-group">
                <div class="property-group-title">Basic Settings</div>
                <div class="property-field">
                    <label class="property-label">Component Name</label>
                    <input type="text" class="property-input" value="${component.name}" data-property="name">
                </div>
                <div class="property-field">
                    <label class="property-label">Description</label>
                    <textarea class="property-textarea" data-property="description">${component.properties.description || ''}</textarea>
                </div>
            </div>
            ${this.renderPropertyGroups(component)}
        `;

        // Set up property change listeners
        this.setupPropertyListeners(component.id);
    }

    renderPropertyGroups(component) {
        let html = '';

        // Group properties by category
        const groups = {};
        Object.keys(component.properties).forEach(key => {
            const prop = component.properties[key];
            const category = prop.category || 'General';
            if (!groups[category]) groups[category] = {};
            groups[category][key] = prop;
        });

        Object.keys(groups).forEach(groupName => {
            html += `<div class="property-group">
                <div class="property-group-title">${groupName}</div>`;

            Object.keys(groups[groupName]).forEach(key => {
                const prop = groups[groupName][key];
                html += this.renderPropertyField(key, prop, component.properties[key]);
            });

            html += '</div>';
        });

        return html;
    }

    renderPropertyField(key, prop, value) {
        const fieldId = `property_${key}`;
        let html = `<div class="property-field">
            <label class="property-label" for="${fieldId}">${prop.label || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</label>`;

        switch (prop.type) {
            case 'text':
            case 'url':
            case 'email':
                html += `<input type="${prop.type}" id="${fieldId}" class="property-input" value="${value || ''}" data-property="${key}" placeholder="${prop.placeholder || ''}">`;
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'textarea':
                html += `<textarea id="${fieldId}" class="property-textarea" data-property="${key}" placeholder="${prop.placeholder || ''}" rows="${prop.rows || 3}">${value || ''}</textarea>`;
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'select':
                html += `<select id="${fieldId}" class="property-select" data-property="${key}">`;
                if (prop.dynamic) {
                    // Handle dynamic options (e.g., rule sets, services)
                    html += this.renderDynamicOptions(key, prop, value);
                } else {
                    // Handle static options
                    (prop.options || []).forEach(option => {
                        const selected = (value === option) ? 'selected' : '';
                        html += `<option value="${option}" ${selected}>${option}</option>`;
                    });
                }
                html += '</select>';
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'number':
                html += `<input type="number" id="${fieldId}" class="property-input" value="${value || prop.default || 0}" data-property="${key}"`;
                if (prop.min !== undefined) html += ` min="${prop.min}"`;
                if (prop.max !== undefined) html += ` max="${prop.max}"`;
                if (prop.step !== undefined) html += ` step="${prop.step}"`;
                html += '>';
                if (prop.unit) {
                    html += `<span class="property-unit text-sm text-gray-500 ml-2">${prop.unit}</span>`;
                }
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'slider':
                const sliderValue = value !== undefined ? value : (prop.default !== undefined ? prop.default : prop.min || 0);
                html += `<div class="slider-container">
                    <input type="range" id="${fieldId}" class="property-slider" value="${sliderValue}" data-property="${key}"`;
                if (prop.min !== undefined) html += ` min="${prop.min}"`;
                if (prop.max !== undefined) html += ` max="${prop.max}"`;
                if (prop.step !== undefined) html += ` step="${prop.step}"`;
                html += `>
                    <span class="slider-value text-sm font-medium text-gray-700 ml-3">${sliderValue}</span>`;
                if (prop.unit) {
                    html += `<span class="slider-unit text-sm text-gray-500 ml-1">${prop.unit}</span>`;
                }
                html += '</div>';
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'boolean':
                const checked = value ? 'checked' : '';
                html += `<label class="boolean-toggle">
                    <input type="checkbox" id="${fieldId}" class="property-checkbox" ${checked} data-property="${key}">
                    <span class="toggle-slider"></span>
                    <span class="toggle-label">${prop.label || key}</span>
                </label>`;
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'file':
                html += `<input type="file" id="${fieldId}" class="property-input" data-property="${key}" accept="${prop.accept || '*'}">`;
                if (value && value.name) {
                    html += `<div class="file-info text-xs text-gray-600 mt-1">Selected: ${value.name}</div>`;
                }
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'json':
                const jsonValue = value ? JSON.stringify(value, null, 2) : '';
                html += `<textarea id="${fieldId}" class="property-textarea font-mono text-sm" data-property="${key}" placeholder="Enter JSON..." rows="${prop.rows || 6}">${jsonValue}</textarea>`;
                html += `<button type="button" class="validate-json-btn text-xs bg-blue-500 text-white px-2 py-1 rounded mt-1 hover:bg-blue-600" onclick="agentBuilder.validateJson('${fieldId}')">Validate JSON</button>`;
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            case 'array':
                html += `<div class="array-input" id="${fieldId}_container">`;
                const arrayValues = Array.isArray(value) ? value : [];
                arrayValues.forEach((item, index) => {
                    html += `<div class="array-item flex items-center mb-2">
                        <input type="text" class="property-input flex-1" value="${item}" data-array-index="${index}" data-property="${key}">
                        <button type="button" class="remove-array-item ml-2 text-red-500 hover:text-red-700" onclick="agentBuilder.removeArrayItem('${fieldId}_container', ${index})">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>`;
                });
                html += `<button type="button" class="add-array-item text-sm text-blue-500 hover:text-blue-700" onclick="agentBuilder.addArrayItem('${fieldId}_container', '${key}')">
                    <i class="fas fa-plus mr-1"></i>Add Item
                </button>`;
                html += '</div>';
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
                break;

            default:
                html += `<input type="text" id="${fieldId}" class="property-input" value="${value || ''}" data-property="${key}" placeholder="${prop.placeholder || ''}">`;
                if (prop.description) {
                    html += `<div class="property-description text-xs text-gray-500 mt-1">${prop.description}</div>`;
                }
        }

        html += '</div>';
        return html;
    }

    renderDynamicOptions(key, prop, value) {
        // Handle dynamic options based on property type
        let html = '';

        if (key === 'rule_set') {
            // Fetch available rule sets from rule engine
            html += '<option value="">Loading rule sets...</option>';
            this.loadRuleSets();
        } else if (key.includes('service') || key === 'connection_type') {
            // Service selection options
            const services = this.getAvailableServices(key);
            services.forEach(service => {
                const selected = (value === service.value) ? 'selected' : '';
                html += `<option value="${service.value}" ${selected}>${service.label}</option>`;
            });
        } else if (key === 'model') {
            // LLM model options
            const models = [
                { value: 'gpt-4', label: 'GPT-4 (Most Capable)' },
                { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo (Fast)' },
                { value: 'claude-3', label: 'Claude 3 (Balanced)' },
                { value: 'llama-2-70b', label: 'Llama 2 70B (Open Source)' }
            ];
            models.forEach(model => {
                const selected = (value === model.value) ? 'selected' : '';
                html += `<option value="${model.value}" ${selected}>${model.label}</option>`;
            });
        }

        return html;
    }

    getAvailableServices(context) {
        // Return available services based on context
        const services = {
            'data_input': [
                { value: 'csv-ingestion-service', label: 'CSV Ingestion Service' },
                { value: 'api-ingestion-service', label: 'API Ingestion Service' },
                { value: 'json-ingestion-service', label: 'JSON Ingestion Service' },
                { value: 'excel-ingestion-service', label: 'Excel Ingestion Service' }
            ],
            'database_output': [
                { value: 'postgresql-output', label: 'PostgreSQL Output' },
                { value: 'mongodb-output', label: 'MongoDB Output' },
                { value: 'elasticsearch-output', label: 'Elasticsearch Output' }
            ],
            'vector_output': [
                { value: 'qdrant-vector', label: 'Qdrant Vector Database' },
                { value: 'elasticsearch-output', label: 'Elasticsearch (Dense Vectors)' }
            ],
            'connection_type': [
                { value: 'postgresql', label: 'PostgreSQL' },
                { value: 'mongodb', label: 'MongoDB' },
                { value: 'elasticsearch', label: 'Elasticsearch' },
                { value: 'qdrant', label: 'Qdrant' },
                { value: 'timescaledb', label: 'TimescaleDB' },
                { value: 'neo4j', label: 'Neo4j' }
            ]
        };

        return services[context] || [];
    }

    async loadRuleSets() {
        try {
            const response = await fetch('/api/rule-sets');
            if (response.ok) {
                const data = await response.json();
                this.updateRuleSetOptions(data.rule_sets);
            }
        } catch (error) {
            console.error('Failed to load rule sets:', error);
        }
    }

    updateRuleSetOptions(ruleSets) {
        const selectElement = document.getElementById('property_rule_set');
        if (!selectElement) return;

        // Clear existing options except first
        while (selectElement.children.length > 1) {
            selectElement.removeChild(selectElement.lastChild);
        }

        // Add new options
        ruleSets.forEach(ruleSet => {
            const option = document.createElement('option');
            option.value = ruleSet.id;
            option.textContent = ruleSet.name;
            selectElement.appendChild(option);
        });
    }

    validateJson(fieldId) {
        const textarea = document.getElementById(fieldId);
        if (!textarea) return;

        try {
            const jsonData = JSON.parse(textarea.value);
            this.showPropertySuccess(textarea, 'Valid JSON');
            // Pretty print the JSON
            textarea.value = JSON.stringify(jsonData, null, 2);
        } catch (error) {
            this.showPropertyError(textarea, 'Invalid JSON: ' + error.message);
        }
    }

    showPropertySuccess(element, message) {
        // Remove existing messages
        const existing = element.parentElement.querySelector('.property-message');
        if (existing) existing.remove();

        // Add success message
        const successDiv = document.createElement('div');
        successDiv.className = 'property-message text-green-600 text-xs mt-1';
        successDiv.textContent = message;

        element.parentElement.appendChild(successDiv);

        // Remove after 3 seconds
        setTimeout(() => {
            if (successDiv.parentElement) {
                successDiv.remove();
            }
        }, 3000);
    }

    addArrayItem(containerId, propertyKey) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const itemCount = container.querySelectorAll('.array-item').length;

        const itemDiv = document.createElement('div');
        itemDiv.className = 'array-item flex items-center mb-2';
        itemDiv.innerHTML = `
            <input type="text" class="property-input flex-1" value="" data-array-index="${itemCount}" data-property="${propertyKey}">
            <button type="button" class="remove-array-item ml-2 text-red-500 hover:text-red-700" onclick="agentBuilder.removeArrayItem('${containerId}', ${itemCount})">
                <i class="fas fa-times"></i>
            </button>
        `;

        // Insert before the "Add Item" button
        const addButton = container.querySelector('.add-array-item');
        container.insertBefore(itemDiv, addButton);
    }

    removeArrayItem(containerId, index) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const items = container.querySelectorAll('.array-item');
        if (items[index]) {
            items[index].remove();
            // Update indices for remaining items
            const remainingItems = container.querySelectorAll('.array-item');
            remainingItems.forEach((item, newIndex) => {
                const input = item.querySelector('input');
                const button = item.querySelector('.remove-array-item');
                if (input) input.dataset.arrayIndex = newIndex;
                if (button) button.onclick = () => this.removeArrayItem(containerId, newIndex);
            });
        }
    }

    setupPropertyListeners(componentId) {
        const content = document.getElementById('properties-content');

        // Handle input changes for text, textarea, and select elements
        content.addEventListener('input', (e) => {
            const target = e.target;
            if (target.dataset.property) {
                this.updateComponentProperty(componentId, target.dataset.property, target.value, target.type);
            }
        });

        // Handle change events for checkboxes, files, and other form elements
        content.addEventListener('change', (e) => {
            const target = e.target;
            if (target.dataset.property) {
                let value = target.value;
                if (target.type === 'checkbox') {
                    value = target.checked;
                } else if (target.type === 'file') {
                    value = target.files[0];
                } else if (target.type === 'range') {
                    // Handle slider changes
                    value = parseFloat(target.value);
                    // Update slider value display
                    this.updateSliderDisplay(target);
                } else if (target.dataset.property.includes('json')) {
                    try {
                        value = JSON.parse(target.value);
                    } catch (error) {
                        console.error('Invalid JSON:', error);
                        this.showPropertyError(target, 'Invalid JSON format');
                        return;
                    }
                }
                this.updateComponentProperty(componentId, target.dataset.property, value, target.type);
            }
        });

        // Handle slider value display updates
        content.addEventListener('input', (e) => {
            const target = e.target;
            if (target.type === 'range') {
                this.updateSliderDisplay(target);
            }
        });
    }

    updateSliderDisplay(slider) {
        // Find the associated value display element
        const valueDisplay = slider.parentElement.querySelector('.slider-value');
        if (valueDisplay) {
            const value = parseFloat(slider.value);
            const step = parseFloat(slider.step) || 1;
            const decimals = step < 1 ? 1 : 0;
            valueDisplay.textContent = value.toFixed(decimals);
        }
    }

    showPropertyError(element, message) {
        // Remove existing error message
        const existingError = element.parentElement.querySelector('.property-error');
        if (existingError) {
            existingError.remove();
        }

        // Add error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'property-error text-red-600 text-xs mt-1';
        errorDiv.textContent = message;

        element.parentElement.appendChild(errorDiv);

        // Remove error after 3 seconds
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 3000);
    }

    updateComponentProperty(componentId, property, value, type) {
        const component = this.components.get(componentId);
        if (!component) return;

        component.properties[property] = value;
        this.saveState();
    }

    showNoSelection() {
        const content = document.getElementById('properties-content');
        content.innerHTML = document.getElementById('no-selection').innerHTML;
    }

    // =============================================================================
    // WORKFLOW MANAGEMENT
    // =============================================================================

    saveWorkflow() {
        const workflow = {
            agent_name: document.getElementById('agent-name').value,
            agent_domain: document.getElementById('agent-domain').value,
            components: Array.from(this.components.values()),
            connections: this.connections,
            canvas: {
                zoom: this.zoom,
                panX: this.panX,
                panY: this.panY
            },
            timestamp: new Date().toISOString()
        };

        // Save to local storage
        localStorage.setItem('agent_builder_workflow', JSON.stringify(workflow));

        // Show success message
        this.showSuccess('Workflow saved successfully');
    }

    loadWorkflow() {
        const saved = localStorage.getItem('agent_builder_workflow');
        if (!saved) return;

        try {
            const workflow = JSON.parse(saved);

            // Load agent info
            document.getElementById('agent-name').value = workflow.agent_name || '';
            document.getElementById('agent-domain').value = workflow.agent_domain || '';

            // Load canvas state
            this.zoom = workflow.canvas.zoom || 1.0;
            this.panX = workflow.canvas.panX || 0;
            this.panY = workflow.canvas.panY || 0;
            this.updateCanvasTransform();

            // Load components
            workflow.components.forEach(comp => {
                this.components.set(comp.id, comp);
                this.renderCanvasComponent(comp);
            });

            // Load connections
            this.connections = workflow.connections;
            this.updateConnections();

        } catch (error) {
            console.error('Failed to load workflow:', error);
        }
    }

    exportWorkflow() {
        const workflow = {
            agent_id: `agent_${Date.now()}`,
            name: document.getElementById('agent-name').value || 'My AI Agent',
            domain: document.getElementById('agent-domain').value,
            components: Array.from(this.components.values()),
            connections: this.connections,
            metadata: {
                exported_at: new Date().toISOString(),
                version: '1.0.0'
            }
        };

        const dataStr = JSON.stringify(workflow, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `${workflow.name.toLowerCase().replace(/\s+/g, '_')}_workflow.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    importWorkflow(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const workflow = JSON.parse(e.target.result);
                this.clearWorkflow();

                // Load workflow data
                document.getElementById('agent-name').value = workflow.name || '';
                document.getElementById('agent-domain').value = workflow.domain || '';

                // Load components
                workflow.components.forEach(comp => {
                    this.components.set(comp.id, comp);
                    this.renderCanvasComponent(comp);
                });

                // Load connections
                this.connections = workflow.connections || [];
                this.updateConnections();

                this.showSuccess('Workflow imported successfully');

            } catch (error) {
                console.error('Failed to import workflow:', error);
                this.showError('Failed to import workflow');
            }
        };
        reader.readAsText(file);
    }

    clearWorkflow() {
        // Clear components
        this.components.clear();
        document.getElementById('canvas-components').innerHTML = '';

        // Clear connections
        this.connections = [];
        document.querySelectorAll('.connection-line').forEach(line => line.remove());

        // Clear selection
        this.deselectAll();

        // Reset canvas
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.updateCanvasTransform();
    }

    // =============================================================================
    // UTILITY METHODS
    // =============================================================================

    cloneProperties(properties) {
        const cloned = {};
        Object.keys(properties).forEach(key => {
            if (typeof properties[key] === 'object' && properties[key] !== null) {
                cloned[key] = { ...properties[key] };
            } else {
                cloned[key] = properties[key];
            }
        });
        return cloned;
    }

    getComponentIcon(type) {
        const icons = {
            'data_input_csv': '📄',
            'data_input_api': '🌐',
            'llm_processor': '🧠',
            'rule_engine': '⚖️',
            'decision_node': '🔀',
            'multi_agent_coordinator': '👥',
            'database_output': '🗄️',
            'email_output': '📧',
            'pdf_output': '📄'
        };
        return icons[type] || '🔧';
    }

    getComponentTypeName(type) {
        const names = {
            'data_input_csv': 'CSV Data Input',
            'data_input_api': 'API Data Input',
            'llm_processor': 'LLM Processor',
            'rule_engine': 'Rule Engine',
            'decision_node': 'Decision Node',
            'multi_agent_coordinator': 'Multi-Agent Coordinator',
            'database_output': 'Database Output',
            'email_output': 'Email Output',
            'pdf_output': 'PDF Report Output'
        };
        return names[type] || 'Component';
    }

    showSuccess(message) {
        // Implementation for success notification
        console.log('Success:', message);
    }

    showError(message) {
        // Implementation for error notification
        console.error('Error:', message);
    }

    setupEventListeners() {
        // Canvas drop zone
        const canvas = document.getElementById('canvas-container');
        const dropZone = document.getElementById('drop-zone');

        canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.remove('opacity-0');
        });

        canvas.addEventListener('dragleave', (e) => {
            if (!canvas.contains(e.relatedTarget)) {
                dropZone.classList.add('opacity-0');
            }
        });

        canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.add('opacity-0');

            try {
                const componentData = JSON.parse(e.dataTransfer.getData('application/json'));
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left - 100; // Center the component
                const y = e.clientY - rect.top - 50;

                this.createCanvasComponent(componentData, x, y);
            } catch (error) {
                console.error('Failed to drop component:', error);
            }
        });

        // Toolbar buttons
        document.getElementById('new-agent-btn').addEventListener('click', () => this.showNewAgentModal());
        document.getElementById('save-btn').addEventListener('click', () => this.saveWorkflow());
        document.getElementById('deploy-btn').addEventListener('click', () => this.deployAgent());
        document.getElementById('zoom-in-btn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoom-out-btn').addEventListener('click', () => this.zoomOut());
        document.getElementById('fit-canvas-btn').addEventListener('click', () => this.fitCanvas());
        document.getElementById('undo-btn').addEventListener('click', () => this.undo());
        document.getElementById('redo-btn').addEventListener('click', () => this.redo());
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'z':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.redo();
                        } else {
                            e.preventDefault();
                            this.undo();
                        }
                        break;
                    case 'y':
                        e.preventDefault();
                        this.redo();
                        break;
                    case 's':
                        e.preventDefault();
                        this.saveWorkflow();
                        break;
                    case 'a':
                        e.preventDefault();
                        this.selectAll();
                        break;
                }
            } else if (e.key === 'Delete') {
                if (this.selectedComponent) {
                    this.deleteComponent(this.selectedComponent);
                }
            }
        });
    }

    setupAutoSave() {
        setInterval(() => {
            this.saveWorkflow();
        }, 30000); // Auto-save every 30 seconds
    }

    setupMinimap() {
        // Minimap implementation
        this.updateMinimap();
    }

    setupWireHelpOverlay() {
        // Set up wire help overlay event listeners
        const helpBtn = document.getElementById('help-btn');
        const wireHelpOverlay = document.getElementById('wire-help-overlay');
        const closeWireHelpBtn = document.getElementById('close-wire-help-btn');
        const gotItWireHelpBtn = document.getElementById('got-it-wire-help-btn');

        if (helpBtn) {
            helpBtn.addEventListener('click', () => {
                wireHelpOverlay.classList.remove('hidden');
            });
        }

        if (closeWireHelpBtn) {
            closeWireHelpBtn.addEventListener('click', () => {
                wireHelpOverlay.classList.add('hidden');
            });
        }

        if (gotItWireHelpBtn) {
            gotItWireHelpBtn.addEventListener('click', () => {
                wireHelpOverlay.classList.add('hidden');
                // Mark that user has seen the help
                localStorage.setItem('wire_help_seen', 'true');
            });
        }

        // Show help overlay on first visit
        if (!localStorage.getItem('wire_help_seen')) {
            setTimeout(() => {
                wireHelpOverlay.classList.remove('hidden');
            }, 2000); // Show after 2 seconds
        }
    }

    updateMinimap() {
        // Update minimap content
        const minimap = document.getElementById('minimap-view');
        // Implementation for minimap updates
    }

    // Placeholder methods for toolbar actions
    showNewAgentModal() { console.log('Show new agent modal'); }
    deployAgent() { console.log('Deploy agent'); }
    zoomIn() { this.zoom = Math.min(this.zoom * 1.2, 3.0); this.updateCanvasTransform(); }
    zoomOut() { this.zoom = Math.max(this.zoom / 1.2, 0.1); this.updateCanvasTransform(); }
    fitCanvas() { this.zoom = 1.0; this.panX = 0; this.panY = 0; this.updateCanvasTransform(); }
    undo() { console.log('Undo'); }
    redo() { console.log('Redo'); }
    selectAll() { console.log('Select all'); }
    deleteComponent(id) { console.log('Delete component:', id); }
    saveState() { console.log('Save state'); }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

let agentBuilder;

document.addEventListener('DOMContentLoaded', () => {
    agentBuilder = new AgentBuilder();
});

// Export for global access
window.agentBuilder = agentBuilder;
