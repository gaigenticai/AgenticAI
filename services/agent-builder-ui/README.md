# Agent Builder UI Service

The Agent Builder UI Service provides a modern, no-code visual interface for creating and configuring AI agents in the Agentic Brain Platform. It features drag-and-drop workflow creation, component configuration panels, real-time validation, and seamless integration with the backend services.

## Features

### 🎨 **Visual Workflow Builder**
- **Drag-and-Drop Interface**: Intuitive component placement on canvas
- **Visual Connections**: Wire components together with connection lines
- **Component Palette**: Pre-built components organized by category
- **Canvas Zoom & Pan**: Navigate large workflows with smooth controls
- **Grid System**: Precise component alignment and positioning

### 🔧 **Component Management**
- **Data Input Components**: CSV, API, and database connectors
- **Processing Components**: LLM processors and rule engines
- **Decision Components**: Conditional logic and branching
- **Output Components**: Database, email, and PDF generators
- **Coordination Components**: Multi-agent workflow management

### ⚙️ **Configuration & Properties**
- **Real-time Property Editing**: Configure components without code
- **Dynamic Form Fields**: Text, numbers, sliders, dropdowns, checkboxes
- **JSON Configuration**: Advanced configuration for complex setups
- **Validation Feedback**: Immediate validation of component settings
- **Property Templates**: Pre-configured settings for common use cases

### 💾 **Workflow Management**
- **Auto-Save**: Automatic workflow saving every 30 seconds
- **Import/Export**: JSON-based workflow serialization
- **Version History**: Track changes and revert modifications
- **Template Library**: Pre-built workflow templates
- **Collaboration**: Multi-user workflow editing support

### 🔗 **Integration Features**
- **Backend Integration**: Real-time communication with all platform services
- **API Endpoints**: RESTful APIs for workflow operations
- **Validation Pipeline**: Automated workflow validation and error checking
- **Deployment Pipeline**: One-click agent deployment to production
- **Monitoring Integration**: Real-time metrics and performance tracking

## User Interface

### Main Components

#### **Component Palette** (Left Sidebar)
- **Categories**: Data Input, Processing, Decision, Coordination, Output
- **Component Cards**: Visual representation with icons and descriptions
- **Drag & Drop**: Intuitive component placement on canvas
- **Search & Filter**: Quick component discovery

#### **Canvas Area** (Main Workspace)
- **Grid Background**: Precise component alignment
- **Zoom Controls**: Scale from 10% to 300%
- **Pan Support**: Smooth navigation of large workflows
- **Connection Lines**: Visual data flow representation
- **Minimap**: Overview of entire workflow

#### **Properties Panel** (Right Sidebar)
- **Component Settings**: Configure selected component properties
- **Dynamic Forms**: Context-sensitive form fields
- **Validation Feedback**: Real-time error checking
- **Advanced Options**: Expert configuration settings

#### **Toolbar** (Top)
- **Agent Settings**: Name, domain, and description
- **File Operations**: Save, export, import workflows
- **Edit Controls**: Undo, redo, zoom, fit canvas
- **Deployment**: One-click agent deployment

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo last action |
| `Ctrl+Y` | Redo last action |
| `Ctrl+S` | Save workflow |
| `Ctrl+A` | Select all components |
| `Delete` | Remove selected component |
| `Mouse Wheel` | Zoom in/out |
| `Space + Drag` | Pan canvas |

## Component Types

### Data Input Components

#### **CSV Data Input**
```json
{
  "service_url": "http://csv-ingestion-service:8001",
  "file_path": "/data/input.csv",
  "delimiter": ",",
  "has_header": true
}
```

#### **API Data Input**
```json
{
  "url": "https://api.example.com/data",
  "method": "GET",
  "headers": {"Authorization": "Bearer token"},
  "params": {"limit": 100}
}
```

### Processing Components

#### **LLM Processor**
```json
{
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 1000,
  "prompt_template": "Analyze the following data: {input}",
  "system_message": "You are a helpful AI assistant."
}
```

#### **Rule Engine**
```json
{
  "rule_set": "fraud_detection_rules",
  "evaluation_mode": "all",
  "fail_on_error": false
}
```

### Decision Components

#### **Decision Node**
```json
{
  "condition_type": "threshold",
  "threshold_value": 0.8,
  "comparison_operator": ">=",
  "true_label": "High Risk",
  "false_label": "Low Risk"
}
```

### Output Components

#### **Database Output**
```json
{
  "connection_type": "postgresql",
  "table_name": "processed_data",
  "connection_string": "postgresql://user:pass@host:5432/db",
  "insert_mode": "upsert",
  "batch_size": 100
}
```

#### **Email Output**
```json
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "username": "agent@company.com",
  "from_email": "agent@company.com",
  "to_emails": ["manager@company.com"],
  "subject_template": "Agent Report: {agent_name}",
  "body_template": "Processing complete. Results: {results}"
}
```

## API Endpoints

### Workflow Operations
- `GET /api/components` - Get available components
- `GET /api/templates` - Get workflow templates
- `POST /api/templates/{id}/instantiate` - Instantiate template
- `POST /api/workflows/validate` - Validate workflow
- `POST /api/agents/deploy` - Deploy agent

### UI Endpoints
- `GET /` - Main Agent Builder interface
- `GET /editor` - Workflow editor
- `GET /templates/{id}` - Template-specific editor
- `GET /health` - Health check

## Configuration

### Environment Variables

#### **Service Configuration**
- `AGENT_BUILDER_UI_HOST` - Service host (default: 0.0.0.0)
- `AGENT_BUILDER_UI_PORT` - Service port (default: 8300)

#### **Backend Integration**
- `TEMPLATE_STORE_URL` - Template store service URL
- `WORKFLOW_ENGINE_URL` - Workflow engine service URL
- `BRAIN_FACTORY_URL` - Brain factory service URL
- `DEPLOYMENT_PIPELINE_URL` - Deployment pipeline service URL
- `PLUGIN_REGISTRY_URL` - Plugin registry service URL

#### **UI Configuration**
- `MAX_CANVAS_WIDTH` - Maximum canvas width (default: 2000)
- `MAX_CANVAS_HEIGHT` - Maximum canvas height (default: 1500)
- `AUTO_SAVE_INTERVAL` - Auto-save interval in seconds (default: 30)

#### **Security**
- `REQUIRE_AUTH` - Enable authentication (default: false)
- `JWT_SECRET` - JWT secret for authentication

## Usage Examples

### Creating a Simple Agent

1. **Start New Agent**
   - Click "New Agent" button
   - Enter agent name and domain
   - Select appropriate template or start blank

2. **Add Components**
   - Drag "CSV Data Input" from palette to canvas
   - Drag "LLM Processor" and connect to data input
   - Drag "Database Output" and connect to processor

3. **Configure Properties**
   - Click on CSV component to configure file path
   - Configure LLM processor with model and prompt
   - Set up database connection for output

4. **Connect Components**
   - Click and drag from output port to input port
   - Visual connection lines show data flow

5. **Validate and Deploy**
   - Click "Validate" to check workflow
   - Click "Deploy" to create the agent

### Advanced Workflow Creation

```javascript
// Programmatic workflow creation
const workflow = {
  name: "Advanced Processing Agent",
  components: [
    {
      id: "csv_input",
      type: "data_input_csv",
      x: 100,
      y: 100,
      properties: {
        file_path: "/data/input.csv",
        delimiter: ","
      }
    },
    {
      id: "llm_processor",
      type: "llm_processor",
      x: 300,
      y: 100,
      properties: {
        model: "gpt-4",
        temperature: 0.7,
        prompt_template: "Analyze this data: {input}"
      }
    }
  ],
  connections: [
    {
      source: { componentId: "csv_input", portIndex: 0 },
      target: { componentId: "llm_processor", portIndex: 0 }
    }
  ]
};
```

## Integration Points

### Template Store Service
- Load pre-built agent templates
- Save custom workflow templates
- Template versioning and management

### Workflow Engine Service
- Real-time workflow validation
- Component dependency checking
- Execution path analysis

### Brain Factory Service
- Convert visual workflows to agent configurations
- Generate agent code from visual design
- Component-to-service mapping

### Deployment Pipeline Service
- Automated agent deployment
- Environment-specific configuration
- Rollback and versioning support

### Plugin Registry Service
- Dynamic component loading
- Plugin management and updates
- Component marketplace integration

## Browser Support

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

## Performance Optimization

### Canvas Performance
- **Virtual Scrolling**: Efficient rendering of large canvases
- **Component Culling**: Only render visible components
- **Connection Optimization**: Efficient connection line rendering
- **Memory Management**: Automatic cleanup of unused resources

### Network Optimization
- **Lazy Loading**: Components loaded on demand
- **Caching**: API responses cached for performance
- **Compression**: Gzip compression for all responses
- **CDN Integration**: Static assets served via CDN

## Security Features

### Authentication & Authorization
- **JWT Integration**: Secure user authentication
- **Role-Based Access**: Component and workflow permissions
- **Session Management**: Secure session handling

### Data Protection
- **Input Validation**: Comprehensive input sanitization
- **XSS Protection**: Cross-site scripting prevention
- **CSRF Protection**: Cross-site request forgery prevention
- **Content Security Policy**: Restrictive CSP headers

### Audit & Compliance
- **Action Logging**: All user actions logged
- **Change Tracking**: Workflow modification history
- **Compliance Reporting**: GDPR and security compliance

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AGENT_BUILDER_UI_PORT=8300
export TEMPLATE_STORE_URL=http://localhost:8203

# Run development server
python app.py
```

### Docker Development

```bash
# Build the service
docker build -t agent-builder-ui .

# Run with docker-compose
docker-compose up agent-builder-ui
```

### Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run UI tests (requires browser)
pytest tests/ui/
```

## Troubleshooting

### Common Issues

#### **Components Not Loading**
- Check network connectivity to backend services
- Verify API endpoints are accessible
- Check browser console for JavaScript errors

#### **Canvas Not Responding**
- Clear browser cache and cookies
- Disable browser extensions
- Check for JavaScript errors in console

#### **Workflow Not Saving**
- Verify local storage is enabled
- Check file permissions for export
- Ensure workflow JSON is valid

#### **Deployment Failing**
- Validate workflow configuration
- Check backend service health
- Review deployment logs

### Debug Mode

Enable debug mode by setting:

```bash
export AGENT_BUILDER_UI_DEBUG=true
```

This provides detailed logging and error information.

## Contributing

### Code Style
- Follow PEP 8 for Python code
- Use ESLint for JavaScript code
- Maintain comprehensive documentation
- Write unit tests for all features

### Component Development
1. Create component definition in `components/`
2. Add visual representation to palette
3. Implement property panel configuration
4. Add validation logic
5. Write comprehensive tests

### UI Enhancement
1. Maintain responsive design principles
2. Ensure accessibility compliance (WCAG 2.1)
3. Optimize for performance
4. Test across all supported browsers

## License

This service is part of the Agentic Brain Platform and follows the same licensing terms.

## Support

For support and questions:
- **Documentation**: `/docs` endpoint provides API documentation
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and feedback
- **Email**: Contact the development team for enterprise support
