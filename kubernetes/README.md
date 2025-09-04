# Agentic Platform - Kubernetes Deployment

This directory contains all the necessary Kubernetes manifests to deploy the Agentic Platform in a production-ready Kubernetes cluster.

## 🏗️ Architecture Overview

The Kubernetes deployment includes:

- **18 Microservices** deployed as individual deployments
- **Infrastructure Services**: PostgreSQL, Redis, RabbitMQ, MinIO
- **Monitoring Stack**: Prometheus, Grafana, Jaeger
- **Security**: OAuth2/OIDC, TLS encryption, RBAC
- **Auto-scaling**: Horizontal Pod Autoscalers
- **Storage**: Persistent Volumes for data persistence
- **Networking**: Ingress with SSL termination
- **Security**: Network Policies and RBAC

## 📁 Directory Structure

```
kubernetes/
├── namespace.yaml              # Namespace definition
├── configmap.yaml             # Application configuration
├── secret.yaml                # Sensitive configuration
├── pvc.yaml                   # Persistent Volume Claims
├── postgresql-deployment.yaml # Database deployment
├── redis-deployment.yaml      # Cache deployment
├── rabbitmq-deployment.yaml   # Message queue deployment
├── minio-deployment.yaml      # Object storage deployment
├── app-services-deployment.yaml # Core application services
├── monitoring-deployment.yaml # Monitoring stack
├── ingress.yaml               # External access configuration
├── hpa.yaml                   # Auto-scaling policies
├── rbac.yaml                  # Role-based access control
├── network-policies.yaml      # Network security policies
└── README.md                  # This file
```

## 🚀 Quick Start Deployment

### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Helm (for ingress controller)
- cert-manager (for SSL certificates)

### 1. Install Prerequisites

```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx

# Install cert-manager for SSL certificates
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### 2. Deploy Namespace and Infrastructure

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Deploy ConfigMaps and Secrets
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# Deploy Persistent Volume Claims
kubectl apply -f pvc.yaml
```

### 3. Deploy Infrastructure Services

```bash
# Deploy database and cache
kubectl apply -f postgresql-deployment.yaml
kubectl apply -f redis-deployment.yaml

# Deploy message queue
kubectl apply -f rabbitmq-deployment.yaml

# Deploy object storage
kubectl apply -f minio-deployment.yaml
```

### 4. Deploy Application Services

```bash
# Deploy core application services
kubectl apply -f app-services-deployment.yaml

# Deploy monitoring stack
kubectl apply -f monitoring-deployment.yaml
```

### 5. Configure Networking and Security

```bash
# Apply RBAC
kubectl apply -f rbac.yaml

# Apply Network Policies
kubectl apply -f network-policies.yaml

# Configure Ingress
kubectl apply -f ingress.yaml

# Configure Auto-scaling
kubectl apply -f hpa.yaml
```

## 🔧 Configuration

### Environment Variables

Update the following configurations before deployment:

#### ConfigMap (`configmap.yaml`)
- Database connection details
- Redis configuration
- RabbitMQ settings
- MinIO endpoints
- Jaeger configuration

#### Secrets (`secret.yaml`)
- Database passwords
- JWT secrets
- MinIO credentials
- TLS certificates
- AWS credentials (for backup)

### SSL/TLS Certificates

The deployment uses cert-manager for automatic SSL certificate management. Update the Ingress configuration with your domain names.

### Storage Classes

Modify the PVC configurations to use your cluster's storage classes:

```yaml
storageClassName: your-storage-class
```

## 📊 Monitoring and Observability

### Accessing Monitoring Tools

```bash
# Grafana (Visualization)
kubectl port-forward svc/grafana 3000:3000 -n agentic-platform
# Access: http://localhost:3000 (admin/admin)

# Prometheus (Metrics)
kubectl port-forward svc/prometheus 9090:9090 -n agentic-platform
# Access: http://localhost:9090

# Jaeger (Distributed Tracing)
kubectl port-forward svc/jaeger 16686:16686 -n agentic-platform
# Access: http://localhost:16686
```

### Pre-configured Dashboards

The deployment includes several pre-configured Grafana dashboards:

- **System Overview**: Overall platform health and performance
- **Data Pipeline**: Ingestion, processing, and output metrics
- **Security Monitoring**: Authentication and authorization metrics
- **Infrastructure**: Database, cache, and message queue metrics

## 🔒 Security Configuration

### RBAC Roles

Three RBAC roles are configured:

- **agentic-platform-admin**: Full access to all resources
- **agentic-platform-developer**: Read/write access to application resources
- **agentic-platform-viewer**: Read-only access to resources

### Network Security

Network Policies restrict traffic between services:

- Default deny-all policy
- Allow ingress from ingress controller
- Service-to-service communication restrictions
- External access controls

### Secrets Management

Sensitive data is stored in Kubernetes Secrets:

- Database credentials
- JWT signing keys
- TLS certificates
- API keys

## 📈 Scaling and Performance

### Horizontal Pod Autoscaling

HPAs are configured for:

- **Ingestion Coordinator**: CPU (70%), Memory (80%), HTTP requests
- **GraphQL API**: CPU (60%), Memory (75%), GraphQL queries
- **Data Lake**: CPU (70%), Memory (80%)

### Resource Limits

Default resource requests and limits:

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

## 🔄 Backup and Recovery

### Automated Backups

The platform includes automated backup schedules:

- **Database**: Daily backups at 2 AM
- **Cache**: Hourly snapshots
- **Configuration**: Daily configuration backups
- **Object Storage**: Weekly full backups

### Manual Backups

```bash
# Trigger manual backup
kubectl exec -it deployment/backup-orchestration -n agentic-platform -- \
  python -c "from backup_service import create_backup; create_backup('database')"
```

## 🚨 Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod status
kubectl get pods -n agentic-platform

# Check pod logs
kubectl logs -f pod/pod-name -n agentic-platform

# Check events
kubectl get events -n agentic-platform
```

#### Service Unavailable
```bash
# Check service endpoints
kubectl get endpoints -n agentic-platform

# Check service configuration
kubectl describe svc service-name -n agentic-platform
```

#### Ingress Issues
```bash
# Check ingress status
kubectl get ingress -n agentic-platform

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/nginx-ingress-controller
```

### Health Checks

All services include health check endpoints:

```bash
# Check service health
curl http://service-name:port/health
```

## 📚 API Reference

### Service Endpoints

- **GraphQL API**: `https://api.agentic-platform.com/graphql`
- **Authentication**: `https://api.agentic-platform.com/auth`
- **REST API**: `https://api.agentic-platform.com/api/v1`
- **Dashboard**: `https://dashboard.agentic-platform.com`
- **Grafana**: `https://grafana.agentic-platform.com`
- **Jaeger**: `https://jaeger.agentic-platform.com`

## 🔧 Maintenance

### Updating Services

```bash
# Update deployment image
kubectl set image deployment/service-name container=new-image:tag -n agentic-platform

# Rolling restart
kubectl rollout restart deployment/service-name -n agentic-platform
```

### Database Maintenance

```bash
# Connect to database
kubectl exec -it deployment/postgresql-ingestion -n agentic-platform -- psql -U agentic_user

# Backup database
kubectl exec -it deployment/postgresql-ingestion -n agentic-platform -- \
  pg_dump -U agentic_user agentic_ingestion > backup.sql
```

### Monitoring Maintenance

```bash
# Clean up old metrics
kubectl exec -it deployment/prometheus -n agentic-platform -- \
  find /prometheus -name "*.db" -mtime +30 -delete
```

## 🎯 Production Checklist

- [ ] Update domain names in Ingress
- [ ] Configure SSL certificates
- [ ] Set production database credentials
- [ ] Configure backup storage (AWS S3, GCP, etc.)
- [ ] Set up monitoring alerts
- [ ] Configure log aggregation
- [ ] Set up CI/CD pipelines
- [ ] Configure disaster recovery
- [ ] Perform security audit
- [ ] Load testing
- [ ] Documentation review

## 📞 Support

For issues and support:

1. Check the troubleshooting section
2. Review service logs
3. Check monitoring dashboards
4. Contact the platform team

---

## 🚀 Deployment Script

For automated deployment, use the provided deployment script:

```bash
#!/bin/bash
# deploy.sh

echo "🚀 Deploying Agentic Platform to Kubernetes..."

# Apply manifests in order
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pvc.yaml
kubectl apply -f postgresql-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f rabbitmq-deployment.yaml
kubectl apply -f minio-deployment.yaml
kubectl apply -f app-services-deployment.yaml
kubectl apply -f monitoring-deployment.yaml
kubectl apply -f rbac.yaml
kubectl apply -f network-policies.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

echo "✅ Deployment completed!"
echo "🌐 Access your platform at:"
echo "  - API: https://api.agentic-platform.com"
echo "  - Dashboard: https://dashboard.agentic-platform.com"
echo "  - Grafana: https://grafana.agentic-platform.com"
echo "  - Jaeger: https://jaeger.agentic-platform.com"
```

**🎉 Your Agentic Platform is now production-ready on Kubernetes!**
