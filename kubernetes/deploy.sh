#!/bin/bash
# Agentic Platform - Kubernetes Deployment Script
# This script deploys the entire platform to a Kubernetes cluster

set -e  # Exit on any error

echo "🚀 Agentic Platform - Kubernetes Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi

    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        print_warning "Helm is not installed. Ingress controller installation will be skipped."
    fi

    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please configure kubectl."
        exit 1
    fi

    print_success "Prerequisites check completed"
}

# Function to install ingress controller
install_ingress() {
    if command -v helm &> /dev/null; then
        print_status "Installing NGINX Ingress Controller..."

        # Add helm repo if not exists
        helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx 2>/dev/null || true
        helm repo update

        # Install ingress controller
        helm upgrade --install nginx-ingress ingress-nginx/ingress-nginx \
            --namespace ingress-nginx \
            --create-namespace \
            --set controller.service.type=LoadBalancer \
            --wait

        print_success "NGINX Ingress Controller installed"
    else
        print_warning "Skipping ingress controller installation (Helm not available)"
    fi
}

# Function to install cert-manager
install_cert_manager() {
    if command -v helm &> /dev/null; then
        print_status "Installing cert-manager..."

        # Add helm repo if not exists
        helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true
        helm repo update

        # Install cert-manager
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

        # Wait for cert-manager to be ready
        kubectl wait --for=condition=available --timeout=300s deployment/cert-manager -n cert-manager

        print_success "cert-manager installed"
    else
        print_warning "Skipping cert-manager installation (Helm not available)"
    fi
}

# Function to deploy namespace
deploy_namespace() {
    print_status "Creating namespace..."
    kubectl apply -f namespace.yaml
    print_success "Namespace created"
}

# Function to deploy configuration
deploy_config() {
    print_status "Deploying configuration..."
    kubectl apply -f configmap.yaml
    kubectl apply -f secret.yaml
    kubectl apply -f pvc.yaml
    print_success "Configuration deployed"
}

# Function to deploy infrastructure
deploy_infrastructure() {
    print_status "Deploying infrastructure services..."

    # Deploy database
    kubectl apply -f postgresql-deployment.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/postgresql-ingestion -n agentic-platform

    # Deploy cache
    kubectl apply -f redis-deployment.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/redis-ingestion -n agentic-platform

    # Deploy message queue
    kubectl apply -f rabbitmq-deployment.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/rabbitmq -n agentic-platform

    # Deploy object storage
    kubectl apply -f minio-deployment.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/minio-bronze -n agentic-platform

    print_success "Infrastructure services deployed"
}

# Function to deploy application services
deploy_applications() {
    print_status "Deploying application services..."

    # Deploy core services
    kubectl apply -f app-services-deployment.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/ingestion-coordinator -n agentic-platform

    print_success "Application services deployed"
}

# Function to deploy monitoring
deploy_monitoring() {
    print_status "Deploying monitoring stack..."

    kubectl apply -f monitoring-deployment.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n agentic-platform

    print_success "Monitoring stack deployed"
}

# Function to deploy security
deploy_security() {
    print_status "Deploying security configurations..."

    kubectl apply -f rbac.yaml
    kubectl apply -f network-policies.yaml

    print_success "Security configurations deployed"
}

# Function to deploy networking
deploy_networking() {
    print_status "Deploying networking configuration..."

    kubectl apply -f ingress.yaml
    kubectl apply -f hpa.yaml

    print_success "Networking configuration deployed"
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."

    # Check if all pods are running
    local total_pods=$(kubectl get pods -n agentic-platform --no-headers | wc -l)
    local running_pods=$(kubectl get pods -n agentic-platform --no-headers | grep Running | wc -l)

    if [ "$total_pods" -eq "$running_pods" ]; then
        print_success "All pods are running ($running_pods/$total_pods)"
    else
        print_warning "Some pods are not running ($running_pods/$total_pods)"
        kubectl get pods -n agentic-platform
    fi

    # Check services
    local services=$(kubectl get svc -n agentic-platform --no-headers | wc -l)
    print_success "$services services deployed"

    # Check ingress
    local ingress=$(kubectl get ingress -n agentic-platform --no-headers | wc -l)
    if [ "$ingress" -gt 0 ]; then
        print_success "Ingress configured"
    else
        print_warning "No ingress found"
    fi
}

# Function to show access information
show_access_info() {
    echo ""
    echo "🌐 Access Information"
    echo "===================="

    # Get ingress IP or hostname
    local ingress_ip=$(kubectl get svc nginx-ingress-ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

    if [ -n "$ingress_ip" ]; then
        echo "External IP: $ingress_ip"
        echo ""
        echo "Access URLs:"
        echo "  - API:        https://api.agentic-platform.com"
        echo "  - Dashboard:  https://dashboard.agentic-platform.com"
        echo "  - Grafana:    https://grafana.agentic-platform.com"
        echo "  - Jaeger:     https://jaeger.agentic-platform.com"
    else
        echo "Configure DNS to point to your ingress controller IP:"
        echo "  - api.agentic-platform.com"
        echo "  - dashboard.agentic-platform.com"
        echo "  - grafana.agentic-platform.com"
        echo "  - jaeger.agentic-platform.com"
    fi

    echo ""
    echo "🔧 Service Ports (for local access):"
    echo "  - GraphQL API:    8100"
    echo "  - OAuth2/OIDC:    8093"
    echo "  - Data Lake:      8090"
    echo "  - Grafana:        3000"
    echo "  - Jaeger:         16686"
    echo "  - Prometheus:     9090"

    echo ""
    echo "📊 Monitoring Commands:"
    echo "  # Port forward services for local access"
    echo "  kubectl port-forward svc/grafana 3000:3000 -n agentic-platform"
    echo "  kubectl port-forward svc/jaeger 16686:16686 -n agentic-platform"
    echo "  kubectl port-forward svc/graphql-api 8100:8100 -n agentic-platform"
}

# Main deployment function
main() {
    echo ""
    print_status "Starting Agentic Platform deployment..."

    check_prerequisites
    install_ingress
    install_cert_manager
    deploy_namespace
    deploy_config
    deploy_infrastructure
    deploy_applications
    deploy_monitoring
    deploy_security
    deploy_networking
    verify_deployment

    echo ""
    print_success "🎉 Agentic Platform deployment completed successfully!"
    echo ""

    show_access_info

    echo ""
    echo "📚 Next Steps:"
    echo "  1. Update DNS records to point to your ingress IP"
    echo "  2. Configure SSL certificates with cert-manager"
    echo "  3. Update secrets with production credentials"
    echo "  4. Configure backup storage (AWS S3, GCP, etc.)"
    echo "  5. Set up monitoring alerts and notifications"
    echo "  6. Review security policies and network policies"
    echo "  7. Perform load testing and performance tuning"
    echo ""
    echo "📖 For detailed documentation, see kubernetes/README.md"
}

# Handle command line arguments
case "${1:-}" in
    "check")
        check_prerequisites
        ;;
    "prereqs")
        install_ingress
        install_cert_manager
        ;;
    "infrastructure")
        deploy_namespace
        deploy_config
        deploy_infrastructure
        ;;
    "applications")
        deploy_applications
        ;;
    "monitoring")
        deploy_monitoring
        ;;
    "security")
        deploy_security
        ;;
    "networking")
        deploy_networking
        ;;
    "verify")
        verify_deployment
        ;;
    *)
        main
        ;;
esac
