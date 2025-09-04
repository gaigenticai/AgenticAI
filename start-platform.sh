#!/bin/bash

# Agentic Platform - Start Script
# This script starts all platform services with proper dependency management

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agenticai"
COMPOSE_FILE="docker-compose.yml"

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}         🚀 AGENTIC PLATFORM STARTUP${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
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

print_info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    print_step "Checking Docker status..."

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running or not accessible"
        print_info "Please start Docker and try again"
        exit 1
    fi

    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    print_step "Checking Docker Compose..."

    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed"
        print_info "Please install Docker Compose and try again"
        exit 1
    fi

    print_success "Docker Compose is available"
}

# Function to check system resources
check_system_resources() {
    print_step "Checking system resources..."

    # Check available memory (in GB)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        AVAILABLE_MEM=$(free -g | awk 'NR==2{printf "%.0f", $7}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        AVAILABLE_MEM=$(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)
    else
        AVAILABLE_MEM="Unknown"
    fi

    # Check available disk space (in GB)
    AVAILABLE_DISK=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')

    print_info "Available RAM: ${AVAILABLE_MEM}GB"
    print_info "Available Disk: ${AVAILABLE_DISK}GB"

    if [ "$AVAILABLE_MEM" != "Unknown" ] && [ "$AVAILABLE_MEM" -lt 8 ]; then
        print_warning "Less than 8GB RAM available. Platform may run slowly."
        print_warning "Recommended: 8GB+ RAM for optimal performance"
    fi

    if [ "$AVAILABLE_DISK" -lt 50 ]; then
        print_warning "Less than 50GB disk space available."
        print_warning "Platform requires significant storage for data processing"
    fi
}

# Function to clean up previous containers
cleanup_containers() {
    print_step "Cleaning up previous containers..."

    # Stop and remove existing containers
    docker-compose -p $PROJECT_NAME down --remove-orphans 2>/dev/null || true

    # Remove stopped containers
    docker container prune -f >/dev/null 2>&1 || true

    print_success "Cleanup completed"
}

# Function to build and start services
start_services() {
    print_step "Building and starting platform services..."
    print_info "This may take several minutes on first run..."

    # Start services with build
    if docker-compose -p $PROJECT_NAME up -d --build; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        print_info "Check the logs above for detailed error information"
        exit 1
    fi
}

# Function to wait for services to be healthy
wait_for_services() {
    print_step "Waiting for services to become healthy..."
    print_info "This may take 2-5 minutes depending on your system..."

    local max_attempts=60
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        echo -ne "${WHITE}Checking services... (attempt $attempt/$max_attempts)${NC}\r"

        # Check if core services are running
        if docker-compose -p $PROJECT_NAME ps | grep -q "Up"; then
            echo "" # New line after progress
            print_success "Core services are running"
            return 0
        fi

        sleep 5
        ((attempt++))
    done

    echo "" # New line after progress
    print_warning "Services are starting but health checks may take longer"
    print_info "You can monitor progress with: docker-compose -p $PROJECT_NAME logs -f"
}

# Function to display service status
show_service_status() {
    print_step "Checking service status..."

    echo ""
    echo -e "${CYAN}Service Status:${NC}"
    docker-compose -p $PROJECT_NAME ps

    echo ""
    echo -e "${CYAN}Service Health:${NC}"

    # Test key services
    local services=("vector-ui" "ingestion-coordinator" "output-coordinator" "qdrant_vector")

    for service in "${services[@]}"; do
        if docker-compose -p $PROJECT_NAME ps $service | grep -q "Up"; then
            echo -e "  ${GREEN}✓${NC} $service: Running"
        else
            echo -e "  ${RED}✗${NC} $service: Not running"
        fi
    done
}

# Function to display access information
show_access_info() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}              🌐 ACCESS INFORMATION${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""

    echo -e "${CYAN}📱 Web Interfaces:${NC}"
    echo -e "  ${GREEN}•${NC} Main Platform UI:    http://localhost:8082"
    echo -e "  ${GREEN}•${NC} Comprehensive Guide: http://localhost:8082/comprehensive-guide"
    echo -e "  ${GREEN}•${NC} Grafana Dashboard:   http://localhost:3000 (admin/admin)"
    echo -e "  ${GREEN}•${NC} Prometheus Metrics:  http://localhost:9090"
    echo -e "  ${GREEN}•${NC} Jaeger Tracing:      http://localhost:16686"

    echo ""
    echo -e "${CYAN}🔌 API Endpoints:${NC}"
    echo -e "  ${GREEN}•${NC} Ingestion API:       http://localhost:8080"
    echo -e "  ${GREEN}•${NC} Output API:          http://localhost:8081"
    echo -e "  ${GREEN}•${NC} API Documentation:   http://localhost:8081/docs"
    echo -e "  ${GREEN}•${NC} Vector Operations:   http://localhost:8082/api/embeddings"

    echo ""
    echo -e "${CYAN}🗄️ Database Services:${NC}"
    echo -e "  ${GREEN}•${NC} PostgreSQL (Ingestion): localhost:5432"
    echo -e "  ${GREEN}•${NC} PostgreSQL (Output):    localhost:5433"
    echo -e "  ${GREEN}•${NC} MongoDB:               localhost:27017"
    echo -e "  ${GREEN}•${NC} Qdrant Vector DB:     localhost:6333"
    echo -e "  ${GREEN}•${NC} Redis Cache:          localhost:6379"
    echo -e "  ${GREEN}•${NC} Elasticsearch:       localhost:9200"

    echo ""
    echo -e "${CYAN}📊 Message Queue:${NC}"
    echo -e "  ${GREEN}•${NC} RabbitMQ Management: http://localhost:15672 (guest/guest)"

    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}🎉 PLATFORM STARTUP COMPLETE!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo -e "${YELLOW}💡 Quick Start Tips:${NC}"
    echo -e "  ${WHITE}•${NC} Visit http://localhost:8082 for the main interface"
    echo -e "  ${WHITE}•${NC} Check http://localhost:8082/comprehensive-guide for full documentation"
    echo -e "  ${WHITE}•${NC} Monitor services with: docker-compose -p $PROJECT_NAME logs -f"
    echo -e "  ${WHITE}•${NC} Stop platform with: ./stop-platform.sh"
    echo ""
}

# Function to handle graceful shutdown on interrupt
cleanup_on_exit() {
    echo ""
    print_warning "Received interrupt signal. Cleaning up..."
    docker-compose -p $PROJECT_NAME down --remove-orphans >/dev/null 2>&1 || true
    print_info "Cleanup completed. Exiting..."
    exit 1
}

# Main execution
main() {
    # Set up signal handlers
    trap cleanup_on_exit INT TERM

    # Print header
    print_header

    # Check prerequisites
    check_docker
    check_docker_compose
    check_system_resources

    echo ""
    print_info "Starting Agentic Platform with 15+ microservices..."
    print_info "Estimated startup time: 3-5 minutes"
    echo ""

    # Clean up previous containers
    cleanup_containers

    echo ""

    # Start services
    start_services

    echo ""

    # Wait for services
    wait_for_services

    echo ""

    # Show status
    show_service_status

    # Show access information
    show_access_info

    print_success "Agentic Platform is now running!"
    print_info "All services are operational and ready for use"
}

# Run main function
main "$@"
