#!/bin/bash

# Agentic Platform - Stop Script
# This script gracefully stops all platform services and cleans up resources

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
    echo -e "${BLUE}         🛑 AGENTIC PLATFORM SHUTDOWN${NC}"
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
        print_info "Cannot stop services if Docker is not running"
        exit 1
    fi

    print_success "Docker is running"
}

# Function to check if any platform services are running
check_running_services() {
    print_step "Checking for running platform services..."

    local running_containers=$(docker-compose -p $PROJECT_NAME ps -q 2>/dev/null | wc -l)

    if [ "$running_containers" -eq 0 ]; then
        print_warning "No platform services are currently running"
        print_info "Nothing to stop. Platform may already be shut down."
        return 1
    else
        print_info "Found $running_containers running service(s)"
        return 0
    fi
}

# Function to show current service status
show_current_status() {
    print_step "Current service status:"

    echo ""
    echo -e "${CYAN}Running Services:${NC}"
    docker-compose -p $PROJECT_NAME ps --format "table {{.Name}}\t{{.Service}}\t{{.State}}\t{{.Ports}}"

    echo ""
    echo -e "${CYAN}Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" $(docker-compose -p $PROJECT_NAME ps -q) 2>/dev/null || echo "Unable to retrieve resource usage"
}

# Function to gracefully stop services
stop_services() {
    print_step "Stopping platform services gracefully..."
    print_info "This may take a moment as services shut down properly..."

    # Stop services with timeout
    if timeout 300 docker-compose -p $PROJECT_NAME down --timeout 60; then
        print_success "Services stopped successfully"
    else
        print_warning "Some services didn't stop gracefully within timeout"
        print_info "Forcing shutdown of remaining services..."
        docker-compose -p $PROJECT_NAME down --timeout 30 -v || true
    fi
}

# Function to clean up resources
cleanup_resources() {
    print_step "Cleaning up platform resources..."

    # Remove stopped containers
    local stopped_containers=$(docker container ls -a --filter "label=com.docker.compose.project=$PROJECT_NAME" --filter "status=exited" -q | wc -l)

    if [ "$stopped_containers" -gt 0 ]; then
        print_info "Removing $stopped_containers stopped container(s)..."
        docker container rm $(docker container ls -a --filter "label=com.docker.compose.project=$PROJECT_NAME" --filter "status=exited" -q) >/dev/null 2>&1 || true
        print_success "Containers cleaned up"
    else
        print_info "No stopped containers to clean up"
    fi

    # Remove unused networks
    print_info "Cleaning up unused networks..."
    docker network prune -f >/dev/null 2>&1 || true

    # Remove dangling images (optional - only if user confirms)
    local dangling_images=$(docker images -f "dangling=true" -q | wc -l)
    if [ "$dangling_images" -gt 0 ]; then
        print_info "Found $dangling_images dangling image(s)"
        read -p "Remove dangling images? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker image prune -f >/dev/null 2>&1 || true
            print_success "Dangling images removed"
        else
            print_info "Skipping image cleanup"
        fi
    fi
}

# Function to show final status
show_final_status() {
    print_step "Final status check..."

    echo ""
    echo -e "${CYAN}Remaining Resources:${NC}"

    # Check for any remaining platform containers
    local remaining=$(docker container ls -a --filter "label=com.docker.compose.project=$PROJECT_NAME" -q | wc -l)
    if [ "$remaining" -eq 0 ]; then
        print_success "All platform containers removed"
    else
        print_warning "$remaining platform container(s) still exist"
        echo -e "${YELLOW}Remaining containers:${NC}"
        docker container ls -a --filter "label=com.docker.compose.project=$PROJECT_NAME" --format "table {{.Names}}\t{{.Status}}"
    fi

    # Check disk usage
    echo ""
    echo -e "${CYAN}Disk Usage:${NC}"
    docker system df

    # Show system info
    echo ""
    echo -e "${CYAN}System Resources:${NC}"
    echo -e "  Docker Version: $(docker --version)"
    echo -e "  Docker Compose: $(docker-compose --version 2>/dev/null || docker compose version 2>/dev/null || echo 'Not available')"
}

# Function to provide restart instructions
show_restart_info() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}✅ PLATFORM SHUTDOWN COMPLETE${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo -e "${YELLOW}🔄 To restart the platform:${NC}"
    echo -e "  ${WHITE}•${NC} Run: ${GREEN}./start-platform.sh${NC}"
    echo ""
    echo -e "${YELLOW}📊 To monitor resources:${NC}"
    echo -e "  ${WHITE}•${NC} View Docker: ${CYAN}docker system df${NC}"
    echo -e "  ${WHITE}•${NC} Clean more: ${CYAN}docker system prune${NC}"
    echo ""
    echo -e "${YELLOW}📁 To view logs:${NC}"
    echo -e "  ${WHITE}•${NC} Platform logs: ${CYAN}docker-compose -p $PROJECT_NAME logs${NC}"
    echo ""
}

# Function to handle graceful shutdown on interrupt
cleanup_on_exit() {
    echo ""
    print_warning "Shutdown interrupted. Attempting graceful cleanup..."
    docker-compose -p $PROJECT_NAME down --timeout 30 >/dev/null 2>&1 || true
    print_info "Emergency cleanup completed. Exiting..."
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

    # Check if services are running
    if ! check_running_services; then
        show_restart_info
        exit 0
    fi

    echo ""

    # Show current status
    show_current_status

    echo ""
    print_warning "About to stop Agentic Platform services..."
    read -p "Continue with shutdown? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Shutdown cancelled by user"
        exit 0
    fi

    echo ""

    # Stop services
    stop_services

    echo ""

    # Clean up resources
    cleanup_resources

    echo ""

    # Show final status
    show_final_status

    # Show restart info
    show_restart_info

    print_success "Agentic Platform shutdown completed successfully!"
}

# Run main function
main "$@"
