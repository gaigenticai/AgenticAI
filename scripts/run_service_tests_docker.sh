#!/bin/bash

# Docker-Based Service Testing Script
# This script provides an easy way to run Docker-based tests for individual services
# during development and CI/CD pipelines.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agentic-platform"
DOCKER_COMPOSE_FILE="docker-compose.yml"
TEST_TIMEOUT="600"  # 10 minutes

# Function to print colored output
print_info() {
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

# Function to check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi

    print_success "Docker is available"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi

    print_success "Docker Compose is available"
}

# Function to start test infrastructure
start_test_infrastructure() {
    print_info "Starting test infrastructure..."

    # Start databases and message queues needed for testing
    docker-compose -f "$DOCKER_COMPOSE_FILE" -p "$PROJECT_NAME" up -d \
        postgresql_ingestion \
        redis_ingestion \
        rabbitmq \
        qdrant_vector

    # Wait for services to be ready
    print_info "Waiting for test infrastructure to be ready..."
    sleep 30

    print_success "Test infrastructure started"
}

# Function to run service tests in Docker
run_service_tests() {
    local service_name="$1"
    local test_image="${service_name}:test"

    print_info "Running Docker tests for service: $service_name"

    # Build test image
    print_info "Building test image: $test_image"
    if ! docker build -f "services/${service_name//-/_}/Dockerfile.test" \
                     -t "$test_image" \
                     "services/${service_name//-/_}/"; then
        print_error "Failed to build test image for $service_name"
        return 1
    fi

    # Run tests in container
    print_info "Running tests in Docker container..."

    local container_name="test_${service_name}_$(date +%s)"

    if docker run --name "$container_name" \
                 --network "${PROJECT_NAME}_default" \
                 --env-file .env \
                 --rm \
                 -v "$(pwd)/test-results:/app/test-results" \
                 "$test_image"; then
        print_success "Tests passed for $service_name"
        return 0
    else
        print_error "Tests failed for $service_name"
        # Show test logs
        docker logs "$container_name" 2>/dev/null || true
        return 1
    fi
}

# Function to run all service tests
run_all_service_tests() {
    print_info "Running Docker tests for all services..."

    local services=(
        "ingestion-coordinator"
        "output-coordinator"
        "agent-orchestrator"
        "plugin-registry"
        "workflow-engine"
        "vector-ui"
        "brain-factory"
        "data-encryption"
        "port-manager"
    )

    local failed_services=()
    local passed_services=()

    for service in "${services[@]}"; do
        if [ -d "services/${service//-/_}" ]; then
            if run_service_tests "$service"; then
                passed_services+=("$service")
            else
                failed_services+=("$service")
            fi
        else
            print_warning "Service directory not found: services/${service//-/_}"
        fi
    done

    # Print summary
    echo
    echo "========================================"
    echo "DOCKER SERVICE TESTING SUMMARY"
    echo "========================================"

    echo "Passed services (${#passed_services[@]}):"
    for service in "${passed_services[@]}"; do
        echo -e "  ${GREEN}✓${NC} $service"
    done

    if [ ${#failed_services[@]} -gt 0 ]; then
        echo
        echo "Failed services (${#failed_services[@]}):"
        for service in "${failed_services[@]}"; do
            echo -e "  ${RED}✗${NC} $service"
        done
    fi

    echo
    echo "Total: $((${#passed_services[@]} + ${#failed_services[@]}))"
    echo "Passed: ${#passed_services[@]}"
    echo "Failed: ${#failed_services[@]}"

    if [ ${#failed_services[@]} -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# Function to cleanup test infrastructure
cleanup_test_infrastructure() {
    print_info "Cleaning up test infrastructure..."

    docker-compose -f "$DOCKER_COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans

    print_success "Test infrastructure cleaned up"
}

# Function to show usage
show_usage() {
    cat << EOF
Docker-Based Service Testing Script

Usage:
    $0 [OPTIONS] [SERVICE_NAME]

Options:
    --all, -a          Run tests for all services
    --infrastructure   Start/stop test infrastructure only
    --cleanup          Cleanup test infrastructure
    --help, -h         Show this help message

Arguments:
    SERVICE_NAME       Name of specific service to test (e.g., ingestion-coordinator)

Examples:
    $0 --all                          # Test all services
    $0 ingestion-coordinator         # Test specific service
    $0 --infrastructure              # Start test infrastructure
    $0 --cleanup                     # Cleanup test infrastructure

Environment Variables:
    PROJECT_NAME      Docker Compose project name (default: agentic-platform)
    TEST_TIMEOUT      Test timeout in seconds (default: 600)
    DOCKER_COMPOSE_FILE Docker Compose file path (default: docker-compose.yml)

EOF
}

# Main execution
main() {
    # Parse arguments
    local run_all=false
    local infrastructure_only=false
    local cleanup_only=false
    local service_name=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --all|-a)
                run_all=true
                shift
                ;;
            --infrastructure)
                infrastructure_only=true
                shift
                ;;
            --cleanup)
                cleanup_only=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                service_name="$1"
                shift
                ;;
        esac
    done

    # Prerequisites
    check_docker
    check_docker_compose

    # Handle different modes
    if [ "$cleanup_only" = true ]; then
        cleanup_test_infrastructure
        exit 0
    fi

    if [ "$infrastructure_only" = true ]; then
        start_test_infrastructure
        print_info "Test infrastructure is running. Press Ctrl+C to stop."
        trap cleanup_test_infrastructure INT TERM
        while true; do sleep 1; done
    fi

    # Create test results directory
    mkdir -p test-results

    # Start test infrastructure
    start_test_infrastructure

    # Run tests
    local exit_code=0
    if [ "$run_all" = true ]; then
        if ! run_all_service_tests; then
            exit_code=1
        fi
    elif [ -n "$service_name" ]; then
        if ! run_service_tests "$service_name"; then
            exit_code=1
        fi
    else
        print_error "No service specified. Use --all or specify a service name."
        show_usage
        exit_code=1
    fi

    # Cleanup
    cleanup_test_infrastructure

    exit $exit_code
}

# Run main function
main "$@"
