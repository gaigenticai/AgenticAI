/**
 * Professional Vector UI JavaScript Implementation
 * Enhanced with modern UX patterns, error handling, and interactive features
 */

class VectorUI {
    constructor() {
        this.currentView = 'dashboard';
        this.apiBaseUrl = window.location.origin;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadDashboard();
        this.setupRealTimeUpdates();
        this.initializeTooltips();
    }

    bindEvents() {
        // Navigation events
        document.querySelectorAll('[data-nav]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const view = e.target.closest('[data-nav]').dataset.nav;
                this.navigateTo(view);
            });
        });

        // Form submissions
        document.addEventListener('submit', (e) => {
            if (e.target.matches('.vector-form')) {
                e.preventDefault();
                this.handleFormSubmission(e.target);
            }
        });

        // Modal close events
        document.addEventListener('click', (e) => {
            if (e.target.matches('.modal-close') || e.target.matches('.modal-overlay')) {
                this.closeModal();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    navigateTo(view) {
        this.currentView = view;
        this.showLoading();

        // Update active navigation
        document.querySelectorAll('[data-nav]').forEach(btn => {
            btn.classList.remove('bg-primary-100', 'text-primary-900');
            btn.classList.add('text-gray-600', 'hover:text-gray-900');
        });

        const activeBtn = document.querySelector(`[data-nav="${view}"]`);
        if (activeBtn) {
            activeBtn.classList.remove('text-gray-600', 'hover:text-gray-900');
            activeBtn.classList.add('bg-primary-100', 'text-primary-900');
        }

        // Load view content
        this.loadView(view);
    }

    async loadView(view) {
        try {
            const response = await fetch(`/${view}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const html = await response.text();
            document.getElementById('main-content').innerHTML = html;

            // Re-bind events for new content
            this.bindDynamicEvents();

            this.hideLoading();
            this.showToast(`Loaded ${view} view`, 'success');

        } catch (error) {
            console.error('Error loading view:', error);
            this.showError('Failed to load view. Please try again.');
            this.hideLoading();
        }
    }

    bindDynamicEvents() {
        // Re-bind form events
        document.querySelectorAll('.vector-form').forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleFormSubmission(form);
            });
        });

        // Bind API action buttons
        document.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.closest('[data-action]').dataset.action;
                this.handleAction(action, e.target);
            });
        });
    }

    async handleFormSubmission(form) {
        const formData = new FormData(form);
        const action = form.dataset.action;

        // Use shared UI utils for better error handling
        if (window.uiUtils) {
            return window.uiUtils.handleAsyncOperation(async () => {
                let response;
                if (form.method.toLowerCase() === 'post') {
                    if (form.enctype === 'multipart/form-data') {
                        response = await fetch(form.action, {
                            method: 'POST',
                            body: formData
                        });
                    } else {
                        const data = Object.fromEntries(formData);
                        response = await fetch(form.action, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(data)
                        });
                    }
                }

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
                }

                const result = await response.json();

                if (result.error) {
                    throw new Error(result.error);
                }

                // Refresh current view if needed
                if (form.dataset.refresh) {
                    this.loadView(this.currentView);
                }

                return result;
            }, {
                loadingMessage: 'Processing your request...',
                successMessage: 'Operation completed successfully!',
                errorMessage: 'Operation failed. Please try again.'
            });
        }

        // Fallback to original implementation
        this.showLoading('Processing...');
        try {
            let response;
            if (form.method.toLowerCase() === 'post') {
                if (form.enctype === 'multipart/form-data') {
                    response = await fetch(form.action, {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    const data = Object.fromEntries(formData);
                    response = await fetch(form.action, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                }
            }

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            this.showToast('Operation completed successfully!', 'success');

            // Refresh current view if needed
            if (form.dataset.refresh) {
                this.loadView(this.currentView);
            }

            return result;
        } catch (error) {
            console.error('Form submission error:', error);
            this.showError(error.message || 'Operation failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async handleAction(action, element) {
        this.showLoading();

        try {
            switch (action) {
                case 'generate-embeddings':
                    await this.generateEmbeddings();
                    break;
                case 'search-vectors':
                    await this.searchVectors();
                    break;
                case 'index-documents':
                    await this.indexDocuments();
                    break;
                case 'view-collections':
                    await this.viewCollections();
                    break;
                default:
                    console.warn('Unknown action:', action);
            }
        } catch (error) {
            console.error('Action error:', error);
            this.showError(error.message || 'Action failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async generateEmbeddings() {
        const text = prompt('Enter text to generate embeddings for:');
        if (!text) return;

        const response = await fetch('/api/embeddings/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: [text] })
        });

        if (!response.ok) throw new Error('Failed to generate embeddings');

        const result = await response.json();
        this.showResultModal('Embeddings Generated', JSON.stringify(result, null, 2));
    }

    async searchVectors() {
        const query = prompt('Enter search query:');
        if (!query) return;

        const response = await fetch('/api/vectors/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, limit: 10 })
        });

        if (!response.ok) throw new Error('Search failed');

        const result = await response.json();
        this.showSearchResults(result);
    }

    async viewCollections() {
        const response = await fetch('/api/vectors/collections');
        if (!response.ok) throw new Error('Failed to load collections');

        const result = await response.json();
        this.showCollectionsModal(result);
    }

    showSearchResults(results) {
        const modal = this.createModal('Search Results', `
            <div class="space-y-4">
                ${results.results.map(result => `
                    <div class="border rounded-lg p-4 hover:bg-gray-50">
                        <div class="font-medium text-gray-900">${result.text}</div>
                        <div class="text-sm text-gray-600 mt-1">Score: ${result.score.toFixed(4)}</div>
                        ${result.metadata ? `<div class="text-xs text-gray-500 mt-2">${JSON.stringify(result.metadata)}</div>` : ''}
                    </div>
                `).join('')}
            </div>
        `);
        document.body.appendChild(modal);
    }

    showCollectionsModal(data) {
        const modal = this.createModal('Vector Collections',
            `<div class="text-center">
                <div class="text-2xl font-bold text-gray-900 mb-2">${data.count}</div>
                <div class="text-gray-600">Collections</div>
                ${data.collections.length > 0 ? `
                    <div class="mt-4">
                        <div class="text-sm font-medium text-gray-700 mb-2">Available Collections:</div>
                        <div class="flex flex-wrap gap-2">
                            ${data.collections.map(collection => `
                                <span class="px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm">
                                    ${collection}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>`
        );
        document.body.appendChild(modal);
    }

    showResultModal(title, content) {
        const modal = this.createModal(title,
            `<pre class="bg-gray-100 p-4 rounded-lg text-sm overflow-x-auto">${content}</pre>`
        );
        document.body.appendChild(modal);
    }

    createModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="modal-content bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-96 overflow-y-auto">
                <div class="flex items-center justify-between p-6 border-b">
                    <h3 class="text-lg font-semibold text-gray-900">${title}</h3>
                    <button class="modal-close text-gray-400 hover:text-gray-600">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="p-6">${content}</div>
            </div>
        `;
        return modal;
    }

    closeModal() {
        const modal = document.querySelector('.modal-overlay');
        if (modal) {
            modal.remove();
        }
    }

    showLoading(message = 'Loading...') {
        let overlay = document.querySelector('.loading-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner mb-4"></div>
                    <div class="text-gray-900 font-medium">${message}</div>
                </div>
            `;
            document.body.appendChild(overlay);
        }
        overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast fixed top-4 right-4 px-4 py-2 rounded-lg text-white z-50 fade-in`;
        toast.style.backgroundColor = type === 'success' ? '#10b981' :
                                   type === 'error' ? '#ef4444' :
                                   type === 'warning' ? '#f59e0b' : '#6b7280';

        toast.innerHTML = `
            <div class="flex items-center gap-2">
                <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation' : 'info'}-circle"></i>
                <span>${message}</span>
            </div>
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    setupRealTimeUpdates() {
        // Update metrics every 30 seconds
        setInterval(() => {
            this.updateMetrics();
        }, 30000);

        // Check system health every 60 seconds
        setInterval(() => {
            this.checkHealth();
        }, 60000);
    }

    async updateMetrics() {
        try {
            const response = await fetch('/api/metrics');
            if (response.ok) {
                const metrics = await response.json();
                this.updateMetricDisplays(metrics);
            }
        } catch (error) {
            console.warn('Failed to update metrics:', error);
        }
    }

    updateMetricDisplays(metrics) {
        // Update metric cards with new data
        Object.entries(metrics).forEach(([key, value]) => {
            const element = document.getElementById(`metric-${key}`);
            if (element) {
                element.textContent = this.formatNumber(value);
            }
        });
    }

    async checkHealth() {
        try {
            const response = await fetch('/health');
            const isHealthy = response.ok;

            const statusIndicator = document.getElementById('system-status');
            if (statusIndicator) {
                statusIndicator.className = `status-indicator ${isHealthy ? 'status-online' : 'status-offline'}`;
                statusIndicator.innerHTML = `
                    <div class="w-2 h-2 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-red-500'}"></div>
                    <span>${isHealthy ? 'Online' : 'Offline'}</span>
                `;
            }
        } catch (error) {
            console.warn('Health check failed:', error);
        }
    }

    initializeTooltips() {
        // Add tooltips to elements with data-tooltip attribute
        document.querySelectorAll('[data-tooltip]').forEach(element => {
            element.classList.add('tooltip');
            const tooltip = document.createElement('span');
            tooltip.className = 'tooltip-text';
            tooltip.textContent = element.dataset.tooltip;
            element.appendChild(tooltip);
        });
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    loadDashboard() {
        this.navigateTo('dashboard');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.vectorUI = new VectorUI();
});

// Export for global access
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VectorUI;
}
