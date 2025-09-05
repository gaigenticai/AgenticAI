/**
 * Shared UI Utilities for Professional User Experience
 * Provides common functionality for error handling, feedback, and UX enhancements
 */

class UIUtils {
    constructor() {
        this.init();
    }

    init() {
        this.createGlobalElements();
        this.bindGlobalEvents();
    }

    createGlobalElements() {
        // Create toast container if it doesn't exist
        if (!document.getElementById('toast-container')) {
            const toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.className = 'fixed top-4 right-4 z-50 space-y-2';
            document.body.appendChild(toastContainer);
        }

        // Create loading overlay if it doesn't exist
        if (!document.getElementById('loading-overlay')) {
            const loadingOverlay = document.createElement('div');
            loadingOverlay.id = 'loading-overlay';
            loadingOverlay.className = 'loading-overlay';
            loadingOverlay.innerHTML = `
                <div class="bg-white rounded-lg shadow-xl p-6 flex items-center gap-4">
                    <div class="loading-spinner"></div>
                    <div class="text-gray-900 font-medium" id="loading-message">Loading...</div>
                </div>
            `;
            document.body.appendChild(loadingOverlay);
        }

        // Create modal container if it doesn't exist
        if (!document.getElementById('modal-container')) {
            const modalContainer = document.createElement('div');
            modalContainer.id = 'modal-container';
            document.body.appendChild(modalContainer);
        }
    }

    bindGlobalEvents() {
        // Global error handling
        window.addEventListener('error', (e) => {
            console.error('Global error:', e.error);
            this.showToast('An unexpected error occurred', 'error');
        });

        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
            this.showToast('An unexpected error occurred', 'error');
        });

        // Network status monitoring
        window.addEventListener('online', () => {
            this.showToast('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.showToast('Connection lost', 'warning');
        });
    }

    // ===== LOADING STATES =====
    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            const messageEl = document.getElementById('loading-message');
            if (messageEl) messageEl.textContent = message;
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    // ===== TOAST NOTIFICATIONS =====
    showToast(message, type = 'info', duration = 3000) {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type} max-w-sm`;
        toast.innerHTML = `
            <div class="flex items-center gap-3">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
                <button class="ml-auto text-white/70 hover:text-white" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times text-sm"></i>
                </button>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Auto remove after duration
        setTimeout(() => {
            if (toast.parentElement) {
                toast.style.animation = 'slideOutRight 0.3s ease-out';
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    }

    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    // ===== MODAL SYSTEM =====
    showModal(title, content, options = {}) {
        const modalContainer = document.getElementById('modal-container');
        if (!modalContainer) return;

        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content max-w-2xl w-full mx-4">
                <div class="flex items-center justify-between p-6 border-b">
                    <h3 class="text-lg font-semibold text-gray-900">${title}</h3>
                    <button class="text-gray-400 hover:text-gray-600 modal-close" aria-label="Close modal">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="p-6">${content}</div>
                ${options.footer ? `<div class="flex justify-end gap-3 p-6 border-t bg-gray-50">${options.footer}</div>` : ''}
            </div>
        `;

        modalContainer.appendChild(modal);

        // Bind close events
        modal.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay') || e.target.classList.contains('modal-close')) {
                this.closeModal(modal);
            }
        });

        // Focus management
        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusableElements.length > 0) {
            focusableElements[0].focus();
        }

        return modal;
    }

    closeModal(modal) {
        if (modal) {
            modal.style.animation = 'modalFadeOut 0.3s ease-out';
            setTimeout(() => modal.remove(), 300);
        }
    }

    // ===== ERROR HANDLING =====
    async handleAsyncOperation(operation, options = {}) {
        const {
            loadingMessage = 'Processing...',
            successMessage = 'Operation completed successfully',
            errorMessage = 'Operation failed'
        } = options;

        this.showLoading(loadingMessage);

        try {
            const result = await operation();
            if (successMessage) {
                this.showToast(successMessage, 'success');
            }
            return result;
        } catch (error) {
            console.error('Operation error:', error);
            this.showToast(error.message || errorMessage, 'error');
            throw error;
        } finally {
            this.hideLoading();
        }
    }

    // ===== FORM VALIDATION =====
    validateForm(form) {
        const inputs = form.querySelectorAll('input, select, textarea');
        let isValid = true;
        const errors = [];

        inputs.forEach(input => {
            // Remove existing error styling
            input.classList.remove('border-red-500');
            const errorEl = input.parentElement.querySelector('.field-error');
            if (errorEl) errorEl.remove();

            // Check required fields
            if (input.hasAttribute('required') && !input.value.trim()) {
                this.showFieldError(input, 'This field is required');
                isValid = false;
                errors.push(`${input.name || input.id} is required`);
            }

            // Email validation
            if (input.type === 'email' && input.value && !this.isValidEmail(input.value)) {
                this.showFieldError(input, 'Please enter a valid email address');
                isValid = false;
                errors.push('Invalid email format');
            }

            // URL validation
            if (input.type === 'url' && input.value && !this.isValidUrl(input.value)) {
                this.showFieldError(input, 'Please enter a valid URL');
                isValid = false;
                errors.push('Invalid URL format');
            }
        });

        return { isValid, errors };
    }

    showFieldError(input, message) {
        input.classList.add('border-red-500');

        const errorEl = document.createElement('p');
        errorEl.className = 'field-error text-red-600 text-sm mt-1';
        errorEl.textContent = message;

        input.parentElement.appendChild(errorEl);
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    }

    // ===== UTILITY FUNCTIONS =====
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    formatDate(date) {
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }).format(new Date(date));
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // ===== ACCESSIBILITY =====
    announceToScreenReader(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.style.position = 'absolute';
        announcement.style.left = '-10000px';
        announcement.style.width = '1px';
        announcement.style.height = '1px';
        announcement.style.overflow = 'hidden';

        announcement.textContent = message;
        document.body.appendChild(announcement);

        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }

    // ===== THEME SUPPORT =====
    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }

    getTheme() {
        return localStorage.getItem('theme') || 'light';
    }

    toggleTheme() {
        const currentTheme = this.getTheme();
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.uiUtils = new UIUtils();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIUtils;
}
