/**
 * Professional Agent Builder UI Enhancements
 * Adds modern UX features, error handling, and user feedback
 */

class AgentBuilderEnhancements {
    constructor() {
        this.init();
    }

    init() {
        this.addLoadingStates();
        this.addToastNotifications();
        this.addModalSystem();
        this.addKeyboardShortcuts();
        this.addAccessibilityFeatures();
        this.addRealTimeValidation();
    }

    addLoadingStates() {
        // Add loading overlay functionality
        this.createLoadingOverlay();

        // Hook into existing agentBuilder events
        if (window.agentBuilder) {
            const originalLoadComponents = window.agentBuilder.loadComponents;
            window.agentBuilder.loadComponents = async () => {
                this.showLoading('Loading components...');
                try {
                    await originalLoadComponents.call(window.agentBuilder);
                    this.showToast('Components loaded successfully', 'success');
                } catch (error) {
                    console.error('Error loading components:', error);
                    this.showToast('Failed to load components', 'error');
                } finally {
                    this.hideLoading();
                }
            };
        }
    }

    createLoadingOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="bg-white rounded-lg shadow-xl p-6 flex items-center gap-4">
                <div class="loading-spinner"></div>
                <div class="text-gray-900 font-medium" id="loading-message">Loading...</div>
            </div>
        `;
        document.body.appendChild(overlay);
    }

    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            document.getElementById('loading-message').textContent = message;
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    addToastNotifications() {
        this.toastContainer = document.createElement('div');
        this.toastContainer.id = 'toast-container';
        this.toastContainer.className = 'fixed top-4 right-4 z-50 space-y-2';
        document.body.appendChild(this.toastContainer);
    }

    showToast(message, type = 'info', duration = 3000) {
        if (!this.toastContainer) this.addToastNotifications();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type} max-w-sm`;
        toast.innerHTML = `
            <div class="flex items-center gap-3">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
                <button class="ml-auto text-white/70 hover:text-white" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        this.toastContainer.appendChild(toast);

        // Auto remove after duration
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
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

    addModalSystem() {
        this.modalContainer = document.createElement('div');
        this.modalContainer.id = 'modal-container';
        document.body.appendChild(this.modalContainer);
    }

    showModal(title, content, options = {}) {
        if (!this.modalContainer) this.addModalSystem();

        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content max-w-2xl w-full mx-4">
                <div class="flex items-center justify-between p-6 border-b">
                    <h3 class="text-lg font-semibold text-gray-900">${title}</h3>
                    <button class="text-gray-400 hover:text-gray-600 modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="p-6">${content}</div>
                ${options.footer ? `<div class="flex justify-end gap-3 p-6 border-t bg-gray-50">${options.footer}</div>` : ''}
            </div>
        `;

        this.modalContainer.appendChild(modal);

        // Bind close events
        modal.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay') || e.target.classList.contains('modal-close')) {
                modal.remove();
            }
        });
    }

    addKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + S: Save workflow
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.handleSave();
            }

            // Ctrl/Cmd + Z: Undo
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                this.handleUndo();
            }

            // Ctrl/Cmd + Y or Ctrl/Cmd + Shift + Z: Redo
            if (((e.ctrlKey || e.metaKey) && e.key === 'y') ||
                ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z')) {
                e.preventDefault();
                this.handleRedo();
            }

            // Delete: Remove selected component
            if (e.key === 'Delete' || e.key === 'Backspace') {
                this.handleDelete();
            }

            // Escape: Deselect all
            if (e.key === 'Escape') {
                this.deselectAll();
            }
        });
    }

    handleSave() {
        this.showToast('Saving workflow...', 'info');
        // Trigger save functionality
        if (window.agentBuilder && window.agentBuilder.saveWorkflow) {
            window.agentBuilder.saveWorkflow();
        }
    }

    handleUndo() {
        if (window.agentBuilder && window.agentBuilder.undo) {
            window.agentBuilder.undo();
            this.showToast('Action undone', 'info');
        }
    }

    handleRedo() {
        if (window.agentBuilder && window.agentBuilder.redo) {
            window.agentBuilder.redo();
            this.showToast('Action redone', 'info');
        }
    }

    handleDelete() {
        if (window.agentBuilder && window.agentBuilder.deleteSelected) {
            window.agentBuilder.deleteSelected();
            this.showToast('Component deleted', 'warning');
        }
    }

    deselectAll() {
        if (window.agentBuilder && window.agentBuilder.deselectAll) {
            window.agentBuilder.deselectAll();
        }
    }

    addAccessibilityFeatures() {
        // Add ARIA labels and roles
        document.querySelectorAll('.component').forEach(component => {
            component.setAttribute('role', 'button');
            component.setAttribute('tabindex', '0');
            component.setAttribute('aria-label', 'Draggable component');
        });

        // Add focus management for modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                const modal = document.querySelector('.modal-content');
                if (modal) {
                    const focusableElements = modal.querySelectorAll(
                        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                    );
                    const firstElement = focusableElements[0];
                    const lastElement = focusableElements[focusableElements.length - 1];

                    if (e.shiftKey) {
                        if (document.activeElement === firstElement) {
                            lastElement.focus();
                            e.preventDefault();
                        }
                    } else {
                        if (document.activeElement === lastElement) {
                            firstElement.focus();
                            e.preventDefault();
                        }
                    }
                }
            }
        });
    }

    addRealTimeValidation() {
        // Add real-time validation for component connections
        if (window.agentBuilder) {
            const originalConnect = window.agentBuilder.connect || (() => {});
            window.agentBuilder.connect = (source, target) => {
                try {
                    const result = originalConnect.call(window.agentBuilder, source, target);
                    this.validateConnection(source, target);
                    return result;
                } catch (error) {
                    this.showToast('Invalid connection: ' + error.message, 'error');
                    throw error;
                }
            };
        }
    }

    validateConnection(source, target) {
        // Basic validation - can be extended
        if (source === target) {
            throw new Error('Cannot connect component to itself');
        }

        // Check for circular dependencies
        if (this.detectCircularDependency(source, target)) {
            throw new Error('Circular dependency detected');
        }

        this.showToast('Connection created successfully', 'success');
    }

    detectCircularDependency(source, target) {
        // Simplified circular dependency check
        // In a real implementation, this would traverse the graph
        return false;
    }

    addContextMenu() {
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();

            const target = e.target.closest('.component');
            if (target) {
                this.showContextMenu(e, target);
            }
        });
    }

    showContextMenu(event, component) {
        const menu = document.createElement('div');
        menu.className = 'context-menu fixed bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50';
        menu.style.left = event.pageX + 'px';
        menu.style.top = event.pageY + 'px';

        menu.innerHTML = `
            <button class="context-menu-item w-full text-left px-4 py-2 hover:bg-gray-100 flex items-center gap-2">
                <i class="fas fa-copy"></i> Duplicate
            </button>
            <button class="context-menu-item w-full text-left px-4 py-2 hover:bg-gray-100 flex items-center gap-2">
                <i class="fas fa-trash"></i> Delete
            </button>
            <hr class="my-1">
            <button class="context-menu-item w-full text-left px-4 py-2 hover:bg-gray-100 flex items-center gap-2">
                <i class="fas fa-cog"></i> Properties
            </button>
        `;

        document.body.appendChild(menu);

        // Remove menu when clicking elsewhere
        const removeMenu = (e) => {
            if (!menu.contains(e.target)) {
                menu.remove();
                document.removeEventListener('click', removeMenu);
            }
        };

        setTimeout(() => document.addEventListener('click', removeMenu), 100);
    }

    addDragDropEnhancements() {
        // Enhanced drag feedback
        document.addEventListener('dragstart', (e) => {
            if (e.target.classList.contains('component')) {
                e.target.classList.add('dragging');
                this.showToast('Dragging component...', 'info');
            }
        });

        document.addEventListener('dragend', (e) => {
            if (e.target.classList.contains('component')) {
                e.target.classList.remove('dragging');
            }
        });

        document.addEventListener('drop', (e) => {
            this.showToast('Component placed successfully', 'success');
        });
    }
}

// Initialize enhancements when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.agentBuilderEnhancements = new AgentBuilderEnhancements();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AgentBuilderEnhancements;
}
