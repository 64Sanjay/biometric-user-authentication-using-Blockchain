// Main JavaScript for Biometric Authentication System

// API Base URL
const API_BASE = '';

// Utility Functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// API Functions
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error checking status:', error);
        return { success: false, error: error.message };
    }
}

async function enrollUser(userId, vaultData) {
    try {
        const response = await fetch(`${API_BASE}/api/enroll`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, vault_data: vaultData })
        });
        return await response.json();
    } catch (error) {
        console.error('Error enrolling user:', error);
        return { success: false, error: error.message };
    }
}

async function authenticateUser(userId) {
    try {
        const response = await fetch(`${API_BASE}/api/authenticate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId })
        });
        return await response.json();
    } catch (error) {
        console.error('Error authenticating user:', error);
        return { success: false, error: error.message };
    }
}

async function revokeVault(userId, vaultIndex) {
    try {
        const response = await fetch(`${API_BASE}/api/revoke`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, vault_index: vaultIndex })
        });
        return await response.json();
    } catch (error) {
        console.error('Error revoking vault:', error);
        return { success: false, error: error.message };
    }
}

// Status Indicator Update
async function updateStatusIndicator() {
    const statusElement = document.getElementById('system-status');
    if (!statusElement) return;
    
    const status = await checkSystemStatus();
    
    if (status.success && status.data.system_ready) {
        statusElement.innerHTML = '<span class="badge bg-success">System Online</span>';
    } else {
        statusElement.innerHTML = '<span class="badge bg-danger">System Offline</span>';
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Biometric Authentication System Loaded');
    
    // Update status indicator if exists
    updateStatusIndicator();
    
    // Add form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
});

// Export functions for use in templates
window.BiometricAuth = {
    checkSystemStatus,
    enrollUser,
    authenticateUser,
    revokeVault,
    showAlert
};
