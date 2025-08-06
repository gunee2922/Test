// Enhanced EOS Floating Tool JavaScript

class EOSFloatingTool {
    constructor() {
        this.markers = [];
        this.currentMarker = null;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.settings = this.loadSettings();
        this.contextMenu = null;
        this.init();
    }

    init() {
        this.createElements();
        this.bindEvents();
        this.applySettings();
        this.createInitialMarker();
        this.showWelcome();
    }

    createElements() {
        // Get DOM elements
        this.eosMarker = document.getElementById('eosMarker');
        this.settingsPanel = document.getElementById('settingsPanel');
        this.controlBar = document.getElementById('controlBar');
        this.contextMenu = document.getElementById('contextMenu');
        this.helpOverlay = document.getElementById('helpOverlay');
        
        // Control elements
        this.colorPicker = document.getElementById('colorPicker');
        this.sizeSlider = document.getElementById('sizeSlider');
        this.fontSelect = document.getElementById('fontSelect');
        this.opacitySlider = document.getElementById('opacitySlider');
        this.snapToEdges = document.getElementById('snapToEdges');
        this.autoPosition = document.getElementById('autoPosition');
        this.enablePulse = document.getElementById('enablePulse');
        this.enableBounce = document.getElementById('enableBounce');
        this.enableFade = document.getElementById('enableFade');
        this.markerText = document.getElementById('markerText');
    }

    bindEvents() {
        // Marker events
        this.eosMarker.addEventListener('mousedown', this.onMarkerMouseDown.bind(this));
        this.eosMarker.addEventListener('contextmenu', this.onMarkerContextMenu.bind(this));
        this.eosMarker.addEventListener('dblclick', this.onMarkerDoubleClick.bind(this));
        
        // Global events
        document.addEventListener('mousemove', this.onMouseMove.bind(this));
        document.addEventListener('mouseup', this.onMouseUp.bind(this));
        document.addEventListener('keydown', this.onKeyDown.bind(this));
        document.addEventListener('click', this.onDocumentClick.bind(this));
        
        // Control button events
        document.getElementById('toggleVisibility').addEventListener('click', this.toggleVisibility.bind(this));
        document.getElementById('openSettings').addEventListener('click', this.openSettings.bind(this));
        document.getElementById('toggleControls').addEventListener('click', this.toggleControls.bind(this));
        document.getElementById('closeSettings').addEventListener('click', this.closeSettings.bind(this));
        
        // Settings events
        this.colorPicker.addEventListener('input', this.onColorChange.bind(this));
        this.sizeSlider.addEventListener('input', this.onSizeChange.bind(this));
        this.fontSelect.addEventListener('change', this.onFontChange.bind(this));
        this.opacitySlider.addEventListener('input', this.onOpacityChange.bind(this));
        this.snapToEdges.addEventListener('change', this.onSnapChange.bind(this));
        this.autoPosition.addEventListener('change', this.onAutoPositionChange.bind(this));
        this.enablePulse.addEventListener('change', this.onPulseChange.bind(this));
        this.enableBounce.addEventListener('change', this.onBounceChange.bind(this));
        this.enableFade.addEventListener('change', this.onFadeChange.bind(this));
        
        // Action button events
        document.getElementById('addMarker').addEventListener('click', this.addMarker.bind(this));
        document.getElementById('removeMarker').addEventListener('click', this.removeMarker.bind(this));
        document.getElementById('centerMarker').addEventListener('click', this.centerMarker.bind(this));
        document.getElementById('exportSettings').addEventListener('click', this.exportSettings.bind(this));
        document.getElementById('importSettings').addEventListener('click', this.importSettings.bind(this));
        document.getElementById('resetSettings').addEventListener('click', this.resetSettings.bind(this));
        
        // Context menu events
        this.contextMenu.addEventListener('click', this.onContextMenuClick.bind(this));
        
        // Help events
        document.getElementById('closeHelp').addEventListener('click', this.closeHelp.bind(this));
        
        // Text editing events
        document.getElementById('eosText').addEventListener('blur', this.onTextChange.bind(this));
        document.getElementById('eosText').addEventListener('keydown', this.onTextKeyDown.bind(this));
        
        // Import file events
        document.getElementById('importFile').addEventListener('change', this.onFileImport.bind(this));
        
        // Window events
        window.addEventListener('resize', this.onWindowResize.bind(this));
        window.addEventListener('beforeunload', this.saveSettings.bind(this));
    }

    createInitialMarker() {
        const marker = {
            id: Date.now(),
            element: this.eosMarker,
            text: this.settings.defaultText || 'EOS',
            x: this.settings.position?.x || 50,
            y: this.settings.position?.y || 50
        };
        
        this.markers.push(marker);
        this.currentMarker = marker;
        this.updateMarkerPosition(marker);
        this.updateMarkerText(marker);
    }

    onMarkerMouseDown(e) {
        if (e.button !== 0) return; // Only left click
        
        e.preventDefault();
        this.isDragging = true;
        this.currentMarker = this.getMarkerFromElement(e.target.closest('.eos-marker'));
        
        const rect = this.currentMarker.element.getBoundingClientRect();
        this.dragOffset.x = e.clientX - rect.left;
        this.dragOffset.y = e.clientY - rect.top;
        
        this.currentMarker.element.classList.add('dragging');
        document.body.style.userSelect = 'none';
        
        if (this.settings.enableBounce) {
            this.currentMarker.element.classList.add('bounce');
        }
    }

    onMouseMove(e) {
        if (!this.isDragging || !this.currentMarker) return;
        
        const x = e.clientX - this.dragOffset.x;
        const y = e.clientY - this.dragOffset.y;
        
        this.currentMarker.x = x;
        this.currentMarker.y = y;
        
        if (this.settings.snapToEdges) {
            this.snapToEdges(this.currentMarker);
        }
        
        this.updateMarkerPosition(this.currentMarker);
        
        if (this.settings.autoPosition) {
            this.avoidContentOverlap(this.currentMarker);
        }
    }

    onMouseUp() {
        if (!this.isDragging) return;
        
        this.isDragging = false;
        document.body.style.userSelect = '';
        
        if (this.currentMarker) {
            this.currentMarker.element.classList.remove('dragging');
            
            if (this.settings.enableBounce) {
                setTimeout(() => {
                    this.currentMarker.element.classList.remove('bounce');
                }, 600);
            }
            
            this.savePosition(this.currentMarker);
        }
    }

    onMarkerContextMenu(e) {
        e.preventDefault();
        this.currentMarker = this.getMarkerFromElement(e.target.closest('.eos-marker'));
        this.showContextMenu(e.clientX, e.clientY);
    }

    onMarkerDoubleClick(e) {
        e.preventDefault();
        this.currentMarker = this.getMarkerFromElement(e.target.closest('.eos-marker'));
        const textElement = this.currentMarker.element.querySelector('[contenteditable]');
        textElement.focus();
        this.selectAllText(textElement);
    }

    onKeyDown(e) {
        // Keyboard shortcuts
        if (e.altKey) {
            switch (e.key.toLowerCase()) {
                case 'v':
                    e.preventDefault();
                    this.toggleVisibility();
                    break;
                case 's':
                    e.preventDefault();
                    this.openSettings();
                    break;
                case 'h':
                    e.preventDefault();
                    this.toggleControls();
                    break;
            }
        }
        
        if (e.key === 'Escape') {
            this.closeAllPanels();
        }
        
        if (e.key === 'F1') {
            e.preventDefault();
            this.toggleHelp();
        }
    }

    onDocumentClick(e) {
        // Close context menu if clicking outside
        if (!this.contextMenu.contains(e.target)) {
            this.hideContextMenu();
        }
    }

    onColorChange() {
        this.settings.color = this.colorPicker.value;
        this.applyColorToMarkers();
        this.saveSettings();
    }

    onSizeChange() {
        this.settings.size = parseInt(this.sizeSlider.value);
        document.getElementById('sizeValue').textContent = this.settings.size + 'px';
        this.applySizeToMarkers();
        this.saveSettings();
    }

    onFontChange() {
        this.settings.font = this.fontSelect.value;
        this.applyFontToMarkers();
        this.saveSettings();
    }

    onOpacityChange() {
        this.settings.opacity = parseFloat(this.opacitySlider.value);
        document.getElementById('opacityValue').textContent = Math.round(this.settings.opacity * 100) + '%';
        this.applyOpacityToMarkers();
        this.saveSettings();
    }

    onSnapChange() {
        this.settings.snapToEdges = this.snapToEdges.checked;
        this.saveSettings();
    }

    onAutoPositionChange() {
        this.settings.autoPosition = this.autoPosition.checked;
        this.saveSettings();
    }

    onPulseChange() {
        this.settings.enablePulse = this.enablePulse.checked;
        this.applyPulseToMarkers();
        this.saveSettings();
    }

    onBounceChange() {
        this.settings.enableBounce = this.enableBounce.checked;
        this.saveSettings();
    }

    onFadeChange() {
        this.settings.enableFade = this.enableFade.checked;
        this.saveSettings();
    }

    onTextChange(e) {
        if (this.currentMarker) {
            this.currentMarker.text = e.target.textContent;
            this.saveSettings();
        }
    }

    onTextKeyDown(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            e.target.blur();
        }
    }

    onWindowResize() {
        this.markers.forEach(marker => {
            this.constrainToViewport(marker);
        });
    }

    toggleVisibility() {
        const isVisible = this.markers.some(marker => marker.element.style.display !== 'none');
        
        this.markers.forEach(marker => {
            if (isVisible) {
                if (this.settings.enableFade) {
                    marker.element.classList.add('fade-out');
                    setTimeout(() => {
                        marker.element.style.display = 'none';
                        marker.element.classList.remove('fade-out');
                    }, 300);
                } else {
                    marker.element.style.display = 'none';
                }
            } else {
                marker.element.style.display = 'block';
                if (this.settings.enableFade) {
                    marker.element.classList.add('fade-in');
                    setTimeout(() => {
                        marker.element.classList.remove('fade-in');
                    }, 500);
                }
            }
        });
    }

    openSettings() {
        this.settingsPanel.classList.add('open');
    }

    closeSettings() {
        this.settingsPanel.classList.remove('open');
    }

    toggleControls() {
        this.controlBar.classList.toggle('hidden');
    }

    addMarker() {
        const newMarker = this.createMarkerElement();
        this.markers.push(newMarker);
        this.currentMarker = newMarker;
        this.saveSettings();
    }

    removeMarker() {
        if (this.markers.length <= 1) {
            this.showNotification('Cannot remove the last marker', 'warning');
            return;
        }
        
        const lastMarker = this.markers[this.markers.length - 1];
        if (lastMarker.element !== this.eosMarker) {
            document.body.removeChild(lastMarker.element);
        } else {
            lastMarker.element.style.display = 'none';
        }
        
        this.markers.pop();
        this.currentMarker = this.markers[this.markers.length - 1];
        this.saveSettings();
    }

    centerMarker() {
        if (!this.currentMarker) return;
        
        const centerX = (window.innerWidth - this.currentMarker.element.offsetWidth) / 2;
        const centerY = (window.innerHeight - this.currentMarker.element.offsetHeight) / 2;
        
        this.currentMarker.x = centerX;
        this.currentMarker.y = centerY;
        
        this.updateMarkerPosition(this.currentMarker);
        this.savePosition(this.currentMarker);
    }

    exportSettings() {
        const settingsData = {
            ...this.settings,
            markers: this.markers.map(marker => ({
                text: marker.text,
                x: marker.x,
                y: marker.y
            }))
        };
        
        const blob = new Blob([JSON.stringify(settingsData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'eos-settings.json';
        a.click();
        
        URL.revokeObjectURL(url);
        this.showNotification('Settings exported successfully', 'success');
    }

    importSettings() {
        document.getElementById('importFile').click();
    }

    onFileImport(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const importedSettings = JSON.parse(event.target.result);
                this.settings = { ...this.getDefaultSettings(), ...importedSettings };
                this.applySettings();
                this.saveSettings();
                this.showNotification('Settings imported successfully', 'success');
            } catch (error) {
                this.showNotification('Invalid settings file', 'error');
            }
        };
        reader.readAsText(file);
    }

    resetSettings() {
        if (confirm('Are you sure you want to reset all settings to default?')) {
            this.settings = this.getDefaultSettings();
            this.applySettings();
            this.saveSettings();
            this.showNotification('Settings reset to default', 'success');
        }
    }

    showContextMenu(x, y) {
        this.contextMenu.style.left = x + 'px';
        this.contextMenu.style.top = y + 'px';
        this.contextMenu.classList.add('show');
        
        // Adjust position if menu goes off screen
        setTimeout(() => {
            const rect = this.contextMenu.getBoundingClientRect();
            if (rect.right > window.innerWidth) {
                this.contextMenu.style.left = (x - rect.width) + 'px';
            }
            if (rect.bottom > window.innerHeight) {
                this.contextMenu.style.top = (y - rect.height) + 'px';
            }
        }, 0);
    }

    hideContextMenu() {
        this.contextMenu.classList.remove('show');
    }

    onContextMenuClick(e) {
        const action = e.target.dataset.action;
        if (!action) return;
        
        switch (action) {
            case 'edit':
                this.onMarkerDoubleClick({ target: this.currentMarker.element });
                break;
            case 'clone':
                this.cloneMarker(this.currentMarker);
                break;
            case 'settings':
                this.openSettings();
                break;
            case 'center':
                this.centerMarker();
                break;
            case 'remove':
                this.removeSpecificMarker(this.currentMarker);
                break;
        }
        
        this.hideContextMenu();
    }

    cloneMarker(marker) {
        const newMarker = this.createMarkerElement();
        newMarker.text = marker.text;
        newMarker.x = marker.x + 20;
        newMarker.y = marker.y + 20;
        
        this.updateMarkerPosition(newMarker);
        this.updateMarkerText(newMarker);
        this.markers.push(newMarker);
        this.saveSettings();
    }

    removeSpecificMarker(marker) {
        if (this.markers.length <= 1) {
            this.showNotification('Cannot remove the last marker', 'warning');
            return;
        }
        
        const index = this.markers.indexOf(marker);
        if (index !== -1) {
            if (marker.element !== this.eosMarker) {
                document.body.removeChild(marker.element);
            } else {
                marker.element.style.display = 'none';
            }
            
            this.markers.splice(index, 1);
            this.currentMarker = this.markers[0];
            this.saveSettings();
        }
    }

    createMarkerElement() {
        const marker = document.createElement('div');
        marker.className = 'eos-marker';
        marker.draggable = true;
        marker.tabIndex = 0;
        marker.setAttribute('role', 'button');
        marker.setAttribute('aria-label', 'Floating EOS marker');
        
        const textSpan = document.createElement('span');
        textSpan.contentEditable = true;
        textSpan.spellcheck = false;
        textSpan.textContent = this.markerText.value || 'EOS';
        
        marker.appendChild(textSpan);
        document.body.appendChild(marker);
        
        // Bind events for new marker
        marker.addEventListener('mousedown', this.onMarkerMouseDown.bind(this));
        marker.addEventListener('contextmenu', this.onMarkerContextMenu.bind(this));
        marker.addEventListener('dblclick', this.onMarkerDoubleClick.bind(this));
        textSpan.addEventListener('blur', this.onTextChange.bind(this));
        textSpan.addEventListener('keydown', this.onTextKeyDown.bind(this));
        
        const newMarkerData = {
            id: Date.now(),
            element: marker,
            text: textSpan.textContent,
            x: 100,
            y: 100
        };
        
        this.applyStylesToMarker(newMarkerData);
        
        return newMarkerData;
    }

    snapToEdges(marker) {
        const snapDistance = 20;
        const rect = marker.element.getBoundingClientRect();
        
        // Snap to left edge
        if (marker.x < snapDistance) {
            marker.x = 0;
        }
        
        // Snap to right edge
        if (marker.x + rect.width > window.innerWidth - snapDistance) {
            marker.x = window.innerWidth - rect.width;
        }
        
        // Snap to top edge
        if (marker.y < snapDistance) {
            marker.y = 0;
        }
        
        // Snap to bottom edge
        if (marker.y + rect.height > window.innerHeight - snapDistance) {
            marker.y = window.innerHeight - rect.height;
        }
        
        marker.element.classList.add('snapped');
        setTimeout(() => {
            marker.element.classList.remove('snapped');
        }, 200);
    }

    avoidContentOverlap(marker) {
        // Simple implementation - avoid overlapping with settings panel
        const settingsRect = this.settingsPanel.getBoundingClientRect();
        const markerRect = marker.element.getBoundingClientRect();
        
        if (this.settingsPanel.classList.contains('open')) {
            if (marker.x + markerRect.width > settingsRect.left && marker.x < settingsRect.right &&
                marker.y + markerRect.height > settingsRect.top && marker.y < settingsRect.bottom) {
                marker.x = settingsRect.left - markerRect.width - 10;
            }
        }
    }

    constrainToViewport(marker) {
        const rect = marker.element.getBoundingClientRect();
        
        if (marker.x + rect.width > window.innerWidth) {
            marker.x = window.innerWidth - rect.width;
        }
        
        if (marker.y + rect.height > window.innerHeight) {
            marker.y = window.innerHeight - rect.height;
        }
        
        if (marker.x < 0) marker.x = 0;
        if (marker.y < 0) marker.y = 0;
        
        this.updateMarkerPosition(marker);
    }

    updateMarkerPosition(marker) {
        marker.element.style.left = marker.x + 'px';
        marker.element.style.top = marker.y + 'px';
    }

    updateMarkerText(marker) {
        const textElement = marker.element.querySelector('[contenteditable]');
        textElement.textContent = marker.text;
    }

    applyStylesToMarker(marker) {
        marker.element.style.backgroundColor = this.settings.color;
        marker.element.style.fontSize = this.settings.size + 'px';
        marker.element.style.fontFamily = this.settings.font;
        marker.element.style.opacity = this.settings.opacity;
        
        if (this.settings.enablePulse) {
            marker.element.classList.add('pulse');
        } else {
            marker.element.classList.remove('pulse');
        }
    }

    applySettings() {
        // Update UI controls
        this.colorPicker.value = this.settings.color;
        this.sizeSlider.value = this.settings.size;
        this.fontSelect.value = this.settings.font;
        this.opacitySlider.value = this.settings.opacity;
        this.snapToEdges.checked = this.settings.snapToEdges;
        this.autoPosition.checked = this.settings.autoPosition;
        this.enablePulse.checked = this.settings.enablePulse;
        this.enableBounce.checked = this.settings.enableBounce;
        this.enableFade.checked = this.settings.enableFade;
        this.markerText.value = this.settings.defaultText;
        
        // Update value displays
        document.getElementById('sizeValue').textContent = this.settings.size + 'px';
        document.getElementById('opacityValue').textContent = Math.round(this.settings.opacity * 100) + '%';
        
        // Apply to all markers
        this.markers.forEach(marker => {
            this.applyStylesToMarker(marker);
            if (this.settings.position && marker === this.markers[0]) {
                marker.x = this.settings.position.x;
                marker.y = this.settings.position.y;
                this.updateMarkerPosition(marker);
            }
        });
    }

    applyColorToMarkers() {
        this.markers.forEach(marker => {
            marker.element.style.backgroundColor = this.settings.color;
        });
    }

    applySizeToMarkers() {
        this.markers.forEach(marker => {
            marker.element.style.fontSize = this.settings.size + 'px';
        });
    }

    applyFontToMarkers() {
        this.markers.forEach(marker => {
            marker.element.style.fontFamily = this.settings.font;
        });
    }

    applyOpacityToMarkers() {
        this.markers.forEach(marker => {
            marker.element.style.opacity = this.settings.opacity;
        });
    }

    applyPulseToMarkers() {
        this.markers.forEach(marker => {
            if (this.settings.enablePulse) {
                marker.element.classList.add('pulse');
            } else {
                marker.element.classList.remove('pulse');
            }
        });
    }

    getMarkerFromElement(element) {
        return this.markers.find(marker => marker.element === element);
    }

    savePosition(marker) {
        if (marker === this.markers[0]) {
            this.settings.position = { x: marker.x, y: marker.y };
            this.saveSettings();
        }
    }

    selectAllText(element) {
        const range = document.createRange();
        range.selectNodeContents(element);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    }

    closeAllPanels() {
        this.closeSettings();
        this.hideContextMenu();
        this.closeHelp();
    }

    toggleHelp() {
        this.helpOverlay.classList.toggle('show');
    }

    closeHelp() {
        this.helpOverlay.classList.remove('show');
    }

    showWelcome() {
        setTimeout(() => {
            if (!localStorage.getItem('eosToolWelcomeShown')) {
                this.toggleHelp();
                localStorage.setItem('eosToolWelcomeShown', 'true');
            }
        }, 1000);
    }

    showNotification(message, type = 'info') {
        // Simple notification system
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#27ae60' : type === 'warning' ? '#f39c12' : type === 'error' ? '#e74c3c' : '#3498db'};
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 10003;
            animation: slideIn 0.3s ease;
            max-width: 300px;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    getDefaultSettings() {
        return {
            color: '#3498db',
            size: 24,
            font: 'Inter',
            opacity: 0.9,
            snapToEdges: false,
            autoPosition: false,
            enablePulse: false,
            enableBounce: true,
            enableFade: true,
            defaultText: 'EOS',
            position: { x: 50, y: 50 }
        };
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('eosFloatingToolSettings');
            return saved ? { ...this.getDefaultSettings(), ...JSON.parse(saved) } : this.getDefaultSettings();
        } catch (error) {
            console.warn('Failed to load settings:', error);
            return this.getDefaultSettings();
        }
    }

    saveSettings() {
        try {
            localStorage.setItem('eosFloatingToolSettings', JSON.stringify(this.settings));
        } catch (error) {
            console.warn('Failed to save settings:', error);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.eosApp = new EOSFloatingTool();
});

// Add notification styles
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(notificationStyles);