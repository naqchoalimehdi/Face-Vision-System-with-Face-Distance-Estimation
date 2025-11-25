/**
 * UI Controls Manager
 */
class Controls {
    constructor(apiClient, callbacks) {
        this.api = apiClient;
        this.callbacks = callbacks;

        // Elements
        this.cameraSelect = document.getElementById('camera-select');
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.confidenceSlider = document.getElementById('confidence-slider');
        this.confidenceValue = document.getElementById('confidence-value');
        this.showLandmarks = document.getElementById('show-landmarks');
        this.showDistance = document.getElementById('show-distance');
        this.settingsBtn = document.getElementById('settings-btn');
        this.modal = document.getElementById('settings-modal');
        this.modalClose = document.getElementById('modal-close');
        this.modalCancel = document.getElementById('modal-cancel');
        this.modalSave = document.getElementById('modal-save');

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Camera selection
        this.cameraSelect.addEventListener('change', () => {
            const cameraId = this.cameraSelect.value;
            if (cameraId !== '') {
                this.startBtn.disabled = false;
            } else {
                this.startBtn.disabled = true;
            }
        });

        // Start/Stop buttons
        this.startBtn.addEventListener('click', () => {
            const cameraId = parseInt(this.cameraSelect.value);
            this.callbacks.onStart(cameraId);
            this.updateButtonState(true);
        });

        this.stopBtn.addEventListener('click', () => {
            this.callbacks.onStop();
            this.updateButtonState(false);
        });

        // Confidence slider
        this.confidenceSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            this.confidenceValue.textContent = parseFloat(value).toFixed(2);
            this.callbacks.onConfigUpdate({ confidence_threshold: parseFloat(value) });
        });

        // Checkboxes
        this.showLandmarks.addEventListener('change', (e) => {
            this.callbacks.onDisplayUpdate({ showLandmarks: e.target.checked });
        });

        this.showDistance.addEventListener('change', (e) => {
            this.callbacks.onDisplayUpdate({ showDistance: e.target.checked });
        });

        // Modal controls
        this.settingsBtn.addEventListener('click', () => {
            this.modal.classList.add('active');
        });

        const closeModal = () => {
            this.modal.classList.remove('active');
        };

        this.modalClose.addEventListener('click', closeModal);
        this.modalCancel.addEventListener('click', closeModal);

        this.modalSave.addEventListener('click', () => {
            const config = {
                distanceMethod: document.getElementById('distance-method').value,
                faceWidth: parseFloat(document.getElementById('face-width').value),
                maxFaces: parseInt(document.getElementById('max-faces').value)
            };
            this.callbacks.onConfigUpdate(config);
            closeModal();
        });

        // Close modal on outside click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                closeModal();
            }
        });
    }

    /**
     * Populate camera list
     */
    async loadCameras() {
        try {
            const cameras = await this.api.listCameras();
            this.cameraSelect.innerHTML = '<option value="">Select a camera...</option>';

            cameras.forEach(cam => {
                const option = document.createElement('option');
                option.value = cam.camera_id;
                option.textContent = `${cam.name} (${cam.width}x${cam.height} @ ${cam.fps}fps)`;
                this.cameraSelect.appendChild(option);
            });

            if (cameras.length > 0) {
                this.cameraSelect.value = cameras[0].camera_id;
                this.startBtn.disabled = false;
            }
        } catch (error) {
            console.error('Failed to load cameras:', error);
            this.cameraSelect.innerHTML = '<option value="">Error loading cameras</option>';
        }
    }

    /**
     * Update button states
     * @param {boolean} isRunning 
     */
    updateButtonState(isRunning) {
        this.startBtn.disabled = isRunning;
        this.stopBtn.disabled = !isRunning;
        this.cameraSelect.disabled = isRunning;
    }
}
