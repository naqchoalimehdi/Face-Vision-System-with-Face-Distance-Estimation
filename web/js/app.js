/**
 * Main Application Logic
 */
class App {
    constructor() {
        // Initialize components
        this.api = new ApiClient();

        // Determine WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        this.ws = new WebSocketClient(wsUrl);

        this.renderer = new Renderer('video-canvas');

        this.controls = new Controls(this.api, {
            onStart: (cameraId) => this.startStreaming(cameraId),
            onStop: () => this.stopStreaming(),
            onConfigUpdate: (config) => this.updateConfig(config),
            onDisplayUpdate: (settings) => this.renderer.updateSettings(settings)
        });

        // State
        this.activeCameraId = null;
        this.isRunning = false;

        // UI Elements
        this.fpsDisplay = document.getElementById('fps-display');
        this.faceCountDisplay = document.getElementById('face-count');
        this.statusDisplay = document.getElementById('status-display');
        this.frameCountDisplay = document.getElementById('frame-count');
        this.videoOverlay = document.getElementById('video-overlay');

        this.init();
    }

    async init() {
        await this.controls.loadCameras();
        this.setupWebSocketListeners();
    }

    setupWebSocketListeners() {
        this.ws.on('connected', () => {
            this.statusDisplay.textContent = 'Streaming';
            this.statusDisplay.style.color = 'var(--success)';
            this.videoOverlay.classList.add('hidden');
        });

        this.ws.on('frame', (data) => {
            this.renderer.render(data.frame, data.detections);
            this.updateStats(data);
        });

        this.ws.on('error', (error) => {
            this.statusDisplay.textContent = 'Error';
            this.statusDisplay.style.color = 'var(--error)';
            console.error('Stream error:', error);
        });

        this.ws.on('close', () => {
            if (this.isRunning) {
                this.statusDisplay.textContent = 'Reconnecting...';
                this.statusDisplay.style.color = 'var(--warning)';
            } else {
                this.statusDisplay.textContent = 'Stopped';
                this.statusDisplay.style.color = 'var(--text-muted)';
                this.videoOverlay.classList.remove('hidden');
            }
        });
    }

    async startStreaming(cameraId) {
        try {
            this.statusDisplay.textContent = 'Starting...';

            // Start camera on backend
            await this.api.startCamera(cameraId);

            // Connect WebSocket
            this.ws.connect(cameraId);

            this.activeCameraId = cameraId;
            this.isRunning = true;

        } catch (error) {
            console.error('Failed to start streaming:', error);
            this.statusDisplay.textContent = 'Failed to start';
            this.statusDisplay.style.color = 'var(--error)';
            this.controls.updateButtonState(false);
        }
    }

    async stopStreaming() {
        try {
            this.isRunning = false;
            this.ws.disconnect();

            if (this.activeCameraId !== null) {
                await this.api.stopCamera(this.activeCameraId);
            }

            this.activeCameraId = null;
            this.renderer.clear();
            this.resetStats();

        } catch (error) {
            console.error('Error stopping stream:', error);
        }
    }

    async updateConfig(config) {
        if (this.activeCameraId !== null) {
            try {
                // Map frontend config to backend config
                const backendConfig = {};
                if (config.confidence_threshold !== undefined) {
                    backendConfig.confidence_threshold = config.confidence_threshold;
                }

                if (Object.keys(backendConfig).length > 0) {
                    await this.api.updateCameraConfig(this.activeCameraId, backendConfig);
                }
            } catch (error) {
                console.error('Failed to update config:', error);
            }
        }
    }

    updateStats(data) {
        this.fpsDisplay.textContent = data.fps.toFixed(1);
        this.faceCountDisplay.textContent = data.detections.length;
        this.frameCountDisplay.textContent = data.frame_count;
    }

    resetStats() {
        this.fpsDisplay.textContent = '0';
        this.faceCountDisplay.textContent = '0';
        this.frameCountDisplay.textContent = '0';
    }
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
