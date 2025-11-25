/**
 * API Client for Face Vision System
 */
class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    /**
     * List available cameras
     * @returns {Promise<Array>} List of cameras
     */
    async listCameras() {
        try {
            const response = await fetch(`${this.baseUrl}/api/cameras`);
            if (!response.ok) throw new Error('Failed to list cameras');
            return await response.json();
        } catch (error) {
            console.error('Error listing cameras:', error);
            throw error;
        }
    }

    /**
     * Start camera stream
     * @param {number} cameraId 
     * @returns {Promise<Object>} Response
     */
    async startCamera(cameraId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/cameras/${cameraId}/start`, {
                method: 'POST'
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start camera');
            }
            return await response.json();
        } catch (error) {
            console.error(`Error starting camera ${cameraId}:`, error);
            throw error;
        }
    }

    /**
     * Stop camera stream
     * @param {number} cameraId 
     * @returns {Promise<Object>} Response
     */
    async stopCamera(cameraId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/cameras/${cameraId}/stop`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to stop camera');
            return await response.json();
        } catch (error) {
            console.error(`Error stopping camera ${cameraId}:`, error);
            throw error;
        }
    }

    /**
     * Get camera configuration
     * @param {number} cameraId 
     * @returns {Promise<Object>} Configuration
     */
    async getCameraConfig(cameraId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/cameras/${cameraId}/config`);
            if (!response.ok) throw new Error('Failed to get camera config');
            return await response.json();
        } catch (error) {
            console.error('Error getting camera config:', error);
            throw error;
        }
    }

    /**
     * Update camera configuration
     * @param {number} cameraId 
     * @param {Object} config 
     * @returns {Promise<Object>} Response
     */
    async updateCameraConfig(cameraId, config) {
        try {
            const response = await fetch(`${this.baseUrl}/api/cameras/${cameraId}/config`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            if (!response.ok) throw new Error('Failed to update camera config');
            return await response.json();
        } catch (error) {
            console.error('Error updating camera config:', error);
            throw error;
        }
    }

    /**
     * Start calibration
     * @returns {Promise<Object>} Response
     */
    async startCalibration() {
        try {
            const response = await fetch(`${this.baseUrl}/api/calibration/start`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to start calibration');
            return await response.json();
        } catch (error) {
            console.error('Error starting calibration:', error);
            throw error;
        }
    }

    /**
     * Compute calibration
     * @param {string} cameraId 
     * @returns {Promise<Object>} Calibration result
     */
    async computeCalibration(cameraId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/calibration/compute?camera_id=${cameraId}`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to compute calibration');
            return await response.json();
        } catch (error) {
            console.error('Error computing calibration:', error);
            throw error;
        }
    }

    /**
     * Get calibration status
     * @returns {Promise<Object>} Status
     */
    async getCalibrationStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/api/calibration/status`);
            if (!response.ok) throw new Error('Failed to get calibration status');
            return await response.json();
        } catch (error) {
            console.error('Error getting calibration status:', error);
            throw error;
        }
    }
}
