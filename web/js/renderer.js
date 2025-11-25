/**
 * Canvas Renderer for Video and Overlays
 */
class Renderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.image = new Image();

        // Configuration
        this.showLandmarks = true;
        this.showDistance = true;
        this.bboxColor = '#00ff00';
        this.textColor = '#ffffff';
        this.landmarkColor = '#ff0000';

        // Track colors for different IDs
        this.trackColors = [
            '#00ff00', '#ff0000', '#0000ff', '#ffff00',
            '#ff00ff', '#00ffff', '#8000ff', '#ff8000'
        ];
    }

    /**
     * Render frame and detections
     * @param {string} frameBase64 Base64 encoded image
     * @param {Array} detections List of detections
     */
    render(frameBase64, detections) {
        this.image.onload = () => {
            // Update canvas size to match image
            if (this.canvas.width !== this.image.width || this.canvas.height !== this.image.height) {
                this.canvas.width = this.image.width;
                this.canvas.height = this.image.height;
            }

            // Draw video frame
            this.ctx.drawImage(this.image, 0, 0);

            // Draw detections
            if (detections && detections.length > 0) {
                this.drawDetections(detections);
            }
        };

        this.image.src = `data:image/jpeg;base64,${frameBase64}`;
    }

    /**
     * Draw detections on canvas
     * @param {Array} detections 
     */
    drawDetections(detections) {
        detections.forEach(det => {
            const { bbox, track_id, confidence, distance_cm } = det;
            const color = track_id !== null
                ? this.trackColors[track_id % this.trackColors.length]
                : this.bboxColor;

            // Draw bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(
                bbox.x1,
                bbox.y1,
                bbox.x2 - bbox.x1,
                bbox.y2 - bbox.y1
            );

            // Draw label background
            const labelParts = [];
            if (track_id !== null) labelParts.push(`ID:${track_id}`);
            if (this.showDistance && distance_cm) labelParts.push(`${distance_cm.toFixed(1)}cm`);

            if (labelParts.length > 0) {
                const label = labelParts.join(' | ');
                this.ctx.font = '16px Inter, sans-serif';
                const textMetrics = this.ctx.measureText(label);
                const textHeight = 16;
                const padding = 4;

                this.ctx.fillStyle = color;
                this.ctx.fillRect(
                    bbox.x1,
                    bbox.y1 - textHeight - padding * 2,
                    textMetrics.width + padding * 2,
                    textHeight + padding * 2
                );

                // Draw label text
                this.ctx.fillStyle = '#000000';
                this.ctx.fillText(
                    label,
                    bbox.x1 + padding,
                    bbox.y1 - padding
                );
            }
        });
    }

    /**
     * Update display settings
     * @param {Object} settings 
     */
    updateSettings(settings) {
        if (settings.showLandmarks !== undefined) this.showLandmarks = settings.showLandmarks;
        if (settings.showDistance !== undefined) this.showDistance = settings.showDistance;
    }

    /**
     * Clear canvas
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}
