/**
 * WebSocket Client for Real-time Streaming
 */
class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.listeners = {
            'frame': [],
            'connected': [],
            'error': [],
            'close': []
        };
    }

    /**
     * Connect to WebSocket server
     * @param {number} cameraId 
     */
    connect(cameraId = 0) {
        if (this.ws) {
            this.ws.close();
        }

        const wsUrl = `${this.url}?camera_id=${cameraId}`;
        console.log(`Connecting to WebSocket: ${wsUrl}`);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.emit('connected', { cameraId });
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'frame') {
                    this.emit('frame', data);
                } else if (data.type === 'connected') {
                    console.log('Server confirmed connection:', data.message);
                } else if (data.type === 'error') {
                    this.emit('error', data.message);
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('error', 'Connection error');
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            this.emit('close', event);

            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                setTimeout(() => {
                    console.log(`Reconnecting... Attempt ${this.reconnectAttempts + 1}`);
                    this.reconnectAttempts++;
                    this.connect(cameraId);
                }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
            }
        };
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Add event listener
     * @param {string} event 
     * @param {Function} callback 
     */
    on(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event].push(callback);
        }
    }

    /**
     * Remove event listener
     * @param {string} event 
     * @param {Function} callback 
     */
    off(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        }
    }

    /**
     * Emit event
     * @param {string} event 
     * @param {any} data 
     */
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }
}
