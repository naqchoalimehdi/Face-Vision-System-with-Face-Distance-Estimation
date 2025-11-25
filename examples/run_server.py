"""Run the Face Vision System server."""

import uvicorn
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import get_config


def main():
    """Run the server."""
    # Load config
    config = get_config()
    api_config = config.get_section('api')
    
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    print(f"Starting Face Vision System server on {host}:{port}")
    print(f"Web UI: http://localhost:{port}")
    
    # Run server
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
