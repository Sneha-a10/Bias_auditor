"""
Simple MCP Server for Bias Auditor.

Provides bias pattern resources via a simple HTTP API
that follows MCP-like patterns.
"""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

from resources import (
    get_bias_pattern,
    get_fairness_metric,
    get_mitigation_strategy,
    get_all_bias_patterns,
    get_all_fairness_metrics,
    get_all_mitigation_strategies,
    get_strategies_for_stage,
    get_patterns_for_stage
)


class MCPHandler(BaseHTTPRequestHandler):
    """HTTP handler for MCP server."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parse_qs(parsed_path.query)
        
        try:
            if path == "/bias_patterns":
                # Get all bias patterns or filter by stage
                stage = query.get("stage", [None])[0]
                if stage:
                    data = get_patterns_for_stage(stage.upper())
                else:
                    data = get_all_bias_patterns()
                self._send_json_response(data)
            
            elif path.startswith("/bias_pattern/"):
                # Get specific bias pattern
                pattern_id = path.split("/")[-1]
                data = get_bias_pattern(pattern_id)
                if data:
                    self._send_json_response(data)
                else:
                    self._send_error(404, f"Bias pattern '{pattern_id}' not found")
            
            elif path == "/fairness_metrics":
                # Get all fairness metrics
                data = get_all_fairness_metrics()
                self._send_json_response(data)
            
            elif path.startswith("/fairness_metric/"):
                # Get specific fairness metric
                metric_id = path.split("/")[-1]
                data = get_fairness_metric(metric_id)
                if data:
                    self._send_json_response(data)
                else:
                    self._send_error(404, f"Fairness metric '{metric_id}' not found")
            
            elif path == "/mitigation_strategies":
                # Get all mitigation strategies or filter by stage
                stage = query.get("stage", [None])[0]
                if stage:
                    data = get_strategies_for_stage(stage.upper())
                else:
                    data = get_all_mitigation_strategies()
                self._send_json_response(data)
            
            elif path.startswith("/mitigation_strategy/"):
                # Get specific mitigation strategy
                strategy_id = path.split("/")[-1]
                data = get_mitigation_strategy(strategy_id)
                if data:
                    self._send_json_response(data)
                else:
                    self._send_error(404, f"Mitigation strategy '{strategy_id}' not found")
            
            elif path == "/health":
                # Health check
                self._send_json_response({"status": "healthy"})
            
            else:
                self._send_error(404, "Endpoint not found")
        
        except Exception as e:
            self._send_error(500, str(e))
    
    def _send_json_response(self, data: Any):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        error_data = {"error": message, "code": code}
        self.wfile.write(json.dumps(error_data).encode())
    
    def log_message(self, format, *args):
        """Custom log message format."""
        print(f"[MCP Server] {format % args}")


def run_server(host: str = "localhost", port: int = 8001):
    """
    Run the MCP server.
    
    Args:
        host: Server host
        port: Server port
    """
    server_address = (host, port)
    httpd = HTTPServer(server_address, MCPHandler)
    
    print(f"MCP Server running on http://{host}:{port}")
    print("Available endpoints:")
    print("  GET /bias_patterns")
    print("  GET /bias_pattern/{id}")
    print("  GET /fairness_metrics")
    print("  GET /fairness_metric/{id}")
    print("  GET /mitigation_strategies")
    print("  GET /mitigation_strategy/{id}")
    print("  GET /health")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
