"""
MCP Client for Bias Auditor.

Connects to the MCP server to fetch bias patterns,
fairness metrics, and mitigation strategies.
"""
import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))
MCP_BASE_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}"


class MCPClient:
    """Client for connecting to MCP bias pattern server."""
    
    def __init__(self, base_url: str = MCP_BASE_URL):
        """
        Initialize MCP client.
        
        Args:
            base_url: Base URL of MCP server
        """
        self.base_url = base_url
    
    def get_bias_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get bias pattern by ID.
        
        Args:
            pattern_id: Pattern identifier
        
        Returns:
            Pattern data or None
        """
        try:
            response = requests.get(f"{self.base_url}/bias_pattern/{pattern_id}", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_all_bias_patterns(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all bias patterns, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter (DATA, FEATURE, MODEL)
        
        Returns:
            Dictionary of bias patterns
        """
        try:
            url = f"{self.base_url}/bias_patterns"
            if stage:
                url += f"?stage={stage}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
    
    def get_fairness_metric(self, metric_id: str) -> Optional[Dict[str, Any]]:
        """
        Get fairness metric definition.
        
        Args:
            metric_id: Metric identifier
        
        Returns:
            Metric definition or None
        """
        try:
            response = requests.get(f"{self.base_url}/fairness_metric/{metric_id}", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_all_fairness_metrics(self) -> Dict[str, Any]:
        """
        Get all fairness metric definitions.
        
        Returns:
            Dictionary of fairness metrics
        """
        try:
            response = requests.get(f"{self.base_url}/fairness_metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
    
    def get_mitigation_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get mitigation strategy.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Strategy data or None
        """
        try:
            response = requests.get(f"{self.base_url}/mitigation_strategy/{strategy_id}", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_mitigation_strategies(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get mitigation strategies, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter (DATA, FEATURE, MODEL)
        
        Returns:
            Dictionary or list of mitigation strategies
        """
        try:
            url = f"{self.base_url}/mitigation_strategies"
            if stage:
                url += f"?stage={stage}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
    
    def is_available(self) -> bool:
        """
        Check if MCP server is available.
        
        Returns:
            True if server is reachable
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False


# Global client instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """
    Get or create MCP client instance.
    
    Returns:
        MCPClient instance
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client
