"""
Custom tools framework for Bias Auditor.

Provides base classes and utilities for creating reusable tools
that can be used by agents for bias detection and analysis.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Base class for tool inputs."""
    pass


class ToolOutput(BaseModel):
    """Base class for tool outputs."""
    success: bool = Field(description="Whether the tool execution succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class Tool(ABC):
    """
    Base class for custom tools.
    
    All tools must implement:
    - name: Unique tool identifier
    - description: What the tool does
    - input_schema: Pydantic model for inputs
    - output_schema: Pydantic model for outputs
    - _run: Actual tool logic
    """
    
    name: str
    description: str
    input_schema: Type[ToolInput]
    output_schema: Type[ToolOutput]
    
    def __init__(self):
        """Initialize tool."""
        if not hasattr(self, 'name'):
            raise ValueError("Tool must define 'name' attribute")
        if not hasattr(self, 'description'):
            raise ValueError("Tool must define 'description' attribute")
        if not hasattr(self, 'input_schema'):
            raise ValueError("Tool must define 'input_schema' attribute")
        if not hasattr(self, 'output_schema'):
            raise ValueError("Tool must define 'output_schema' attribute")
    
    @abstractmethod
    def _run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool logic.
        
        Args:
            **kwargs: Tool-specific inputs
        
        Returns:
            Dictionary matching output_schema
        """
        pass
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the tool with validation.
        
        Args:
            **kwargs: Tool inputs
        
        Returns:
            Validated output dictionary
        """
        try:
            # Validate inputs
            validated_input = self.input_schema(**kwargs)
            
            # Execute tool
            result = self._run(**validated_input.model_dump())
            
            # Validate outputs
            validated_output = self.output_schema(**result)
            
            return validated_output.model_dump()
        
        except Exception as e:
            return self.output_schema(
                success=False,
                error=f"{type(e).__name__}: {str(e)}"
            ).model_dump()
    
    def __str__(self) -> str:
        return f"Tool({self.name}): {self.description}"


class ToolRegistry:
    """Registry for managing available tools."""
    
    _tools: Dict[str, Tool] = {}
    
    @classmethod
    def register(cls, tool: Tool):
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        cls._tools[tool.name] = tool
    
    @classmethod
    def get(cls, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance or None
        """
        return cls._tools.get(name)
    
    @classmethod
    def list_tools(cls) -> Dict[str, str]:
        """
        List all registered tools.
        
        Returns:
            Dictionary of tool names to descriptions
        """
        return {name: tool.description for name, tool in cls._tools.items()}
