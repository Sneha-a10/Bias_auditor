"""
Gemini-powered bias explanation agent.

Uses Google's Gemini API to generate intelligent, context-aware
explanations of bias patterns and actionable recommendations.
"""
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class GeminiAgent:
    """
    Gemini-powered agent for bias analysis and recommendations.
    
    Provides:
    - Enhanced bias explanations
    - Context-aware recommendations
    - Natural language analysis
    """
    
    def __init__(self, model_name: str = GEMINI_MODEL):
        """
        Initialize Gemini agent.
        
        Args:
            model_name: Gemini model to use
        """
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def generate_bias_explanation(
        self,
        primary_origin: str,
        checkpoint_summary: List[Dict[str, Any]],
        data_bias: Dict[str, Any],
        feature_bias: Dict[str, Any],
        model_bias: Dict[str, Any]
    ) -> str:
        """
        Generate enhanced bias explanation using Gemini.
        
        Args:
            primary_origin: Primary bias origin (DATA/FEATURE/MODEL)
            checkpoint_summary: Summary of all checkpoints
            data_bias: Data bias analysis results
            feature_bias: Feature bias analysis results
            model_bias: Model bias analysis results
        
        Returns:
            Enhanced explanation string
        """
        prompt = self._build_explanation_prompt(
            primary_origin, checkpoint_summary, data_bias, feature_bias, model_bias
        )
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback to basic explanation
            return f"Bias primarily originates in the {primary_origin} stage. " \
                   f"Error generating detailed explanation: {str(e)}"
    
    def generate_recommendations(
        self,
        checkpoint_summary: List[Dict[str, Any]],
        primary_origin: str,
        data_bias: Dict[str, Any],
        feature_bias: Dict[str, Any],
        model_bias: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Generate actionable recommendations using Gemini.
        
        Args:
            checkpoint_summary: Summary of all checkpoints
            primary_origin: Primary bias origin
            data_bias: Data bias analysis results
            feature_bias: Feature bias analysis results
            model_bias: Model bias analysis results
        
        Returns:
            Dictionary of recommendations by stage
        """
        prompt = self._build_recommendations_prompt(
            checkpoint_summary, primary_origin, data_bias, feature_bias, model_bias
        )
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_recommendations(response.text)
        except Exception as e:
            # Fallback to empty recommendations
            return {
                "Data": [f"Error generating recommendations: {str(e)}"],
                "Features": [],
                "Model": []
            }
    
    def _build_explanation_prompt(
        self,
        primary_origin: str,
        checkpoint_summary: List[Dict[str, Any]],
        data_bias: Dict[str, Any],
        feature_bias: Dict[str, Any],
        model_bias: Dict[str, Any]
    ) -> str:
        """Build prompt for bias explanation."""
        prompt = f"""You are an AI fairness expert analyzing bias in a machine learning pipeline.

**Primary Bias Origin:** {primary_origin}

**Checkpoint Summary:**
"""
        for checkpoint in checkpoint_summary:
            prompt += f"\n- **{checkpoint['stage']}**: {checkpoint['status']} (score: {checkpoint['score']:.2f})"
            if checkpoint['flagged_issues']:
                prompt += f"\n  Issues: {', '.join(checkpoint['flagged_issues'])}"
        
        prompt += f"""

**Detailed Analysis:**

**Data Checkpoint:**
- Bias Score: {data_bias['overall']['data_bias_score']:.2f}
- Summary: {data_bias['overall']['summary']}

**Feature Checkpoint:**
- Bias Score: {feature_bias['flags']['feature_bias_score']:.2f}
- Summary: {feature_bias['flags']['summary']}

**Model Checkpoint:**
- Bias Score: {model_bias['flags']['model_bias_score']:.2f}
- Summary: {model_bias['flags']['summary']}

**Task:**
Generate a clear, concise explanation (2-3 sentences) of:
1. Why {primary_origin} is identified as the primary bias origin
2. How bias at this stage affects downstream components
3. The key evidence supporting this conclusion

Keep the explanation accessible to both technical and non-technical stakeholders.
"""
        return prompt
    
    def _build_recommendations_prompt(
        self,
        checkpoint_summary: List[Dict[str, Any]],
        primary_origin: str,
        data_bias: Dict[str, Any],
        feature_bias: Dict[str, Any],
        model_bias: Dict[str, Any]
    ) -> str:
        """Build prompt for recommendations."""
        prompt = f"""You are an AI fairness expert providing actionable recommendations to mitigate bias.

**Primary Bias Origin:** {primary_origin}

**Detected Issues:**
"""
        for checkpoint in checkpoint_summary:
            if checkpoint['flagged_issues']:
                prompt += f"\n**{checkpoint['stage']}:**\n"
                for issue in checkpoint['flagged_issues']:
                    prompt += f"- {issue}\n"
        
        prompt += """

**Task:**
Generate specific, actionable recommendations organized by stage (Data, Features, Model).
For each recommendation:
1. Be concrete and specific (not generic advice)
2. Prioritize based on the primary bias origin
3. Include implementation details where possible
4. Focus on the most impactful fixes first

Format your response as:

**Data:**
- [Specific recommendation 1]
- [Specific recommendation 2]

**Features:**
- [Specific recommendation 1]

**Model:**
- [Specific recommendation 1]

Provide 1-3 recommendations per stage, focusing on the stages with detected issues.
"""
        return prompt
    
    def _parse_recommendations(self, response_text: str) -> Dict[str, List[str]]:
        """Parse Gemini response into structured recommendations."""
        recommendations = {
            "Data": [],
            "Features": [],
            "Model": []
        }
        
        current_stage = None
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for stage headers
            if line.startswith("**Data:**") or line.lower().startswith("data:"):
                current_stage = "Data"
            elif line.startswith("**Features:**") or line.lower().startswith("features:"):
                current_stage = "Features"
            elif line.startswith("**Model:**") or line.lower().startswith("model:"):
                current_stage = "Model"
            # Check for bullet points
            elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
                if current_stage:
                    # Remove bullet point and clean up
                    rec = line.lstrip("-•* ").strip()
                    if rec:
                        recommendations[current_stage].append(rec)
        
        return recommendations
    
    def analyze_bias_pattern(
        self,
        bias_type: str,
        evidence: Dict[str, Any]
    ) -> str:
        """
        Analyze a specific bias pattern.
        
        Args:
            bias_type: Type of bias (e.g., "demographic_parity_violation")
            evidence: Evidence data
        
        Returns:
            Analysis text
        """
        prompt = f"""Analyze this bias pattern:

**Bias Type:** {bias_type}

**Evidence:**
{evidence}

Provide a brief (1-2 sentences) explanation of:
1. What this bias pattern means
2. Why it's problematic
3. The typical root cause
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing bias pattern: {str(e)}"


def is_gemini_available() -> bool:
    """
    Check if Gemini API is available.
    
    Returns:
        True if API key is configured
    """
    return GEMINI_API_KEY is not None and GEMINI_API_KEY != ""


def get_gemini_agent() -> Optional[GeminiAgent]:
    """
    Get Gemini agent instance if available.
    
    Returns:
        GeminiAgent instance or None
    """
    if is_gemini_available():
        try:
            return GeminiAgent()
        except Exception:
            return None
    return None
