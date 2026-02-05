"""
SHAP Explainer with Gemini Natural Language Integration
Provides regulatory-compliant model interpretability for SR 11-7 compliance
"""
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Any
import asyncio
import os
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for PPNR predictions
    Compliant with Fed SR 11-7 Model Risk Management requirements
    """
    
    def __init__(self, xgboost_model=None):
        self.xgb_model = xgboost_model
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def initialize_explainer(self, X_background: np.ndarray, feature_names: List[str]):
        """Initialize SHAP explainer with background data"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model must be set before initialization")
            
        self.feature_names = feature_names
        
        # Use TreeExplainer for XGBoost (fast and exact)
        self.explainer = shap.TreeExplainer(self.xgb_model.model)
        
        logger.info("SHAP explainer initialized")
        
    def explain_predictions(self, X: np.ndarray) -> Dict:
        """Generate SHAP explanations for a set of predictions"""
        if self.explainer is None:
            raise ValueError("Explainer must be initialized first.")
        
        # Ensure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # ============== DEBUGGING CODE ==============
        print(f"DEBUG: X_static shape: {X.shape}")
        print(f"DEBUG: Expected features: {self.xgb_model.scaler.n_features_in_}")
        print(f"DEBUG: Model feature names: {self.xgb_model.feature_names}")
        print(f"DEBUG: Number of feature names: {len(self.xgb_model.feature_names) if self.xgb_model.feature_names else 'None'}")
        
        if X.shape[1] != self.xgb_model.scaler.n_features_in_:
            print(f"ERROR: Shape mismatch detected!")
            print(f"  - Provided: {X.shape[1]} features")
            print(f"  - Expected: {self.xgb_model.scaler.n_features_in_} features")
        # ============================================
        
        # CRITICAL FIX: Use the wrapper's preprocess method instead of calling scaler directly
        try:
            X_scaled = self.xgb_model.preprocess(X)
            # Add logic here to return explanations as intended
            shap_values = self.explainer.shap_values(X_scaled)
            return {"shap_values": shap_values}
        except ValueError as e:
            logger.error(f"Feature mismatch: {e}")
            raise ValueError(f"Feature shape mismatch. Model expects {self.xgb_model.scaler.n_features_in_} features.")
        
    def explain_single_prediction(self, X_single: np.ndarray, 
                                 prediction: float) -> Dict:
        """
        Generate detailed explanation for a single prediction
        Used for scenario-specific analysis
        """
        if self.explainer is None:
            raise ValueError("Explainer must be initialized first")
            
        X_scaled = self.xgb_model.scaler.transform(X_single.reshape(1, -1))
        shap_values = self.explainer.shap_values(X_scaled)[0]
        
        # Create contribution breakdown
        contributions = {}
        for i, (name, value) in enumerate(zip(self.feature_names, shap_values)):
            contributions[name] = {
                'shap_value': float(value),
                'feature_value': float(X_single[i]),
                'contribution_pct': float(abs(value) / (abs(shap_values).sum() + 1e-10) * 100)
            }
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        ))
        
        return {
            'prediction': prediction,
            'expected_value': self.explainer.expected_value,
            'contributions': sorted_contributions,
            'top_positive_drivers': [
                k for k, v in sorted_contributions.items() 
                if v['shap_value'] > 0
            ][:3],
            'top_negative_drivers': [
                k for k, v in sorted_contributions.items() 
                if v['shap_value'] < 0
            ][:3]
        }
    
    def explain_scenario_impact(self, baseline_X: np.ndarray,
                                stress_X: np.ndarray,
                                baseline_pred: float,
                                stress_pred: float) -> Dict:
        """
        Explain the difference between baseline and stress predictions
        Identifies primary drivers of capital breach
        """
        baseline_scaled = self.xgb_model.scaler.transform(baseline_X.reshape(1, -1))
        stress_scaled = self.xgb_model.scaler.transform(stress_X.reshape(1, -1))
        
        baseline_shap = self.explainer.shap_values(baseline_scaled)[0]
        stress_shap = self.explainer.shap_values(stress_scaled)[0]
        
        # Calculate change in SHAP values
        shap_delta = stress_shap - baseline_shap
        
        impact_breakdown = {}
        for i, name in enumerate(self.feature_names):
            impact_breakdown[name] = {
                'baseline_contribution': float(baseline_shap[i]),
                'stress_contribution': float(stress_shap[i]),
                'delta': float(shap_delta[i]),
                'impact_direction': 'negative' if shap_delta[i] < 0 else 'positive'
            }
        
        # Identify primary driver
        sorted_by_impact = sorted(
            impact_breakdown.items(),
            key=lambda x: abs(x[1]['delta']),
            reverse=True
        )
        
        primary_driver = sorted_by_impact[0][0] if sorted_by_impact else None
        
        return {
            'baseline_prediction': baseline_pred,
            'stress_prediction': stress_pred,
            'prediction_change': stress_pred - baseline_pred,
            'prediction_change_pct': (stress_pred - baseline_pred) / (abs(baseline_pred) + 1e-10) * 100,
            'impact_breakdown': dict(sorted_by_impact),
            'primary_driver': primary_driver,
            'primary_driver_impact': sorted_by_impact[0][1]['delta'] if sorted_by_impact else 0
        }


class GeminiExplainer:
    """
    Natural language explanation generator using Gemini 3 Flash
    Provides human-readable interpretations of SHAP results
    """
    
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        self.chat = None
        
    async def _get_chat(self):
        """Lazy initialize the chat client"""
        if self.chat is None:
            from emergentintegrations.llm.chat import LlmChat
            self.chat = LlmChat(
                api_key=self.api_key,
                session_id="shap-explainer",
                system_message="""You are a regulatory compliance expert specializing in bank stress testing 
                and capital adequacy. You explain complex model outputs in clear, professional language 
                suitable for Fed examiners and bank risk officers. Focus on:
                1. What the key drivers are
                2. Why they matter for capital adequacy
                3. Regulatory implications
                Keep explanations concise but informative."""
            ).with_model("gemini", "gemini-3-flash-preview")
        return self.chat
    
    async def generate_global_explanation(self, shap_results: Dict, 
                                          scenario_name: str) -> str:
        """Generate natural language explanation of global feature importance"""
        
        chat = await self._get_chat()
        
        # Format importance data
        importance_str = "\n".join([
            f"- {name}: {value:.4f}" 
            for name, value in list(shap_results['global_importance'].items())[:8]
        ])
        
        from emergentintegrations.llm.chat import UserMessage
        
        prompt = f"""Analyze this SHAP feature importance from a PPNR stress test model 
        under the "{scenario_name}" scenario:
        
        Feature Importance (mean |SHAP|):
        {importance_str}
        
        Provide a 3-4 sentence explanation of:
        1. Which macro factors drive PPNR losses most
        2. What this means for bank capital planning
        3. Key risk concentration to monitor"""
        
        try:
            response = await chat.send_message(UserMessage(text=prompt))
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_explanation(shap_results, scenario_name)
    
    async def generate_scenario_explanation(self, scenario_impact: Dict,
                                            scenario_name: str) -> str:
        """Generate explanation of scenario-specific impact"""
        
        chat = await self._get_chat()
        
        from emergentintegrations.llm.chat import UserMessage
        
        prompt = f"""Explain this stress test result comparison for a bank:
        
        Scenario: {scenario_name}
        Baseline PPNR: ${scenario_impact['baseline_prediction']*1e9/1e6:.1f}M
        Stressed PPNR: ${scenario_impact['stress_prediction']*1e9/1e6:.1f}M
        Change: {scenario_impact['prediction_change_pct']:.1f}%
        
        Primary Driver: {scenario_impact['primary_driver']}
        Driver Impact: {scenario_impact['primary_driver_impact']:.4f}
        
        Top factors causing PPNR decline:
        {', '.join([f"{k} ({v['delta']:.4f})" for k, v in list(scenario_impact['impact_breakdown'].items())[:5]])}
        
        Provide a 3-4 sentence regulatory explanation:
        1. What's driving the capital erosion
        2. Is this CRE crash, yield curve, or other factors
        3. Supervisory implications"""
        
        try:
            response = await chat.send_message(UserMessage(text=prompt))
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_scenario_explanation(scenario_impact, scenario_name)
    
    async def generate_breach_alert_explanation(self, breach_analysis: Dict,
                                                trajectory_summary: Dict) -> str:
        """Generate explanation when capital breach is detected"""
        
        chat = await self._get_chat()
        
        from emergentintegrations.llm.chat import UserMessage
        
        prompt = f"""Generate a regulatory alert for this capital breach:
        
        Banks breaching CET1 minimum: {breach_analysis['banks_with_breach']} of {breach_analysis['total_banks']}
        First breach quarter: Q{breach_analysis['first_breach_quarter']}
        Worst case: Bank {breach_analysis['worst_breach']['bank_id']} at {breach_analysis['worst_breach']['min_cet1']*100:.1f}% CET1
        Capital shortfall: ${abs(breach_analysis['capital_shortfall'])/1e9:.1f}B
        
        Provide a concise alert message that:
        1. Summarizes the severity
        2. Identifies the timeline
        3. Suggests immediate actions
        4. References SR 11-7 compliance needs"""
        
        try:
            response = await chat.send_message(UserMessage(text=prompt))
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_breach_explanation(breach_analysis)
    
    def _generate_fallback_explanation(self, shap_results: Dict, scenario_name: str) -> str:
        """Fallback when API unavailable"""
        top_features = list(shap_results['global_importance'].keys())[:3]
        return f"""Under the {scenario_name} scenario, the primary drivers of PPNR variation are: 
        {', '.join(top_features)}. Banks with higher exposure to these factors face 
        elevated capital risk and should monitor their stress buffers closely."""
    
    def _generate_fallback_scenario_explanation(self, scenario_impact: Dict, scenario_name: str) -> str:
        """Fallback scenario explanation"""
        driver = scenario_impact['primary_driver']
        change = scenario_impact['prediction_change_pct']
        return f"""The {scenario_name} scenario projects a {abs(change):.1f}% decline in PPNR, 
        primarily driven by {driver}. This indicates potential capital erosion 
        that warrants enhanced monitoring under SR 11-7 guidelines."""
    
    def _generate_fallback_breach_explanation(self, breach_analysis: Dict) -> str:
        """Fallback breach explanation"""
        return f"""CAPITAL BREACH ALERT: {breach_analysis['banks_with_breach']} banks 
        projected to fall below CET1 minimums by Q{breach_analysis['first_breach_quarter']}. 
        Immediate supervisory review recommended per SR 11-7."""
